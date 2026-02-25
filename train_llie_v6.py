#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_llie_supervised_curve_only_with_lossplot.py

- Supervised LLIE training (paired low/normal) with:
  * NanoLLE_G backbone (Ghost+CA/SA+SNR-window MHSA) + Zero-DCE-style curve-only head
  * Loss: L1 + w_ssim*SSIM + w_lpips*LPIPS + w_color*CosineHSV
  * Iteration-based training + patch schedule
  * FULL-RES internal-val (PSNR/SSIM/LPIPS) + FULL-RES challenge-val inference zip
  * Debug strip saving:
      - train strip saved at eval_interval
      - infer strip (challenge low 1 sample) saved at eval_interval
      - (VAL strip 저장은 제거됨: user request)
  * Loss plot:
      - saved (OVERWRITE) every eval_interval
      - train_total + val_total on same figure with x-axis=iter
      - val_total computed only at eval_interval, plotted with its sparse x points
"""

import os
import re
import math
import glob
import time
import zipfile
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# OpenCV (for CLAHE feature)
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# tqdm (optional)
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# LPIPS (optional)  -> training loss용 (tensor)
try:
    import lpips  # pip install lpips
    _HAS_LPIPS = True
except Exception:
    lpips = None
    _HAS_LPIPS = False

# matplotlib (loss plot)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int = 1234):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def natural_key(path: str):
    base = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", base)]


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def pil_load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_tensor01(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


def tensor01_to_pil(x: torch.Tensor) -> Image.Image:
    arr = (x.permute(1, 2, 0).detach().cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)



# ----------------------------
# Input feature extraction (RGB | Gamma | CLAHE | Sobel)
# ----------------------------
def _clahe_enhance_rgb_uint8(
    img_rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
    add_l: int = 15,
    bilateral_d: int = 5,
    bilateral_sigmaColor: int = 15,
    bilateral_sigmaSpace: int = 15,
) -> np.ndarray:
    """
    img_rgb: HxWx3 uint8 (RGB)
    return: HxWx3 uint8 (RGB)
    """
    if not _HAS_CV2:
        raise RuntimeError("cv2 is required for CLAHE feature. Please install opencv-python.")
    if img_rgb.dtype != np.uint8:
        raise ValueError(f"img_rgb must be uint8, got {img_rgb.dtype}")
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"img_rgb must be HxWx3, got {img_rgb.shape}")

    # RGB -> BGR (OpenCV default)
    img_bgr = img_rgb[:, :, ::-1]

    # 1) Bilateral filter (noise suppression while preserving edges)
    denoised = cv2.bilateralFilter(
        img_bgr,
        d=int(bilateral_d),
        sigmaColor=float(bilateral_sigmaColor),
        sigmaSpace=float(bilateral_sigmaSpace),
    )

    # 2) BGR -> LAB
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 3) CLAHE on L channel
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
    cl = clahe.apply(l)

    # 4) Optional brightness lift
    if int(add_l) != 0:
        cl = cv2.add(cl, int(add_l))

    # 5) Merge back & LAB -> BGR -> RGB
    limg = cv2.merge((cl, a, b))
    out_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    out_rgb = out_bgr[:, :, ::-1]
    return out_rgb


def clahe_torch_batch(
    low_rgb01: torch.Tensor,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
    add_l: int = 15,
    bilateral_d: int = 5,
    bilateral_sigmaColor: int = 15,
    bilateral_sigmaSpace: int = 15,
) -> torch.Tensor:
    """
    low_rgb01: (B,3,H,W) float in [0,1] (CPU or CUDA)
    return:    (B,3,H,W) float in [0,1] (same device as input)
    """
    if low_rgb01.dim() != 4 or low_rgb01.size(1) != 3:
        raise ValueError(f"clahe_torch_batch expects (B,3,H,W), got {tuple(low_rgb01.shape)}")

    device = low_rgb01.device
    x = low_rgb01.detach()
    if x.is_cuda:
        x = x.cpu()

    # to uint8 RGB
    x_np = (x.clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255.0 + 0.5).astype(np.uint8)  # (B,H,W,3)

    outs = []
    for i in range(x_np.shape[0]):
        out_rgb = _clahe_enhance_rgb_uint8(
            x_np[i],
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
            add_l=add_l,
            bilateral_d=bilateral_d,
            bilateral_sigmaColor=bilateral_sigmaColor,
            bilateral_sigmaSpace=bilateral_sigmaSpace,
        )
        outs.append(out_rgb)

    out_np = np.stack(outs, axis=0)  # (B,H,W,3)
    out = torch.from_numpy(out_np).permute(0, 3, 1, 2).float() / 255.0  # (B,3,H,W)

    return out.to(device)


def sobel_edge_torch(
    low_rgb01: torch.Tensor,
    scale: float = 8.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Sobel edge (1-channel) computed from luma of RGB.

    low_rgb01: (B,3,H,W) in [0,1]
    Returns:   (B,1,H,W) in [0,1] (approximately; scaled by `scale`)
    """
    if low_rgb01.dim() != 4 or low_rgb01.size(1) != 3:
        raise ValueError(f"sobel_edge_torch expects (B,3,H,W), got {tuple(low_rgb01.shape)}")

    x = low_rgb01.clamp(0.0, 1.0)
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]  # (B,1,H,W)

    kx = torch.tensor(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
        device=x.device, dtype=x.dtype
    ).view(1, 1, 3, 3)

    ky = torch.tensor(
        [[-1.0, -2.0, -1.0],
         [ 0.0,  0.0,  0.0],
         [ 1.0,  2.0,  1.0]],
        device=x.device, dtype=x.dtype
    ).view(1, 1, 3, 3)

    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)

    # L1 magnitude (fast, stable): |gx| + |gy|. For inputs in [0,1],
    # this is in roughly [0, 8]. We divide by `scale` to map ~[0,1].
    denom = max(float(scale), float(eps))
    mag = (gx.abs() + gy.abs()) / denom
    return mag.clamp(0.0, 1.0)


def build_input_features(
    low_rgb01: torch.Tensor,
    gamma: float = 0.5,
    sobel_scale: float = 8.0,
    clahe_clip: float = 2.0,
    clahe_tile: int = 8,
    clahe_add: int = 15,
    clahe_bilateral_d: int = 5,
    clahe_bilateral_sigmaColor: int = 15,
    clahe_bilateral_sigmaSpace: int = 15,
) -> torch.Tensor:
    """
    Builds concatenated input:
        [RGB | Gamma(RGB) | CLAHE(RGB) | SobelEdge(Luma,1ch)]
    Returns: (B,10,H,W)
    """
    if low_rgb01.dim() != 4:
        raise ValueError(f"build_input_features expects (B,C,H,W), got {tuple(low_rgb01.shape)}")

    # If already concatenated, return as-is (supports resume/inference with precomputed features)
    if low_rgb01.size(1) == 10:
        return low_rgb01

    if low_rgb01.size(1) != 3:
        raise ValueError(f"Expected low image with 3 channels (RGB), got {low_rgb01.size(1)} channels")

    # Gamma correction (brighten): y = x^gamma, gamma < 1
    low_gamma = torch.pow(low_rgb01.clamp(0.0, 1.0), float(gamma)).clamp(0.0, 1.0)

    # CLAHE (OpenCV, CPU-based)
    low_clahe = clahe_torch_batch(
        low_rgb01,
        clip_limit=clahe_clip,
        tile_grid_size=clahe_tile,
        add_l=clahe_add,
        bilateral_d=clahe_bilateral_d,
        bilateral_sigmaColor=clahe_bilateral_sigmaColor,
        bilateral_sigmaSpace=clahe_bilateral_sigmaSpace,
    )

    # Sobel edge (1ch) from luma
    low_sobel = sobel_edge_torch(low_rgb01, scale=sobel_scale)

    return torch.cat([low_rgb01, low_gamma, low_clahe, low_sobel], dim=1)

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module) -> float:
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return float((param_size + buffer_size) / (1024 ** 2))


class EMATimer:
    def __init__(self, momentum: float = 0.95):
        self.m = momentum
        self.avg = None

    def update(self, v: float):
        if self.avg is None:
            self.avg = v
        else:
            self.avg = self.m * self.avg + (1 - self.m) * v

    def value(self):
        return float(self.avg) if self.avg is not None else 0.0


def format_seconds(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ----------------------------
# Loss plot (OVERWRITE at eval_interval)
# ----------------------------
def plot_train_val_losses(
    train_iters: List[int],
    train_losses: List[float],
    val_iters: List[int],
    val_losses: List[float],
    save_path: str,
    title: str = "Loss Curve (Train/Val, x=iter)",
    max_train_points: int = 6000,
):
    """
    x축 iter 기준으로 train/val loss를 한 그래프에 그림.
    - train: 길고 촘촘한 시퀀스 (iter마다 기록 가능)
    - val: eval_interval에서만 찍히는 sparse 시퀀스
    save_path는 항상 overwrite로 사용(같은 경로로 저장)
    """

    if len(train_iters) == 0:
        return

    ensure_dir(os.path.dirname(save_path) if os.path.dirname(save_path) else ".")

    # Optional decimation for fast plotting (keeps x-axis correct)
    if len(train_iters) > max_train_points:
        idx = np.linspace(0, len(train_iters) - 1, max_train_points).astype(np.int64)
        t_x = [train_iters[i] for i in idx]
        t_y = [train_losses[i] for i in idx]
    else:
        t_x, t_y = train_iters, train_losses

    plt.figure(figsize=(10, 4))
    plt.plot(t_x, t_y, label="train_total")

    if len(val_iters) > 0:
        plt.plot(val_iters, val_losses, marker="o", linestyle="-", label="val_total")

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ----------------------------
# Debug visualization helpers
# ----------------------------
def _to_3ch(x: torch.Tensor) -> torch.Tensor:
    """
    x: (C,H,W) in any range, float tensor
    return: (3,H,W)
    """
    if x.dim() != 3:
        raise ValueError(f"_to_3ch expects (C,H,W), got {tuple(x.shape)}")
    if x.size(0) == 1:
        return x.repeat(3, 1, 1)
    if x.size(0) == 3:
        return x
    return x[:3]


def vis_gain_map01(gain_map: torch.Tensor, max_gain: float = 4.0) -> torch.Tensor:
    """
    gain_map: (3,H,W) typically >= 1.0 (1 + softplus)
    visualize "how much gain above 1" as [0,1]
    """
    g = gain_map.detach().float()
    g = (g - 1.0) / max(1e-6, (max_gain - 1.0))
    return g


def vis_illum_map01(illum_map: torch.Tensor, max_gain: float = 4.0) -> torch.Tensor:
    """
    illum_map: (1,H,W) or (3,H,W), typically >= 1.0
    """
    m = illum_map.detach().float()
    if m.size(0) == 1:
        m = m.repeat(3, 1, 1)
    m = (m - 1.0) / max(1e-6, (max_gain - 1.0))
    return m


def tensor01_to_pil_rgb(x_chw01: torch.Tensor) -> Image.Image:
    """
    x_chw01: (C,H,W) in [0,1], C=1/3
    """
    x = _to_3ch(x_chw01)
    arr = (x.permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)


def _draw_label(im: Image.Image, text: str) -> Image.Image:
    """
    add small label at top-left
    """
    if text is None or text == "":
        return im
    im = im.copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    pad = 4
    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    draw.rectangle([0, 0, tw + 2 * pad, th + 2 * pad], fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return im


def save_concat_strip(
    input_rgb01: torch.Tensor,
    gain_map: torch.Tensor,
    illum_map: torch.Tensor,
    color_corr: torch.Tensor,
    pred_rgb01: torch.Tensor,
    save_path: str,
    max_gain_vis: float = 4.0,
    add_labels: bool = True,
):
    """
    All inputs are CHW tensors on any device.
    Saves a single horizontal strip:
      input | gain | illum | color_corr | pred
    """
    x = input_rgb01.detach().float().cpu()
    p = pred_rgb01.detach().float().cpu()

    gm = gain_map.detach().float().cpu()
    im = illum_map.detach().float().cpu()
    cc = color_corr.detach().float().cpu()

    H, W = x.shape[1], x.shape[2]

    def _resize_chw(t):
        if t.shape[1] != H or t.shape[2] != W:
            tt = t.unsqueeze(0)
            tt = F.interpolate(tt, size=(H, W), mode="bilinear", align_corners=False)
            return tt[0]
        return t

    gm = _resize_chw(_to_3ch(gm))
    im = _resize_chw(im if im.size(0) in (1, 3) else im[:1])
    cc = _resize_chw(_to_3ch(cc))
    p = _resize_chw(_to_3ch(p))
    x = _resize_chw(_to_3ch(x))

    gm_v = vis_gain_map01(gm, max_gain=max_gain_vis)
    im_v = vis_illum_map01(im if im.size(0) in (1, 3) else im[:1], max_gain=max_gain_vis)
    cc_v = vis_gain_map01(cc, max_gain=max_gain_vis)

    panels = [
        (tensor01_to_pil_rgb(x), "input"),
        (tensor01_to_pil_rgb(gm_v), "gain"),
        (tensor01_to_pil_rgb(im_v), "illum"),
        (tensor01_to_pil_rgb(cc_v), "color_corr"),
        (tensor01_to_pil_rgb(p), "pred"),
    ]
    if add_labels:
        panels = [(_draw_label(im_, lab), lab) for (im_, lab) in panels]

    ims = [im_ for (im_, _) in panels]
    widths = [im_.size[0] for im_ in ims]
    heights = [im_.size[1] for im_ in ims]
    out_w = sum(widths)
    out_h = max(heights)

    strip = Image.new("RGB", (out_w, out_h))
    xoff = 0
    for im_ in ims:
        strip.paste(im_, (xoff, 0))
        xoff += im_.size[0]

    ensure_dir(os.path.dirname(save_path) if os.path.dirname(save_path) else ".")
    strip.save(save_path)


# ----------------------------
# Metrics: PSNR / SSIM / LPIPS (eval용 float)
# ----------------------------
def psnr_torch(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-10) -> float:
    mse = F.mse_loss(pred, gt, reduction="mean").item()
    mse = max(mse, eps)
    return float(10.0 * math.log10(1.0 / mse))


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, device="cpu", dtype=torch.float32):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    k2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # (1,1,ws,ws)
    return k2d


def ssim_map_torch(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    device, dtype = pred.device, pred.dtype
    k = _gaussian_kernel(window_size, sigma, device=device, dtype=dtype).repeat(3, 1, 1, 1)

    mu1 = F.conv2d(pred, k, padding=window_size // 2, groups=3)
    mu2 = F.conv2d(gt, k, padding=window_size // 2, groups=3)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, k, padding=window_size // 2, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, k, padding=window_size // 2, groups=3) - mu2_sq
    sigma12 = F.conv2d(pred * gt, k, padding=window_size // 2, groups=3) - mu1_mu2

    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12
    )
    return ssim_map


def ssim_torch(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return float(ssim_map_torch(pred, gt).mean().detach().cpu().item())


class EvalLPIPS:
    """eval 때만 쓰는 float 반환 wrapper (no_grad)"""
    def __init__(self, net: str = "alex", device: str = "cuda"):
        if not _HAS_LPIPS:
            raise RuntimeError("lpips not installed. pip install lpips")
        self.fn = lpips.LPIPS(net=net).to(device).eval()
        for p in self.fn.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, pred01: torch.Tensor, gt01: torch.Tensor) -> float:
        p = pred01.clamp(0, 1) * 2 - 1
        g = gt01.clamp(0, 1) * 2 - 1
        return float(self.fn(p, g).mean().item())


def build_train_lpips(net: str, device: str):
    """
    training loss용 LPIPS (tensor 반환, grad는 pred로 흐름).
    """
    if not _HAS_LPIPS:
        return None
    fn = lpips.LPIPS(net=net).to(device).eval()
    for p in fn.parameters():
        p.requires_grad_(False)
    return fn


# ============================================================
# Color loss: Cosine HSV
# ============================================================
def rgb_to_hsv_torch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    maxc, _ = x.max(dim=1, keepdim=True)
    minc, _ = x.min(dim=1, keepdim=True)
    v = maxc
    delt = maxc - minc
    s = delt / (maxc + eps)

    delt_safe = delt + eps
    hr = ((g - b) / delt_safe) % 6.0
    hg = ((b - r) / delt_safe) + 2.0
    hb = ((r - g) / delt_safe) + 4.0

    is_r = (maxc == r).to(x.dtype)
    is_g = (maxc == g).to(x.dtype)
    is_b = (maxc == b).to(x.dtype)

    h = (hr * is_r + hg * is_g + hb * is_b) / 6.0
    h = torch.where(delt < eps, torch.zeros_like(h), h)
    h = h % 1.0
    return torch.cat([h, s, v], dim=1)


class CosineHSVLoss(nn.Module):
    def __init__(
        self,
        h_weight=0.5,
        s_weight=2.5,
        v_weight=1.0,
        eps=1e-6,
        gate_hue_by_sat=True,
        hue_gate_power=1.0,
        detach_gate=True,
        sat_under_weight=0.0,
    ):
        super().__init__()
        self.h_weight = float(h_weight)
        self.s_weight = float(s_weight)
        self.v_weight = float(v_weight)
        self.eps = float(eps)
        self.gate_hue_by_sat = bool(gate_hue_by_sat)
        self.hue_gate_power = float(hue_gate_power)
        self.detach_gate = bool(detach_gate)
        self.sat_under_weight = float(sat_under_weight)

    def forward(self, pred, gt):
        pred_hsv = rgb_to_hsv_torch(pred)
        gt_hsv = rgb_to_hsv_torch(gt)

        ph, ps, pv = pred_hsv[:, 0:1], pred_hsv[:, 1:2], pred_hsv[:, 2:3]
        gh, gs, gv = gt_hsv[:, 0:1], gt_hsv[:, 1:2], gt_hsv[:, 2:3]

        two_pi = 2.0 * math.pi
        phx, phy = torch.cos(two_pi * ph), torch.sin(two_pi * ph)
        ghx, ghy = torch.cos(two_pi * gh), torch.sin(two_pi * gh)

        if self.gate_hue_by_sat:
            gate = gs
            if self.detach_gate:
                gate = gate.detach()
            gate = torch.clamp(gate, 0.0, 1.0)
            if self.hue_gate_power != 1.0:
                gate = gate.pow(self.hue_gate_power)
            phx, phy = phx * gate, phy * gate
            ghx, ghy = ghx * gate, ghy * gate

        pred_vec = torch.cat([self.h_weight * phx, self.h_weight * phy, self.s_weight * ps, self.v_weight * pv], dim=1)
        gt_vec = torch.cat([self.h_weight * ghx, self.h_weight * ghy, self.s_weight * gs, self.v_weight * gv], dim=1)

        cos = F.cosine_similarity(pred_vec, gt_vec, dim=1, eps=self.eps)
        cos = torch.clamp(cos, -1.0, 1.0)
        loss = (1.0 - cos).mean()

        if self.sat_under_weight > 0.0:
            under_sat = F.relu(gs - ps)
            loss = loss + self.sat_under_weight * under_sat.mean()
        return loss


# ============================================================
# Model blocks
# ============================================================
def rgb_to_luma(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def GN(ch, max_groups=8, eps=1e-5, affine=True):
    g = min(max_groups, ch)
    while g > 1 and (ch % g != 0):
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=ch, eps=eps, affine=affine)


class GhostModule(nn.Module):
    def __init__(self, in_ch, out_ch, ratio=2, dw_size=3, stride=1, relu=True, gn_groups=8):
        super().__init__()
        init_channels = math.ceil(out_ch / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_ch, init_channels, 1, stride, 0, bias=False),
            GN(init_channels, max_groups=gn_groups),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            GN(new_channels, max_groups=gn_groups),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        return x * attn


class EnhancedGhostBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, gn_groups=8):
        super().__init__()
        self.ghost = GhostModule(in_ch, out_ch, stride=stride, gn_groups=gn_groups)
        self.ca = ChannelAttention(out_ch)
        self.sa = SpatialAttention()

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                GN(out_ch, max_groups=gn_groups),
            )

    def forward(self, x):
        out = self.ghost(x)
        out = self.ca(out)
        out = self.sa(out)
        return out + self.shortcut(x)


class MultiScaleFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(channels * 2, channels, 1)
        self.conv3x3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.conv5x5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.fusion = nn.Conv2d(channels * 2, channels, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x_low, x_high):
        if x_low.shape[2:] != x_high.shape[2:]:
            x_low = F.interpolate(x_low, size=x_high.shape[2:], mode="bilinear", align_corners=False)
        x_cat = self.conv1x1(torch.cat([x_low, x_high], dim=1))
        out = self.fusion(torch.cat([self.conv3x3(x_cat), self.conv5x5(x_cat)], dim=1))
        return self.act(out)


class NanoLLE_G(nn.Module):
    """
    G(backbone) takes multi-channel guidance input (e.g., RGB|Gamma|CLAHE|Sobel),
    but ONLY predicts/enhances the final RGB (3ch).

    Key idea:
      - Extra channels are used as guidance in the encoder/backbone.
      - Curve / Gain / Bias are applied ONLY to the original RGB (first 3 channels).
    """

    def __init__(self, base_channels=24, depths=(2, 2, 3), gn_groups=8, in_ch: int = 10, out_ch: int = 3):
        super().__init__()
        C = base_channels
        d0, d1, db = depths

        self.in_ch = int(in_ch)     # total input channels (guidance concatenation)
        self.out_ch = int(out_ch)   # final image channels (RGB=3)
        self.curve_K = 4

        # Stem uses ALL input channels
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_ch, C, 3, padding=1, bias=False),
            GN(C, max_groups=gn_groups),
            nn.ReLU(inplace=True),
        )

        self.enc0 = nn.Sequential(*[EnhancedGhostBlock(C, C, gn_groups=gn_groups) for _ in range(d0)])
        self.down1 = EnhancedGhostBlock(C, 2 * C, stride=2, gn_groups=gn_groups)
        self.enc1 = nn.Sequential(*[EnhancedGhostBlock(2 * C, 2 * C, gn_groups=gn_groups) for _ in range(d1)])
        self.down2 = EnhancedGhostBlock(2 * C, 4 * C, stride=2, gn_groups=gn_groups)

        self.bottleneck1 = nn.Sequential(*[EnhancedGhostBlock(4 * C, 4 * C, gn_groups=gn_groups) for _ in range(db)])

        self.bottleneck2 = nn.Sequential(*[EnhancedGhostBlock(4 * C, 4 * C, gn_groups=gn_groups) for _ in range(db)])

        self.up2_conv = EnhancedGhostBlock(4 * C, 2 * C, gn_groups=gn_groups)
        self.fuse1 = MultiScaleFusion(2 * C)
        self.up1_conv = EnhancedGhostBlock(2 * C, C, gn_groups=gn_groups)
        self.fuse0 = MultiScaleFusion(C)

        # Heads predict ONLY RGB(3ch)
        # - Zero-DCE style curve params for RGB
        self.curve_head = nn.Sequential(
            nn.Conv2d(C, C // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, self.out_ch * self.curve_K, 1),
            nn.Tanh(),
        )

        self.gain_head = nn.Sequential(
            nn.Conv2d(C, C // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, self.out_ch, 1),
        )
        self.bias_head = nn.Sequential(
            nn.Conv2d(C, C // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, self.out_ch, 1),
        )

        # Refine residual ONLY for RGB
        self.refine = nn.Sequential(
            nn.Conv2d(C + self.out_ch, C, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, self.out_ch, 1),
        )

        # Optional small output mixing (keeps 3ch)
        self.final_conv = nn.Conv2d(self.out_ch, self.out_ch, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, in_ch, H, W)
        # Original RGB is assumed to be the first 3 channels
        x_rgb = x[:, 0:3].clamp(0.0, 1.0)

        f = self.stem(x)
        s0 = self.enc0(f)

        f = self.down1(s0)
        s1 = self.enc1(f)

        f = self.down2(s1)
        f = self.bottleneck1(f)
        # (SNR-aware transformer removed)
        f = self.bottleneck2(f)

        f = F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False)
        f = self.up2_conv(f)
        f = self.fuse1(f, s1)

        f = F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False)
        f = self.up1_conv(f)
        f = self.fuse0(f, s0)

        # Predict RGB-only gain/bias
        gain_map = 1.0 + F.softplus(self.gain_head(f))
        bias = torch.sigmoid(self.bias_head(f)) * 0.5

        # Curve params for RGB
        r = self.curve_head(f)  # (B, out_ch*K, H, W)
        B0, _, H0, W0 = r.shape
        r = r.view(B0, self.curve_K, self.out_ch, H0, W0)

        # Apply curve iteratively (Zero-DCE style) to RGB only
        x_cur = x_rgb
        for k in range(self.curve_K):
            rk = r[:, k]
            x_cur = x_cur + rk * (x_cur - x_cur * x_cur)
        x_cur = x_cur.clamp(0.0, 1.0)

        enhanced = (x_cur * gain_map + bias).clamp(0.0, 1.0)
        residual = self.refine(torch.cat([f, enhanced], dim=1))
        pred = self.final_conv(enhanced + residual).clamp(0.0, 1.0)

        return {
            "prediction": pred,
            "gain_map": gain_map,
            "illum_map": gain_map[:, 0:1],
            "color_correction": gain_map,
}

class NanoLLE_v2(nn.Module):
    """
    GAN/cycle 제거된 supervised 버전.
    """
    def __init__(
        self,
        base_channels=24,
        depths=(2, 2, 3),
        gn_groups=8,
        w_ssim=0.1,
        w_lpips=0.1,
        w_color=1.0,
    ):
        super().__init__()
        self.G = NanoLLE_G(base_channels=base_channels, depths=depths, gn_groups=gn_groups, in_ch=10)
        self.hsv_loss_fn = CosineHSVLoss(h_weight=0.5, s_weight=2.5, v_weight=1.0)

        self.w_ssim = float(w_ssim)
        self.w_lpips = float(w_lpips)
        self.w_color = float(w_color)

    def _ssim_loss(self, pred, gt):
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        win, sigma = 11, 1.5

        coords = torch.arange(win, dtype=pred.dtype, device=pred.device) - win // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

        ch = pred.shape[1]
        window = window.expand(ch, -1, -1, -1).contiguous()
        pad = win // 2

        mu1 = F.conv2d(pred, window, padding=pad, groups=ch)
        mu2 = F.conv2d(gt, window, padding=pad, groups=ch)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        s1sq = (F.conv2d(pred ** 2, window, padding=pad, groups=ch) - mu1_sq).clamp(min=0)
        s2sq = (F.conv2d(gt ** 2, window, padding=pad, groups=ch) - mu2_sq).clamp(min=0)
        s12 = F.conv2d(pred * gt, window, padding=pad, groups=ch) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1sq + s2sq + C2) + 1e-12)
        return 1.0 - ssim_map.mean()

    def _loss_G(self, pred, gt, lpips_fn):
        recon = F.l1_loss(pred, gt)
        ssimL = self._ssim_loss(pred, gt)
        color = self.hsv_loss_fn(pred, gt)

        lp = pred.new_tensor(0.0)
        if lpips_fn is not None:
            pred_lp = pred.clamp(0, 1) * 2 - 1
            gt_lp = gt.clamp(0, 1) * 2 - 1
            lp = lpips_fn(pred_lp, gt_lp).mean()

        total = recon + self.w_ssim * ssimL + self.w_lpips * lp + self.w_color * color
        return total, recon, ssimL, lp, color

    def forward(self, data_batch: Dict[str, torch.Tensor], lpips_fn):
        x = data_batch["LQ_image"]
        gt = data_batch.get("HQ_image", None)

        if x.dim() == 5:  # (B,K,C,H,W)
            B, K, C, H, W = x.shape
            x = x.view(B * K, C, H, W)
            if gt is not None:
                gt = gt.view(B * K, C, H, W)

        outG = self.G(x)
        pred = outG["prediction"]

        if gt is None:
            return {
                "prediction": pred,
                "loss": None,
                "loss_dict": {},
                "gain_map": outG["gain_map"],
                "illum_map": outG["illum_map"],
                "color_correction": outG["color_correction"],
            }

        total, recon, ssimL, lp, color = self._loss_G(pred, gt, lpips_fn)

        loss_dict = {
            "total": float(total.detach()),
            "recon": float(recon.detach()),
            "ssim": float(ssimL.detach()),
            "lpips": float(lp.detach()),
            "color": float(color.detach()),
        }

        return {
            "prediction": pred,
            "loss": total,
            "loss_dict": loss_dict,
            "gain_map": outG["gain_map"],
            "illum_map": outG["illum_map"],
            "color_correction": outG["color_correction"],
        }

    def export_inference_G(self):
        return self.G

# ----------------------------
# Eval helpers
# ----------------------------
@torch.no_grad()
def evaluate_fullres_G(
    model_v2: NanoLLE_v2,
    loader: DataLoader,
    device: str,
    eval_lpips: Optional[EvalLPIPS] = None,
    feature_cfg: Optional[Dict[str, float]] = None,
):
    """
    internal val: G(low)->pred, gt와 metric 계산.
    """
    was_training = model_v2.training
    model_v2.eval()

    psnrs, ssims, lpv = [], [], []
    G = model_v2.export_inference_G()
    G.eval()

    feature_cfg = feature_cfg or {}
    _gamma = float(feature_cfg.get("gamma", 0.5))
    _sobel_scale = float(feature_cfg.get("sobel_scale", 8.0))
    _clahe_clip = float(feature_cfg.get("clahe_clip", 2.0))
    _clahe_tile = int(feature_cfg.get("clahe_tile", 8))
    _clahe_add = int(feature_cfg.get("clahe_add", 15))


    for low, gt in loader:
        low = build_input_features(
            low,
            gamma=_gamma,
            sobel_scale=_sobel_scale,
            clahe_clip=_clahe_clip,
            clahe_tile=_clahe_tile,
            clahe_add=_clahe_add,
        )
        low = low.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        pred = G(low)["prediction"]


        psnrs.append(psnr_torch(pred, gt))
        ssims.append(ssim_torch(pred, gt))
        if eval_lpips is not None:
            lpv.append(eval_lpips(pred, gt))

    if was_training:
        model_v2.train()

    return {
        "psnr": float(np.mean(psnrs)) if psnrs else 0.0,
        "ssim": float(np.mean(ssims)) if ssims else 0.0,
        "lpips": float(np.mean(lpv)) if lpv else 0.0,
    }


@torch.no_grad()
def evaluate_fullres_val_loss(
    model_v2: NanoLLE_v2,
    loader: DataLoader,
    device: str,
    lpips_fn_train: Optional[nn.Module] = None,
    feature_cfg: Optional[Dict[str, float]] = None,
):
    """
    internal val에서 train과 동일한 loss(total) 평균을 계산.
    val은 eval_interval에서만 계산 -> sparse points로 plot.
    """
    was_training = model_v2.training
    model_v2.eval()

    totals = []
    G = model_v2.export_inference_G()
    G.eval()

    feature_cfg = feature_cfg or {}
    _gamma = float(feature_cfg.get("gamma", 0.5))
    _sobel_scale = float(feature_cfg.get("sobel_scale", 8.0))
    _clahe_clip = float(feature_cfg.get("clahe_clip", 2.0))
    _clahe_tile = int(feature_cfg.get("clahe_tile", 8))
    _clahe_add = int(feature_cfg.get("clahe_add", 15))


    for low, gt in loader:
        low = build_input_features(
            low,
            gamma=_gamma,
            sobel_scale=_sobel_scale,
            clahe_clip=_clahe_clip,
            clahe_tile=_clahe_tile,
            clahe_add=_clahe_add,
        )
        low = low.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        pred = G(low)["prediction"]
        total, _, _, _, _ = model_v2._loss_G(pred, gt, lpips_fn_train)
        totals.append(float(total.detach().cpu().item()))

    if was_training:
        model_v2.train()

    return float(np.mean(totals)) if totals else 0.0


@torch.no_grad()
def inference_and_zip_fullres_G(
    model_v2: NanoLLE_v2,
    val_low_paths: List[str],
    device: str,
    out_dir: str,
    zip_path: str,
    feature_cfg: Optional[Dict[str, float]] = None,
):
    """
    FULL-RES inference to outputs + zip.
    """
    was_training = model_v2.training
    model_v2.eval()
    G = model_v2.export_inference_G()
    G.eval()

    feature_cfg = feature_cfg or {}
    _gamma = float(feature_cfg.get("gamma", 0.5))
    _sobel_scale = float(feature_cfg.get("sobel_scale", 8.0))
    _clahe_clip = float(feature_cfg.get("clahe_clip", 2.0))
    _clahe_tile = int(feature_cfg.get("clahe_tile", 8))
    _clahe_add = int(feature_cfg.get("clahe_add", 15))


    ensure_dir(out_dir)
    for p in glob.glob(os.path.join(out_dir, "*")):
        if os.path.isfile(p):
            try:
                os.remove(p)
            except Exception:
                pass

    iterator = enumerate(val_low_paths, start=1)
    if _HAS_TQDM:
        iterator = tqdm(iterator, total=len(val_low_paths), desc="infer(G-fullres)", dynamic_ncols=True, leave=False)

    for _, p in iterator:
        img = pil_load_rgb(p)
        x = pil_to_tensor01(img).unsqueeze(0)
        x = build_input_features(
            x,
            gamma=_gamma,
            sobel_scale=_sobel_scale,
            clahe_clip=_clahe_clip,
            clahe_tile=_clahe_tile,
            clahe_add=_clahe_add,
        )
        x = x.to(device)

        pred = G(x)["prediction"][0]
        out_img = tensor01_to_pil(pred)

        fname = os.path.basename(p)
        out_path = os.path.join(out_dir, fname)

        if fname.lower().endswith((".jpg", ".jpeg")):
            out_img.save(out_path, quality=95, subsampling=0)
        else:
            out_img.save(out_path)

    ensure_dir(os.path.dirname(zip_path) if os.path.dirname(zip_path) else ".")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        outs = sorted(glob.glob(os.path.join(out_dir, "*")), key=natural_key)
        for p in outs:
            if os.path.isfile(p):
                zf.write(p, arcname=os.path.basename(p))

    if was_training:
        model_v2.train()


def composite_score(psnr: float, ssim: float, lpips_v: float) -> float:
    return psnr + 10.0 * ssim - 10.0 * lpips_v


class CSVLogger:
    def __init__(self, path: str, header: List[str]):
        self.path = path
        self.header = header
        self._init_file()

    def _init_file(self):
        ensure_dir(os.path.dirname(self.path) if os.path.dirname(self.path) else ".")
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(self.header)

    def write_row(self, row: List):
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(row)


def save_state_fp16(model: nn.Module, path: str):
    sd = {k: v.detach().cpu().half() for k, v in model.state_dict().items()}
    torch.save(sd, path)


def save_g_only_fp16(model_v2: NanoLLE_v2, path: str):
    g = model_v2.export_inference_G()
    sd = {k: v.detach().cpu().half() for k, v in g.state_dict().items()}
    torch.save(sd, path)


# ----------------------------
# Dataset helpers
# ----------------------------
def find_low_normal_pairs(train_root: str) -> List[Tuple[str, str]]:
    low_paths, normal_paths = [], []
    for ext in ("jpg", "png", "jpeg", "JPG", "PNG", "JPEG"):
        low_paths += glob.glob(os.path.join(train_root, "**", "low", f"*.{ext}"), recursive=True)
        normal_paths += glob.glob(os.path.join(train_root, "**", "normal", f"*.{ext}"), recursive=True)

    low_map = {os.path.basename(p): p for p in low_paths}
    normal_map = {os.path.basename(p): p for p in normal_paths}

    keys = sorted(list(set(low_map.keys()).intersection(set(normal_map.keys()))), key=natural_key)
    return [(low_map[k], normal_map[k]) for k in keys]


def find_challenge_val_lows(val_root: str) -> List[str]:
    paths = []
    for ext in ("jpg", "png", "jpeg", "JPG", "PNG", "JPEG"):
        paths += glob.glob(os.path.join(val_root, "**", "low", f"*.{ext}"), recursive=True)
    return sorted(paths, key=natural_key)


class PairedLLIEDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        train: bool,
        patch_size: int = 256,
        dark_bias: bool = True,
        num_candidates: int = 8,
        aug: bool = True,
    ):
        super().__init__()
        self.pairs = pairs
        self.train = train
        self.patch_size = patch_size
        self.dark_bias = False
        self.num_candidates = num_candidates
        self.aug = aug

    def set_patch_size(self, ps: int):
        self.patch_size = ps

    def __len__(self):
        return len(self.pairs)

    def _random_crop_coords(self, w: int, h: int, ps: int):
        if w <= ps or h <= ps:
            return max((w - ps) // 2, 0), max((h - ps) // 2, 0)
        return np.random.randint(0, w - ps + 1), np.random.randint(0, h - ps + 1)

    def _dark_biased_crop(self, low_img: Image.Image, gt_img: Image.Image, ps: int):
        w, h = low_img.size
        if w < ps or h < ps:
            pad_w, pad_h = max(ps - w, 0), max(ps - h, 0)
            if pad_w > 0 or pad_h > 0:
                low_img = Image.fromarray(np.pad(np.array(low_img), ((0, pad_h), (0, pad_w), (0, 0)), mode="edge"))
                gt_img = Image.fromarray(np.pad(np.array(gt_img), ((0, pad_h), (0, pad_w), (0, 0)), mode="edge"))
                w, h = low_img.size

        best_xy, best_score = None, None
        for _ in range(self.num_candidates):
            x0, y0 = self._random_crop_coords(w, h, ps)
            crop = low_img.crop((x0, y0, x0 + ps, y0 + ps))
            arr = np.array(crop, dtype=np.float32)
            y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
            score = float(y.mean())  # lower = darker
            if best_score is None or score < best_score:
                best_score = score
                best_xy = (x0, y0)

        x0, y0 = best_xy
        return (
            low_img.crop((x0, y0, x0 + ps, y0 + ps)),
            gt_img.crop((x0, y0, x0 + ps, y0 + ps)),
        )

    def _augment(self, low: Image.Image, gt: Image.Image):
        if np.random.rand() < 0.5:
            low = low.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.rand() < 0.5:
            low = low.transpose(Image.FLIP_TOP_BOTTOM)
            gt = gt.transpose(Image.FLIP_TOP_BOTTOM)
        k = np.random.randint(0, 4)
        if k:
            low = low.rotate(90 * k, expand=True)
            gt = gt.rotate(90 * k, expand=True)
        return low, gt

    def __getitem__(self, idx):
        low_p, gt_p = self.pairs[idx]
        low_img = pil_load_rgb(low_p)
        gt_img = pil_load_rgb(gt_p)

        if self.train:
            ps = self.patch_size
            if self.dark_bias:
                low_img, gt_img = self._dark_biased_crop(low_img, gt_img, ps)
            else:
                w, h = low_img.size
                x0, y0 = self._random_crop_coords(w, h, ps)
                low_img = low_img.crop((x0, y0, x0 + ps, y0 + ps))
                gt_img = gt_img.crop((x0, y0, x0 + ps, y0 + ps))
            if self.aug:
                low_img, gt_img = self._augment(low_img, gt_img)

        return pil_to_tensor01(low_img), pil_to_tensor01(gt_img)


# ----------------------------
# Patch schedule
# ----------------------------
@dataclass
class PatchScheduleItem:
    it: int
    patch: int
    batch: int


def build_default_patch_schedule():
    return [
        PatchScheduleItem(it=0,      patch=256,  batch=16),
        PatchScheduleItem(it=40000,  patch=384,  batch=8),
        PatchScheduleItem(it=90000,  patch=512,  batch=5),
        PatchScheduleItem(it=140000, patch=768,  batch=2),
        PatchScheduleItem(it=190000, patch=1024, batch=2),
    ]


def get_schedule_at(schedule: List[PatchScheduleItem], it: int) -> PatchScheduleItem:
    cur = schedule[0]
    for s in schedule:
        if it >= s.it:
            cur = s
        else:
            break
    return cur


# ----------------------------
# Main
# ----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--val_root", type=str, required=True)

    parser.add_argument("--train_count", type=int, default=300)
    parser.add_argument("--internal_val_count", type=int, default=49)

    parser.add_argument("--save_dir", type=str, default="./runs/llie_supervised_curve_only")
    parser.add_argument("--total_iters", type=int, default=200000)
    parser.add_argument("--eval_interval", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)

    # model config
    parser.add_argument("--base_channels", type=int, default=20)  # <= 1MB 목표면 20부터 추천
    parser.add_argument("--depths", type=int, nargs=3, default=[1, 1, 2])
    parser.add_argument("--gn_groups", type=int, default=8)
    parser.add_argument("--curve_K", type=int, default=8)
    # input feature config (RGB | Gamma | CLAHE | Sobel(1ch))
    parser.add_argument("--gamma", type=float, default=0.5, help="gamma correction exponent (<1 brightens)")
    parser.add_argument("--clahe_clip", type=float, default=2.0, help="CLAHE clipLimit")
    parser.add_argument("--clahe_tile", type=int, default=8, help="CLAHE tileGridSize (N -> NxN)")
    parser.add_argument("--clahe_add", type=int, default=15, help="Brightness lift after CLAHE on L channel")
    parser.add_argument("--sobel_scale", type=float, default=8.0, help="Sobel edge scaling divisor (|Gx|+|Gy|)/scale -> [0,1]")


    # loss weights
    parser.add_argument("--w_ssim", type=float, default=0.1)
    parser.add_argument("--w_lpips", type=float, default=0.1)
    parser.add_argument("--w_color", type=float, default=1.0)

    # optim
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=0.0)

    # LPIPS
    parser.add_argument("--lpips_net", type=str, default="alex")

    # best metric
    parser.add_argument("--best_metric", type=str, default="composite", choices=["composite", "ssim", "psnr"])

    # misc
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--export_g_only", action="store_true", help="best 갱신 시 G만 fp16으로도 별도 저장")

    args = parser.parse_args()

    # Feature concat config (must match training/eval/infer)
    feature_cfg = {
        "gamma": args.gamma,
        "sobel_scale": args.sobel_scale,
        "clahe_clip": args.clahe_clip,
        "clahe_tile": args.clahe_tile,
        "clahe_add": args.clahe_add,
    }


    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ensure_dir(args.save_dir)
    ckpt_dir = os.path.join(args.save_dir, "checkpoints")
    infer_dir = os.path.join(args.save_dir, "challenge_val_outputs_fullres")
    vis_root = os.path.join(args.save_dir, "vis_strips")
    ensure_dir(vis_root)
    ensure_dir(ckpt_dir)
    ensure_dir(infer_dir)

    # loss plot output (OVERWRITE)
    loss_plot_path = os.path.join(args.save_dir, "loss_curve.png")

    use_tqdm = _HAS_TQDM and (not args.no_tqdm)
    pbar_holder = {"pbar": None}

    def log_line(msg: str):
        if use_tqdm and pbar_holder["pbar"] is not None:
            pbar_holder["pbar"].write(msg)
        else:
            print(msg)

    # CSV logger
    csv_path = os.path.join(args.save_dir, "train_log.csv")
    csv_logger = CSVLogger(
        csv_path,
        header=[
            "iter", "patch", "batch", "lr",
            "loss_total",
            "recon", "ssim", "lpips", "color",
            "img_per_sec", "eta_sec", "seen_images",
            "val_psnr", "val_ssim", "val_lpips", "val_score", "best_iter",
            "val_loss_total"
        ],
    )

    log_line(f"[Log] CSV -> {csv_path}")
    log_line("[Mode] Training: supervised (G only) | Val/Infer: G only, FULL-RES (NO TILING).")
    log_line(f"[Plot] loss curve -> {loss_plot_path} (overwrite every eval_interval)")

    # ----------------------------
    # Data
    # ----------------------------
    pairs = find_low_normal_pairs(args.train_root)
    if len(pairs) == 0:
        raise RuntimeError(f"No train pairs found under: {args.train_root}")
    pairs = sorted(pairs, key=lambda x: natural_key(x[0]))

    n_train = min(args.train_count, len(pairs))
    n_val = min(args.internal_val_count, max(0, len(pairs) - n_train))
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    unused = len(pairs) - (len(train_pairs) + len(val_pairs))

    log_line(f"[Data] total_pairs={len(pairs)}")
    log_line(f"[Split] train={len(train_pairs)} internal_val={len(val_pairs)} unused={unused}")

    challenge_val_lows = find_challenge_val_lows(args.val_root)
    if len(challenge_val_lows) == 0:
        raise RuntimeError(f"No challenge val low images found under: {args.val_root}")
    log_line(f"[Challenge Val] low images found: {len(challenge_val_lows)} (GT 없음 → inference만)")

    # ----------------------------
    # Model
    # ----------------------------
    model = NanoLLE_v2(
        base_channels=args.base_channels,
        depths=tuple(args.depths),
        gn_groups=args.gn_groups,
        # curve_K=args.curve_K,
        w_ssim=args.w_ssim,
        w_lpips=args.w_lpips,
        w_color=args.w_color,
    ).to(device)

    g_only = model.export_inference_G()
    log_line(f"[Model-G only] params={count_params(g_only):,} | size(fp32+buf)~{get_model_size_mb(g_only):.3f}MB")
    if get_model_size_mb(g_only) > 1.0:
        log_line("[Warn] G-only size가 1MB를 넘음. --base_channels 또는 --depths를 더 줄여야 함.")

    # ----------------------------
    # LPIPS
    # ----------------------------
    train_lpips = build_train_lpips(args.lpips_net, device) if (device == "cuda") else None
    if train_lpips is None:
        log_line("[Warn] Train LPIPS disabled (no CUDA or lpips not installed). w_lpips는 무시됨.")
    eval_lpips = None
    if _HAS_LPIPS and device == "cuda":
        eval_lpips = EvalLPIPS(net=args.lpips_net, device=device)

    # ----------------------------
    # Optim
    # ----------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.total_iters, eta_min=args.lr * 0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device == "cuda"))

    # ----------------------------
    # Patch schedule & loaders
    # ----------------------------
    schedule = build_default_patch_schedule()
    cur = get_schedule_at(schedule, 0)

    train_ds = PairedLLIEDataset(train_pairs, train=True, patch_size=cur.patch, dark_bias=True, num_candidates=8, aug=True)
    val_ds = PairedLLIEDataset(val_pairs, train=False, patch_size=cur.patch, dark_bias=False, aug=False)

    train_loader = DataLoader(
        train_ds, batch_size=cur.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    train_iter = iter(train_loader)

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # ----------------------------
    # Best tracking
    # ----------------------------
    best_score = -1e18
    best_it = -1
    best_path = os.path.join(ckpt_dir, "best_fp16.pth")
    last_path = os.path.join(ckpt_dir, "last_fp16.pth")
    best_g_path = os.path.join(ckpt_dir, "best_Gonly_fp16.pth")

    last_val_psnr = None
    last_val_ssim = None
    last_val_lpips = None
    last_val_score = None
    last_val_loss_total = None

    # ----------------------------
    # Loss history (for plotting)
    # ----------------------------
    train_loss_iters: List[int] = []
    train_loss_vals: List[float] = []
    val_loss_iters: List[int] = []
    val_loss_vals: List[float] = []

    # ----------------------------
    # Train loop
    # ----------------------------
    model.train()
    ema_timer = EMATimer(momentum=0.95)
    last_tick = time.time()
    last_stage = (cur.patch, cur.batch)
    seen_images = 0

    def maybe_update_loader(it: int):
        nonlocal train_loader, train_iter, cur, last_stage
        new = get_schedule_at(schedule, it)

        if (new.patch, new.batch) != last_stage:
            log_line(f"🔄 [Schedule] Iter {it}: patch {cur.patch}->{new.patch}, batch {cur.batch}->{new.batch}")
            last_stage = (new.patch, new.batch)

        if new.patch != cur.patch:
            train_ds.set_patch_size(new.patch)

        if new.batch != cur.batch:
            train_loader = DataLoader(
                train_ds, batch_size=new.batch, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True
            )
            train_iter = iter(train_loader)

        cur = new

    if use_tqdm:
        pbar = tqdm(total=args.total_iters, dynamic_ncols=True, smoothing=0.0, leave=True)
        pbar.set_description("train(supervised)")
        pbar_holder["pbar"] = pbar
    else:
        pbar = None

    for it in range(1, args.total_iters + 1):
        maybe_update_loader(it)

        try:
            low, gt = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            low, gt = next(train_iter)

        # Build concatenated input (RGB | Gamma | CLAHE | Sobel) -> 10ch
        low = build_input_features(
            low,
            gamma=args.gamma,
            sobel_scale=args.sobel_scale,
            clahe_clip=args.clahe_clip,
            clahe_tile=args.clahe_tile,
            clahe_add=args.clahe_add,
        )

        low = low.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)


        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(args.amp and device == "cuda")):
            out = model({"LQ_image": low, "HQ_image": gt}, lpips_fn=train_lpips)
            loss = out["loss"]
            loss_dict = out["loss_dict"]

        scaler.scale(loss).backward()
        if args.grad_clip and args.grad_clip > 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(opt)
        scaler.update()
        scheduler.step()

        # record train loss per-iter (for x-axis iter plot)
        train_loss_iters.append(it)
        train_loss_vals.append(float(loss_dict["total"]))

        now = time.time()
        dt = now - last_tick
        last_tick = now
        ema_timer.update(dt)

        seen_images += int(low.shape[0])

        if pbar is not None:
            pbar.update(1)

        if it % args.log_interval == 0:
            it_time = ema_timer.value()
            ips = (cur.batch / max(it_time, 1e-9))
            eta_sec = (args.total_iters - it) * it_time
            lr = opt.param_groups[0]["lr"]

            postfix = {
                "ps": cur.patch,
                "bs": cur.batch,
                "lr": f"{lr:.2e}",
                "L": f"{float(loss_dict['total']):.4f}",
                "img/s": f"{ips:.2f}",
                "ETA": format_seconds(eta_sec),
                "seen": str(seen_images),
            }
            if last_val_psnr is not None:
                postfix.update({
                    "vP": f"{last_val_psnr:.3f}",
                    "vS": f"{last_val_ssim:.4f}",
                    "vL": f"{last_val_lpips:.4f}",
                    "vSc": f"{last_val_score:.3f}",
                    "vLoss": f"{last_val_loss_total:.4f}",
                    "best@": str(best_it),
                })
            if pbar is not None:
                pbar.set_postfix(postfix)

            log_line(
                f"it={it} | ps={cur.patch} bs={cur.batch} lr={lr:.2e} "
                f"L={loss_dict['total']:.4f} (recon={loss_dict['recon']:.4f}, ssim={loss_dict['ssim']:.4f}, lp={loss_dict['lpips']:.4f}, color={loss_dict['color']:.4f}) "
                f"| {ips:.2f} img/s | ETA={format_seconds(eta_sec)} | seen={seen_images}"
            )

            csv_logger.write_row([
                it, cur.patch, cur.batch, lr,
                float(loss_dict["total"]),
                float(loss_dict["recon"]), float(loss_dict["ssim"]), float(loss_dict["lpips"]), float(loss_dict["color"]),
                float(ips), float(eta_sec), int(seen_images),
                ("" if last_val_psnr is None else float(last_val_psnr)),
                ("" if last_val_ssim is None else float(last_val_ssim)),
                ("" if last_val_lpips is None else float(last_val_lpips)),
                ("" if last_val_score is None else float(last_val_score)),
                best_it if best_it >= 0 else "",
                ("" if last_val_loss_total is None else float(last_val_loss_total)),
            ])

        # internal val (G only, FULL-RES) + debug strips + loss plot
        if (it % args.eval_interval == 0) and (len(val_pairs) > 0):

            # ----------------------------
            # Save debug strips (train/infer) every eval_interval
            # (VAL strip 저장은 제거됨)
            # ----------------------------
            vis_dir = os.path.join(vis_root, f"iter_{it:07d}")
            ensure_dir(vis_dir)

            # 1) TRAIN strip: current batch output
            try:
                save_concat_strip(
                    input_rgb01=low[0],
                    gain_map=out["gain_map"][0],
                    illum_map=out["illum_map"][0],
                    color_corr=out["color_correction"][0],
                    pred_rgb01=out["prediction"][0],
                    save_path=os.path.join(vis_dir, "train_strip.png"),
                    max_gain_vis=4.0,
                    add_labels=True,
                )
            except Exception as e:
                log_line(f"[Warn] train strip save failed: {e}")

            # 2) INFER strip: challenge val low 중 1장만
            try:
                model.eval()
                G = model.export_inference_G()
                G.eval()

                p0 = challenge_val_lows[0]
                img0 = pil_load_rgb(p0)
                x0 = pil_to_tensor01(img0).unsqueeze(0)
                x0 = build_input_features(
                    x0,
                    gamma=args.gamma,
            sobel_scale=args.sobel_scale,
                    clahe_clip=args.clahe_clip,
                    clahe_tile=args.clahe_tile,
                    clahe_add=args.clahe_add,
                )
                x0 = x0.to(device)

                o0 = G(x0)
                save_concat_strip(
                    input_rgb01=x0[0],
                    gain_map=o0["gain_map"][0],
                    illum_map=o0["illum_map"][0],
                    color_corr=o0["color_correction"][0],
                    pred_rgb01=o0["prediction"][0],
                    save_path=os.path.join(vis_dir, f"infer_strip_{os.path.basename(p0)}.png"),
                    max_gain_vis=4.0,
                    add_labels=True,
                )
            except Exception as e:
                log_line(f"[Warn] infer strip save failed: {e}")
            finally:
                model.train()

            if pbar is not None:
                pbar.set_description("eval(G-fullres)")

            # metrics
            metrics = evaluate_fullres_G(model, val_loader, device=device, eval_lpips=eval_lpips, feature_cfg=feature_cfg)
            psnr_v, ssim_v, lpips_v = metrics["psnr"], metrics["ssim"], metrics["lpips"]

            # val loss (same formula as train)
            val_loss_total = evaluate_fullres_val_loss(
                model_v2=model,
                loader=val_loader,
                device=device,
                lpips_fn_train=train_lpips,
                feature_cfg=feature_cfg,
            )
            last_val_loss_total = val_loss_total

            # record val loss history for plotting
            val_loss_iters.append(it)
            val_loss_vals.append(val_loss_total)

            # plot overwrite at every eval interval
            try:
                plot_train_val_losses(
                    train_iters=train_loss_iters,
                    train_losses=train_loss_vals,
                    val_iters=val_loss_iters,
                    val_losses=val_loss_vals,
                    save_path=loss_plot_path,
                    title="Loss Curve (Train/Val, x=iter)",
                    max_train_points=6000,
                )
                log_line(f"[Plot] saved -> {loss_plot_path} (overwritten)")
            except Exception as e:
                log_line(f"[Warn] loss plot save failed: {e}")

            # best score
            if args.best_metric == "ssim":
                score = ssim_v
            elif args.best_metric == "psnr":
                score = psnr_v
            else:
                score = composite_score(psnr_v, ssim_v, lpips_v)

            last_val_psnr, last_val_ssim, last_val_lpips, last_val_score = psnr_v, ssim_v, lpips_v, score

            log_line(
                f"[VAL] it={it} | vPSNR={psnr_v:.4f} vSSIM={ssim_v:.6f} vLPIPS={lpips_v:.6f} "
                f"| vLOSS={val_loss_total:.6f} | score={score:.6f} ({args.best_metric}) | best={best_score:.6f}@{best_it}"
            )

            # save last
            save_state_fp16(model, last_path)

            # save best + (optional) challenge inference zip
            if score > best_score:
                best_score = score
                best_it = it
                save_state_fp16(model, best_path)
                log_line(f"✅ [BEST] it={it} -> {best_path}")

                if args.export_g_only:
                    save_g_only_fp16(model, best_g_path)
                    log_line(f"✅ [BEST-GONLY] it={it} -> {best_g_path}")

                if pbar is not None:
                    pbar.set_description("infer(G-fullres)")

                out_dir = os.path.join(infer_dir, f"iter_{it:07d}")
                out_zip = os.path.join(infer_dir, f"iter_{it:07d}.zip")
                log_line("→ Running challenge-val inference (G only, FULL-RES, NO TILING) ...")
                inference_and_zip_fullres_G(
                    model_v2=model,
                    val_low_paths=challenge_val_lows,
                    device=device,
                    out_dir=out_dir,
                    zip_path=out_zip,
                    feature_cfg=feature_cfg,
                )
                log_line(f"📦 [ZIP] {out_zip}")

            if pbar is not None:
                pbar.set_description("train(supervised)")

    if pbar is not None:
        pbar.close()

    log_line(f"\nDone. best_it={best_it}, best_score={best_score:.6f}")
    log_line(f"Best weights(fp16): {best_path}")
    log_line(f"Last weights(fp16): {last_path}")
    if args.export_g_only:
        log_line(f"Best weights(G-only fp16): {best_g_path}")


if __name__ == "__main__":
    main()