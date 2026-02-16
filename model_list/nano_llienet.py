import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import lpips

def rgb_to_hsv_torch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x: (B,3,H,W), assumed in [0,1]
    returns: (B,3,H,W) with H,S,V in [0,1]
    Differentiable w.r.t. x (except at some non-smooth points like max/min ties).
    """
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]

    maxc, _ = x.max(dim=1, keepdim=True)  # (B,1,H,W)
    minc, _ = x.min(dim=1, keepdim=True)
    v = maxc

    delt = maxc - minc
    s = delt / (maxc + eps)

    # Hue
    # Avoid division by zero
    delt_safe = delt + eps

    # Compute intermediate hues for each channel being max
    # Standard HSV formula normalized to [0,1)
    hr = ((g - b) / delt_safe) % 6.0
    hg = ((b - r) / delt_safe) + 2.0
    hb = ((r - g) / delt_safe) + 4.0

    is_r = (maxc == r).to(x.dtype)
    is_g = (maxc == g).to(x.dtype)
    is_b = (maxc == b).to(x.dtype)

    h = (hr * is_r + hg * is_g + hb * is_b) / 6.0
    # delt == 0 (gray) -> hue undefined, set to 0
    h = torch.where(delt < eps, torch.zeros_like(h), h)
    h = h % 1.0

    return torch.cat([h, s, v], dim=1)

def rgb_to_luma(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def GN(ch, max_groups=8, eps=1e-5, affine=True):
    g = min(max_groups, ch)
    while ch % g != 0:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=ch, eps=eps, affine=affine)

# ============= Ghost Module =============
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
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2,
                      groups=init_channels, bias=False),
            GN(new_channels, max_groups=gn_groups),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


# ============= Spatial Attention =============
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


# ============= Channel Attention =============
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        return x * attn


# ============= Enhanced Ghost Block =============
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


# ============= Multi-Scale Fusion =============
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
            x_low = F.interpolate(x_low, size=x_high.shape[2:], mode='bilinear', align_corners=False)
        x_cat = self.conv1x1(torch.cat([x_low, x_high], dim=1))
        out = self.fusion(torch.cat([self.conv3x3(x_cat), self.conv5x5(x_cat)], dim=1))
        return self.act(out)


# ============= Main Network =============
class NanoLLE(nn.Module):
    """
    NanoLLE (Multi-gamma 제거 버전)

    Pipeline:
      1. Input x (3ch) → Encoder-Decoder → features
      2. Gain-map (곱셈, 구조 보존) + Bias (덧셈, 어두운 영역) + Residual
    """

    def __init__(self, base_channels=24, depths=(2, 2, 3), gn_groups=8):
        super().__init__()
        C = base_channels
        d0, d1, db = depths
        self.patch_size = 256

        # Stem: 3ch → C (multi-gamma 제거 → 3ch 고정)
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=False),
            GN(C, max_groups=gn_groups),
            nn.ReLU(inplace=True),
        )

        # Encoder
        self.enc0 = nn.Sequential(*[EnhancedGhostBlock(C, C, gn_groups=gn_groups) for _ in range(d0)])
        self.down1 = EnhancedGhostBlock(C, 2 * C, stride=2, gn_groups=gn_groups)
        self.enc1 = nn.Sequential(*[EnhancedGhostBlock(2 * C, 2 * C, gn_groups=gn_groups) for _ in range(d1)])
        self.down2 = EnhancedGhostBlock(2 * C, 4 * C, stride=2, gn_groups=gn_groups)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[EnhancedGhostBlock(4 * C, 4 * C, gn_groups=gn_groups) for _ in range(db)])

        # Decoder
        self.up2_conv = EnhancedGhostBlock(4 * C, 2 * C, gn_groups=gn_groups)
        self.fuse1 = MultiScaleFusion(2 * C)
        self.up1_conv = EnhancedGhostBlock(2 * C, C, gn_groups=gn_groups)
        self.fuse0 = MultiScaleFusion(C)

        self.edge_head = nn.Conv2d(C, 1, kernel_size=1)

        # ===== Output Heads =====
        # (1) Gain Map: 곱셈 (구조/색 보존)
        self.gain_head = nn.Sequential(
            nn.Conv2d(C, C // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, 3, 1),
        )

        # (2) Additive Bias: 덧셈 (x≈0 영역 직접 밝히기)
        self.bias_head = nn.Sequential(
            nn.Conv2d(C, C // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, 3, 1),
        )

        # (3) Refinement: 미세 보정
        self.refine = nn.Sequential(
            nn.Conv2d(C + 3, C, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, 3, 1),
        )

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

    def forward(self, data_batch, lpips_fn):
        x = data_batch["LQ_image"]  # (B,3,H,W)
        gt = data_batch.get("HQ_image", None)  # (B,3,H,W) or None

        # import numpy as np
        # import matplotlib.pyplot as plt
        # for batch_idx in range(x.shape[0]):
        #     for patch_idx in range(x.shape[1]):
        #         fig, ax = plt.subplots(1, 2)
        #         ax[0].imshow(np.transpose(x[batch_idx][patch_idx].cpu().detach().numpy(), (1, 2, 0)))
        #         ax[1].imshow(np.transpose(gt[batch_idx][patch_idx].cpu().detach().numpy(), (1, 2, 0)))
        #         plt.show()
        # print(x.shape)

        if x.dim() == 5:  # (B,K,C,H,W)
            B, K, C, H, W = x.shape
            x = x.view(B * K, C, H, W)
            gt = gt.view(B * K, C, H, W)

        # ===== Encoder =====
        f = self.stem(x)
        s0 = self.enc0(f)

        f = self.down1(s0)
        s1 = self.enc1(f)

        f = self.down2(s1)
        f = self.bottleneck(f)

        # ===== Decoder =====
        f = F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False)
        f = self.up2_conv(f)
        f = self.fuse1(f, s1)

        f = F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False)
        f = self.up1_conv(f)
        f = self.fuse0(f, s0)

        edge_pred = torch.sigmoid(self.edge_head(f))  # (B,1,H,W)

        # ===== Gain × input + Bias + Residual =====
        gain_map = 1.0 + F.softplus(self.gain_head(f))  # (B,3,H,W), >= 1
        bias = torch.sigmoid(self.bias_head(f)) * 0.5  # (B,3,H,W), [0,0.5]

        enhanced = (x * gain_map + bias).clamp(0.0, 1.0)
        residual = self.refine(torch.cat([f, enhanced], dim=1))
        pred = (enhanced + residual).clamp(0.0, 1.0)

        # ===== Loss =====
        if gt is not None:
            loss, loss_dict = self._calculate_loss(pred, gt, gain_map, lpips_fn, edge_pred=edge_pred, inp_img=x)
        else:
            loss, loss_dict = None, {}

        return {
            "prediction": pred,
            "loss": loss,
            "loss_dict": loss_dict,
            "gain_map": gain_map,
            "illum_map": gain_map[:, 0:1],
            "color_correction": gain_map,
        }

    # ============================================================
    # SSIM Loss
    # ============================================================
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

        mu1  = F.conv2d(pred,      window, padding=pad, groups=ch)
        mu2  = F.conv2d(gt,        window, padding=pad, groups=ch)
        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2

        s1sq = (F.conv2d(pred**2,    window, padding=pad, groups=ch) - mu1_sq).clamp(min=0)
        s2sq = (F.conv2d(gt**2,      window, padding=pad, groups=ch) - mu2_sq).clamp(min=0)
        s12  =  F.conv2d(pred * gt,  window, padding=pad, groups=ch) - mu1_mu2

        ssim_map = ((2*mu1_mu2 + C1)*(2*s12 + C2)) / ((mu1_sq+mu2_sq+C1)*(s1sq+s2sq+C2))
        return 1.0 - ssim_map.mean()

    # ============================================================
    # Loss
    # ============================================================
    def _calculate_loss(self, pred, gt, gain_map, lpips_fn, edge_pred=None, inp_img=None):
        # 1. Reconstruction (L1)
        recon_l1 = F.l1_loss(pred, gt)

        # 2. SSIM
        ssim_loss = self._ssim_loss(pred, gt)

        # # 3. Color consistency
        # pred_hsv = rgb_to_hsv_torch(pred)
        # gt_hsv = rgb_to_hsv_torch(gt)
        #
        # color_loss = F.l1_loss(
        #     pred_hsv.mean(dim=[2, 3], keepdim=True),
        #     gt_hsv.mean(dim=[2, 3], keepdim=True)
        # )

        # # 3. Color consistency (HSV, pixel-wise, hue wrap-around L1)
        # pred_hsv = rgb_to_hsv_torch(pred)
        # gt_hsv = rgb_to_hsv_torch(gt)
        #
        # color_loss = F.l1_loss(pred_hsv, gt_hsv)

        # pred, gt: (B,3,H,W) in [0,1]
        # pred_hsv = rgb_to_hsv_torch(pred.clamp(0, 1))
        # gt_hsv = rgb_to_hsv_torch(gt.clamp(0, 1))
        #
        # ph, ps, pv = pred_hsv[:, 0:1], pred_hsv[:, 1:2], pred_hsv[:, 2:3]
        # gh, gs, gv = gt_hsv[:, 0:1], gt_hsv[:, 1:2], gt_hsv[:, 2:3]
        #
        # # Hue embedding: theta = 2πH, then (cos, sin)
        # theta_p = 2.0 * math.pi * ph
        # theta_g = 2.0 * math.pi * gh
        #
        # p_hue_vec = torch.cat([torch.cos(theta_p), torch.sin(theta_p)], dim=1)  # (B,2,H,W)
        # g_hue_vec = torch.cat([torch.cos(theta_g), torch.sin(theta_g)], dim=1)  # (B,2,H,W)
        #
        # # Pixel-wise L1 on hue-vector (circularly consistent)
        # hue_loss = (p_hue_vec - g_hue_vec).abs().mean()
        #
        # # Pixel-wise L1 on S,V
        # s_loss = (ps - gs).abs().mean()# + (pv - gv).abs().mean()
        #
        # # Total
        # color_loss = hue_loss + s_loss

        color_loss = hsv_cylinder_cosine_pixelwise_loss(pred, gt)

        # LPIPS Loss
        pred_lp = pred.clamp(0, 1) * 2 - 1
        gt_lp = gt.clamp(0, 1) * 2 - 1

        lpips_loss = lpips_fn(pred_lp, gt_lp).mean()

        # # 5. Frequency loss
        # pred_fft = torch.fft.rfft2(pred, norm='ortho')
        # gt_fft   = torch.fft.rfft2(gt,   norm='ortho')
        # freq_loss = F.l1_loss(torch.abs(pred_fft), torch.abs(gt_fft))

        total_loss = (
            recon_l1   +
            ssim_loss +
            lpips_loss +
            color_loss
            # 0.05 * freq_loss
        )

        # ===== (Aux) Edge loss =====
        edge_loss = pred.new_tensor(0.0)
        if edge_pred is not None:
            gt_y = self._luma(gt)
            gt_edge = self._sobel_mag(gt_y)  # (B,1,H,W)

            # 0..1로 정규화 (이미지별 max로 나눔)
            denom = gt_edge.detach().amax(dim=[2, 3], keepdim=True).clamp_min(1e-6)
            gt_edge_norm = gt_edge / denom

            edge_loss = F.l1_loss(edge_pred, gt_edge_norm)
            total_loss = total_loss + edge_loss

        loss_dict = {
            "total":  total_loss.item(),
            "recon":  recon_l1.item(),
            "ssim":   ssim_loss.item(),
            "lpips": lpips_loss.item(),
            "color":  color_loss.item(),
            "edge": edge_loss.item(),  # 추가
            # "freq":   freq_loss.item(),
        }
        return total_loss, loss_dict

    def _luma(self, x):
        # x: (B,3,H,W) in [0,1]
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    def _sobel_mag(self, y):
        # y: (B,1,H,W)
        kx = y.new_tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]]).view(1, 1, 3, 3)
        ky = y.new_tensor([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]]).view(1, 1, 3, 3)
        gx = F.conv2d(y, kx, padding=1)
        gy = F.conv2d(y, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

def hsv_cylinder_cosine_pixelwise_loss(pred_rgb, gt_rgb, eps=1e-6, beta=1.0):
    """
    pred_rgb, gt_rgb: (B,3,H,W) in [0,1]
    returns scalar loss
    """
    pred_hsv = rgb_to_hsv_torch(pred_rgb.clamp(0, 1))
    gt_hsv   = rgb_to_hsv_torch(gt_rgb.clamp(0, 1))

    ph, ps, pv = pred_hsv[:, 0:1], pred_hsv[:, 1:2], pred_hsv[:, 2:3]
    gh, gs, gv = gt_hsv[:, 0:1], gt_hsv[:, 1:2], gt_hsv[:, 2:3]

    theta_p = 2.0 * math.pi * ph
    theta_g = 2.0 * math.pi * gh

    vp = torch.cat([ps * torch.cos(theta_p),
                    ps * torch.sin(theta_p),
                    beta * pv], dim=1)  # (B,3,H,W)

    vg = torch.cat([gs * torch.cos(theta_g),
                    gs * torch.sin(theta_g),
                    beta * gv], dim=1)  # (B,3,H,W)

    # cosine similarity pixel-wise
    dot = (vp * vg).sum(dim=1, keepdim=True)  # (B,1,H,W)
    vp_norm = torch.sqrt((vp * vp).sum(dim=1, keepdim=True) + eps)
    vg_norm = torch.sqrt((vg * vg).sum(dim=1, keepdim=True) + eps)

    cos_sim = dot / (vp_norm * vg_norm + eps)
    cos_sim = cos_sim.clamp(-1.0, 1.0)

    loss = (1.0 - cos_sim).mean()
    return loss

def hs_cylinder_cosine_pixelwise_loss(pred_rgb, gt_rgb, eps=1e-6):
    # pred_rgb, gt_rgb: (B,3,H,W) in [0,1]
    pred_hsv = rgb_to_hsv_torch(pred_rgb.clamp(0, 1))
    gt_hsv   = rgb_to_hsv_torch(gt_rgb.clamp(0, 1))

    ph, ps = pred_hsv[:, 0:1], pred_hsv[:, 1:2]   # Hue, Sat
    gh, gs = gt_hsv[:, 0:1], gt_hsv[:, 1:2]

    theta_p = 2.0 * math.pi * ph
    theta_g = 2.0 * math.pi * gh

    # HS vector on a cylinder: [S*cos, S*sin, 1]
    ones_p = torch.ones_like(ps)
    ones_g = torch.ones_like(gs)

    vp = torch.cat([ps * torch.cos(theta_p), ps * torch.sin(theta_p), ones_p], dim=1)  # (B,3,H,W)
    vg = torch.cat([gs * torch.cos(theta_g), gs * torch.sin(theta_g), ones_g], dim=1)  # (B,3,H,W)

    # cosine similarity pixel-wise
    dot = (vp * vg).sum(dim=1, keepdim=True)  # (B,1,H,W)
    vp_norm = torch.sqrt((vp * vp).sum(dim=1, keepdim=True) + eps)
    vg_norm = torch.sqrt((vg * vg).sum(dim=1, keepdim=True) + eps)

    cos_sim = dot / (vp_norm * vg_norm + eps)
    # optional clamp for numerical stability
    cos_sim = cos_sim.clamp(-1.0, 1.0)

    loss = (1.0 - cos_sim).mean()
    return loss


# ============================================================
# Utility
# ============================================================
def get_model_size(model):
    param_size  = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)

if __name__ == "__main__":
    model = NanoLLE(base_channels=24, depths=(2, 2, 3), gn_groups=8)
    print(f"Model Size: {get_model_size(model):.3f} MB")
    print(f"Total Params: {sum(p.numel() for p in model.parameters()):,}")

    dummy = {
        "LQ_image": torch.randn(1, 3, 256, 256).clamp(0, 1),
        "HQ_image": torch.randn(1, 3, 256, 256).clamp(0, 1),
    }
    with torch.no_grad():
        out = model(dummy)
    print(f"pred shape : {out['prediction'].shape}")
    print(f"loss       : {out['loss'].item():.4f}")
    print(f"loss_dict  : {out['loss_dict']}")