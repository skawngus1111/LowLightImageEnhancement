import os
import csv

import torch

import numpy as np
import matplotlib.pyplot as plt

from utils.get_functions import get_save_path

def save_model(args, model):
    save_model_path = get_save_path(args)
    save_model_path = os.path.join(save_model_path, 'model_weights')
    os.makedirs(save_model_path, exist_ok=True)
    print(save_model_path)

    torch.save(model.state_dict(), os.path.join(save_model_path, 'model_weight_epoch{}.pth'.format(args.current_epoch)))

# def save_predictions(args, predictions, idx):
#     save_model_path, _ = get_save_path(args)
#     prediction_path = os.path.join(save_model_path, 'predictions')
#     os.makedirs(prediction_path, exist_ok=True)
#     predictions = np.transpose(predictions.squeeze().cpu().detach().numpy(), (1, 2, 0))
#
#     fname = f"{idx+1:04d}.jpg"   # 1 -> 001, 2 -> 002, 12 -> 012, 123 -> 123
#     plt.imsave(os.path.join(prediction_path, fname), predictions)

import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

def _to_hwc_uint8(predictions):
    """
    predictions: torch tensor [B,C,H,W] or [C,H,W] or numpy
    return: uint8 image [H,W,3] or [H,W] (for grayscale)
    """
    if torch.is_tensor(predictions):
        pred = predictions.detach().float()
        if pred.dim() == 4:
            pred = pred[0]  # first in batch

        # [C,H,W] -> [H,W,C]
        if pred.dim() == 3 and pred.size(0) in (1, 3):
            pred = pred.permute(1, 2, 0)

        pred = pred.clamp(0, 1).cpu().numpy()
    else:
        pred = np.asarray(predictions, dtype=np.float32)
        pred = np.clip(pred, 0.0, 1.0)

    # ensure HWC or HW
    if pred.ndim == 3 and pred.shape[2] == 1:
        pred = pred[:, :, 0]  # HW

    # float [0,1] -> uint8
    pred_u8 = (pred * 255.0 + 0.5).astype(np.uint8)
    return pred_u8

def _draw_metrics_on_pil(img_pil, text, margin=10):
    """
    img_pil: PIL.Image
    text: str (multi-line allowed)
    """
    draw = ImageDraw.Draw(img_pil)

    # font: try truetype, fallback to default
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()

    # measure text box
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    x, y = margin, margin
    pad = 6

    # semi-transparent background (RGBA) for readability
    if img_pil.mode != "RGBA":
        img_pil = img_pil.convert("RGBA")
        draw = ImageDraw.Draw(img_pil)

    bg = Image.new("RGBA", (tw + 2*pad, th + 2*pad), (0, 0, 0, 140))
    img_pil.paste(bg, (x, y), bg)

    # white text with optional stroke for extra readability
    draw.multiline_text(
        (x + pad, y + pad),
        text,
        font=font,
        fill=(255, 255, 255, 255),
        spacing=4,
        stroke_width=2,
        stroke_fill=(0, 0, 0, 255),
    )
    return img_pil

def save_predictions(args, predictions, idx, validation=False, psnr=None, ssim=None):
    save_model_path = get_save_path(args)

    if validation:
        prediction_path = os.path.join(save_model_path, f'predictions_valid_epoch{args.current_epoch}')
    else:
        prediction_path = os.path.join(save_model_path, f'predictions_epoch{args.current_epoch}')

    os.makedirs(prediction_path, exist_ok=True)

    fname = f"{idx + 1:04d}.jpg"  # 0001, 0002, ...

    pred_u8 = _to_hwc_uint8(predictions)

    # numpy -> PIL
    if pred_u8.ndim == 2:
        img_pil = Image.fromarray(pred_u8, mode="L").convert("RGB")
    else:
        img_pil = Image.fromarray(pred_u8, mode="RGB")

    # validation일 때만 텍스트 오버레이
    if validation:
        # psnr/ssim이 tensor/np일 수도 있으니 float로 캐스팅
        def _to_float(x):
            if x is None:
                return None
            if torch.is_tensor(x):
                return float(x.detach().cpu().item())
            if isinstance(x, (np.ndarray,)):
                return float(x.item()) if x.size == 1 else float(np.mean(x))
            return float(x)

        psnr_f = _to_float(psnr)
        ssim_f = _to_float(ssim)

        lines = []
        if psnr_f is not None:
            lines.append(f"PSNR: {psnr_f:.2f} dB")
        if ssim_f is not None:
            lines.append(f"SSIM: {ssim_f:.4f}")

        if len(lines) > 0:
            text = "\n".join(lines)
            img_pil = _draw_metrics_on_pil(img_pil, text)

    # 저장 (RGBA였다면 JPG 저장 위해 RGB로)
    if img_pil.mode == "RGBA":
        img_pil = img_pil.convert("RGB")

    img_pil.save(os.path.join(prediction_path, fname), quality=95)

def save_loss_graph(args, train_loss_dict, val_loss_dict):
    save_model_path = get_save_path(args)
    save_path = os.path.join(save_model_path, f'loss_{args.final_epoch}')
    os.makedirs(save_path, exist_ok=True)

    keys = ['total_loss', 'recon_loss', 'ssim_loss', 'lpips_loss', 'color_loss', 'edge_loss']

    # ---------- basic sanity ----------
    # epoch 길이는 가장 짧은 쪽에 맞춤(혹시 중간에 누락되면 에러 방지)
    def _safe_len(d, k):
        v = d.get(k, [])
        return 0 if v is None else len(v)

    n_train = min([_safe_len(train_loss_dict, k) for k in keys] + [10**9])
    n_val   = min([_safe_len(val_loss_dict, k) for k in keys] + [10**9])

    # train/val 둘 다 존재하는 최소 epoch 길이
    n = min(n_train, n_val)
    if n == 0:
        print("[save_loss_graph] No loss history to plot.")
        return

    epochs = np.arange(1, n + 1)

    # ---------- save CSV ----------
    csv_path = os.path.join(save_path, "loss_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["epoch"] + [f"train_{k}" for k in keys] + [f"val_{k}" for k in keys]
        writer.writerow(header)
        for i in range(n):
            row = [int(epochs[i])]
            row += [float(train_loss_dict[k][i]) for k in keys]
            row += [float(val_loss_dict[k][i]) for k in keys]
            writer.writerow(row)

    # ---------- one figure with all losses ----------
    plt.figure(figsize=(12, 7))
    for k in keys:
        plt.plot(epochs, train_loss_dict[k][:n], label=f"train_{k}")
        plt.plot(epochs, val_loss_dict[k][:n], linestyle="--", label=f"val_{k}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Val Loss Trend")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "loss_all.png"), dpi=200)
    plt.close()

    # ---------- per-loss figures ----------
    for k in keys:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_loss_dict[k][:n], label=f"train_{k}")
        plt.plot(epochs, val_loss_dict[k][:n], linestyle="--", label=f"val_{k}")
        plt.xlabel("Epoch")
        plt.ylabel(k)
        plt.title(f"{k} Trend")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"loss_{k}.png"), dpi=200)
        plt.close()

    print(f"[save_loss_graph] Saved plots + csv to: {save_path}")