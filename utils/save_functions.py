import os

import torch

import numpy as np
import matplotlib.pyplot as plt

from utils.get_functions import get_save_path

def save_model(args, model):
    save_model_path, _ = get_save_path(args)
    save_model_path = os.path.join(save_model_path, 'model_weights', 'model_weight_epoch{}.pth'.format(args.current_epoch))
    torch.save(model.state_dict(), save_model_path)

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
        font = ImageFont.truetype("DejaVuSans.ttf", 50)
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
    save_model_path, _ = get_save_path(args)

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
