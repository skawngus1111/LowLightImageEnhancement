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

# utils/save_functions.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def save_predictions(args, predictions, idx):
    save_model_path, _ = get_save_path(args)
    prediction_path = os.path.join(save_model_path, 'predictions_epoch{}'.format(args.current_epoch), )
    os.makedirs(prediction_path, exist_ok=True)

    fname = f"{idx + 1:04d}.jpg"  # 원하는 포맷 (001,002,...)

    # predictions: torch tensor [B,C,H,W] or [C,H,W] or numpy
    if torch.is_tensor(predictions):
        pred = predictions.detach().float()

        # 배치면 첫 장 저장 (원하면 idx로 매핑해서 바꿔도 됨)
        if pred.dim() == 4:
            pred = pred[0]

        # [C,H,W] -> [H,W,C]
        if pred.dim() == 3 and pred.size(0) in (1, 3):
            pred = pred.permute(1, 2, 0)

        pred = pred.clamp(0, 1).cpu().numpy()
    else:
        pred = predictions
        pred = np.asarray(pred, dtype=np.float32)
        pred = np.clip(pred, 0.0, 1.0)

    plt.imsave(os.path.join(prediction_path, fname), pred)
    plt.close()