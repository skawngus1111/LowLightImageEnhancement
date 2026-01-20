import os

import torch

from utils.get_functions import get_save_path

def save_model(args, model):
    save_model_path, _ = get_save_path(args)
    save_model_path = os.path.join(save_model_path, 'model_weights', 'model_weight.pth')
    torch.save(model.state_dict(), save_model_path)