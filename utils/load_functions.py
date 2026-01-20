import os

import torch

from packaging import version

from model_list import model_to_device
from utils.get_functions import get_save_path

def load_model(args, model) :
    model_dirs, _ = get_save_path(args)

    load_path = os.path.join(model_dirs, 'model_weights/model_weight.pth')

    print("Your model is loaded from {}.".format(load_path))
    # Version 1: If the checkpoint is saved with 'weights_only=True', use the following line:
    if version.parse(torch.__version__) >= version.parse("2.0.0"): checkpoint = torch.load(load_path, weights_only=False)
    else: checkpoint = torch.load(load_path)

    model.load_state_dict(checkpoint)
    model = model_to_device(args, model)

    return model