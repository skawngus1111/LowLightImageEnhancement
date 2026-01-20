import os

import torch

def get_device():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"You are using \"{device}\" device.")
    return device

def get_save_path(args):
    save_model_path = os.path.join(args.save_path, "model_weight", args.model_name, args.data_type)
    save_plot_path = os.path.join(args.save_path, "QualitativeResults", args.model_name, args.data_type)

    os.makedirs(os.path.join(save_model_path, 'model_weights'), exist_ok=True)
    os.makedirs(os.path.join(save_model_path, 'test_reports'), exist_ok=True)

    return save_model_path, save_plot_path