import torch
import torch.nn as nn

from model_list.uformer import UFormer
from model_list.restormer import Restormer
from model_list.nano_llienet import NanoLLE

MODEL_ZOO = {
    "UFormer": UFormer,
    "Restormer": Restormer,
    "NanoLLIE": NanoLLE
}

def low_light_image_enhancement_model(args):
    if args.model_name not in MODEL_ZOO:
        raise ValueError(f"Unknown model_name: {args.model_name}. Available: {list(MODEL_ZOO.keys())}")
    model = MODEL_ZOO[args.model_name]()  # instantiate

    return model_to_device(args, model)

def model_to_device(args, model):
    if torch.cuda.device_count() > 1:
        print('Multi GPU activate : {}'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model).to(args.device)
    else:
        model = model.to(args.device)
    return model