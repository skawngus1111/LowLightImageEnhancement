import os

import torch

def get_device():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"You are using \"{device}\" device.")
    return device

def get_save_path(args):
    save_model_path = '{}/{}_{}x{}_{}_{}_{}({}_{})'.format(
        args.model_name,
        args.data_type,
        str(args.image_size), str(args.image_size),
        str(args.train_batch_size),
        args.model_name,
        args.optimizer_name, args.lr, str(args.final_epoch).zfill(3)
    )

    if args.multi_scale_train: save_model_path += '_MS'

    save_model_path = os.path.join(save_model_path, f"{args.our_method_configuration}")

    model_dirs = os.path.join(args.save_path, save_model_path)
    # print(model_dirs)
    os.makedirs(os.path.join(model_dirs, 'model_weights'), exist_ok=True)
    os.makedirs(os.path.join(model_dirs, 'test_reports'), exist_ok=True)

    return save_model_path

    # save_model_path = os.path.join(args.save_path, "model_weight", args.model_name, args.data_type)
    # save_plot_path = os.path.join(args.save_path, "QualitativeResults", args.model_name, args.data_type)
    #
    # os.makedirs(os.path.join(save_model_path, 'model_weights'), exist_ok=True)
    # os.makedirs(os.path.join(save_model_path, 'test_reports'), exist_ok=True)
    #
    # return save_model_path, save_plot_path