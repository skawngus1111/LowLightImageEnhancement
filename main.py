import argparse

from LLIEExperiment.low_light_image_enhancement import LowLightImageEnhancement

def main(args):
    print("Hello! We start experiment for Low-Light Image Enhancement!")

    low_light_image_enhancement_experiment = LowLightImageEnhancement(args)
    if args.train: low_light_image_enhancement_experiment.fit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Following are the arguments that can be passed form the terminal itself!')

    parser.add_argument('--data_path', type=str, default='/media/jhnam0514/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/AwesomeDeepLearning/dataset/LLIE_Dataset/NTIRE2026_Challenge_Dataset/EfficientLLIE')
    parser.add_argument('--save_path', type=str, default='/hdd_ext/hdd6000/Efficient_LowLightImageEnhancement_Challenge')

    # Train parameter
    parser.add_argument('--data_type', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--amp', default=False, action='store_true')
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--optimizer_name', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--final_epoch', type=int, default=100)
    parser.add_argument('--multi_scale_train', default=False, action='store_true')
    parser.add_argument(
        '--our_method_configuration', type=str,
        required=False, default='baseline'
    )

    args = parser.parse_args()

    import os
    import torch

    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count =", torch.cuda.device_count())
    print("current_device =", torch.cuda.current_device())
    print("device_name(0) =", torch.cuda.get_device_name(0))

    main(args)