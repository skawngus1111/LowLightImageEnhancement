import argparse

from LLIEExperiment.low_light_image_enhancement import LowLightImageEnhancement

def main(args):
    print("Hello! We start experiment for Low-Light Image Enhancement!")

    low_light_image_enhancement_experiment = LowLightImageEnhancement(args)
    if args.train: low_light_image_enhancement_experiment.fit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Following are the arguments that can be passed form the terminal itself!')

    parser.add_argument('--data_path', type=str, default='/media/jhnam0514/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/AwesomeDeepLearning/dataset/LLIE_Dataset')
    parser.add_argument('--save_path', type=str, default='/hdd_ext/hdd6000/LowLightImageEnhancement')

    parser.add_argument('--data_type', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--amp', default=False, action='store_true')
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    main(args)