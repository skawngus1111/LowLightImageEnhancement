import os
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import numpy as np
from glob import glob
from PIL import Image
from natsort import natsorted

class NTIRE2026_EfficientLLIETrainDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super(NTIRE2026_EfficientLLIETrainDataset, self).__init__()

        self.dataset_dir = os.path.join(data_dir, 'train')
        self.transform = transform
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        low_quality_folder = os.path.join(self.dataset_dir, 'low-20260203T115952Z-3-001', 'low')
        high_quality_folder = os.path.join(self.dataset_dir, 'normal-20260203T115952Z-3-001', 'normal')

        self.low_quality_folder_list = natsorted(glob(os.path.join(low_quality_folder, '*.jpg')))
        self.high_quality_folder_list = natsorted(glob(os.path.join(high_quality_folder, '*.jpg')))

        assert len(self.low_quality_folder_list) == len(self.high_quality_folder_list)

    def __len__(self):
        return len(self.low_quality_folder_list)

    def __getitem__(self, index):
        low_quality_image_path = self.low_quality_folder_list[index]
        high_quality_image_path = self.high_quality_folder_list[index]

        low_quality_image = Image.open(low_quality_image_path).convert('RGB')
        high_quality_image = Image.open(high_quality_image_path).convert('RGB')

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(low_quality_image)
        # ax[1].imshow(high_quality_image)
        # plt.show()

        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator

        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            low_quality_image = self.transform(low_quality_image)
            random.seed(seed)
            torch.manual_seed(seed)
            high_quality_image = self.transform(high_quality_image)

        data_batch = {
            "LQ_image": low_quality_image, "HQ_image": high_quality_image,
        }

        return data_batch

class NTIRE2026_EfficientLLIEValDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super(NTIRE2026_EfficientLLIEValDataset, self).__init__()

        self.dataset_dir = os.path.join(data_dir, 'val')
        self.transform = transform
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        low_quality_folder = os.path.join(self.dataset_dir, 'low-20260203T123949Z-3-001', 'low')
        self.low_quality_folder_list = natsorted(glob(os.path.join(low_quality_folder, '*.jpg')))

    def __len__(self):
        return len(self.low_quality_folder_list)

    def __getitem__(self, index):
        low_quality_image_path = self.low_quality_folder_list[index]
        low_quality_image = Image.open(low_quality_image_path).convert('RGB')

        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator

        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            low_quality_image = self.transform(low_quality_image)

        data_batch = {
            "LQ_image": low_quality_image, "HQ_image": low_quality_image,
        }

        return data_batch