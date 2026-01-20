import os
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import numpy as np

from dataset.dataset_utils import is_image_file, load_img

class LOLDatasetFromFolderTrain(Dataset):
    def __init__(self, data_dir, transform=None):
        super(LOLDatasetFromFolderTrain, self).__init__()

        self.data_dir = os.path.join(data_dir, 'lol_dataset/our485')
        self.transform = transform
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):
        folder = self.data_dir+'/low'
        folder2= self.data_dir+'/high'
        data_filenames = [os.path.join(folder, x) for x in os.listdir(folder) if is_image_file(x)]
        data_filenames2 = [os.path.join(folder2, x) for x in os.listdir(folder2) if is_image_file(x)]

        im1 = load_img(data_filenames[index])
        im2 = load_img(data_filenames2[index])

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(im1)
        # ax[1].imshow(im2)
        # plt.show()

        _, file1 = os.path.split(data_filenames[index])
        _, file2 = os.path.split(data_filenames2[index])

        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator

        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)
            im2 = self.transform(im2)

        data_batch = {
            "LQ_image": im1, "HQ_image": im2,
            "LQ_image_path": file1, "HQ_image_path": file2,
        }

        return data_batch

    def __len__(self):
        return 485

class LOLDatasetFromFolderTest(Dataset):
    def __init__(self, data_dir, transform=None):
        super(LOLDatasetFromFolderTest, self).__init__()

        self.data_dir = os.path.join(data_dir, 'lol_dataset/eval15')
        self.transform = transform
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):
        folder = self.data_dir+'/low'
        folder2= self.data_dir+'/high'
        data_filenames = [os.path.join(folder, x) for x in os.listdir(folder) if is_image_file(x)]
        data_filenames2 = [os.path.join(folder2, x) for x in os.listdir(folder2) if is_image_file(x)]

        im1 = load_img(data_filenames[index])
        im2 = load_img(data_filenames2[index])

        _, file1 = os.path.split(data_filenames[index])
        _, file2 = os.path.split(data_filenames2[index])

        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator

        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)
            im2 = self.transform(im2)

        data_batch = {
            "LQ_image": im1, "HQ_image": im2,
            "LQ_image_path": file1, "HQ_image_path": file2,
        }

        return data_batch


    def __len__(self):
        return 15