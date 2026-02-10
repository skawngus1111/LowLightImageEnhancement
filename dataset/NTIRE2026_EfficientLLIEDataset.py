# import os
# import random
#
# import torch
# from torch.utils.data import Dataset
# import torchvision.transforms as transforms
#
# import numpy as np
# from glob import glob
# from PIL import Image
# from natsort import natsorted
#
# class NTIRE2026_EfficientLLIETrainDataset(Dataset):
#     def __init__(self, data_dir, transform=None):
#         super(NTIRE2026_EfficientLLIETrainDataset, self).__init__()
#
#         self.dataset_dir = os.path.join(data_dir, 'train')
#         self.transform = transform
#         self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#
#         low_quality_folder = os.path.join(self.dataset_dir, 'low-20260203T115952Z-3-001', 'low')
#         high_quality_folder = os.path.join(self.dataset_dir, 'normal-20260203T115952Z-3-001', 'normal')
#
#         self.low_quality_folder_list = natsorted(glob(os.path.join(low_quality_folder, '*.jpg')))
#         self.high_quality_folder_list = natsorted(glob(os.path.join(high_quality_folder, '*.jpg')))
#
#         assert len(self.low_quality_folder_list) == len(self.high_quality_folder_list)
#
#     def __len__(self):
#         return len(self.low_quality_folder_list)
#
#     def __getitem__(self, index):
#         low_quality_image_path = self.low_quality_folder_list[index]
#         high_quality_image_path = self.high_quality_folder_list[index]
#
#         low_quality_image = Image.open(low_quality_image_path).convert('RGB')
#         high_quality_image = Image.open(high_quality_image_path).convert('RGB')
#
#         # import matplotlib.pyplot as plt
#         # fig, ax = plt.subplots(1, 2)
#         # ax[0].imshow(low_quality_image)
#         # ax[1].imshow(high_quality_image)
#         # plt.show()
#
#         seed = random.randint(1, 1000000)
#         seed = np.random.randint(seed) # make a seed with numpy generator
#
#         if self.transform:
#             random.seed(seed) # apply this seed to img tranfsorms
#             torch.manual_seed(seed) # needed for torchvision 0.7
#             low_quality_image = self.transform(low_quality_image)
#             random.seed(seed)
#             torch.manual_seed(seed)
#             high_quality_image = self.transform(high_quality_image)
#
#         data_batch = {
#             "LQ_image": low_quality_image, "HQ_image": high_quality_image,
#         }
#
#         return data_batch
#
# class NTIRE2026_EfficientLLIEValDataset(Dataset):
#     def __init__(self, data_dir, transform=None):
#         super(NTIRE2026_EfficientLLIEValDataset, self).__init__()
#
#         self.dataset_dir = os.path.join(data_dir, 'val')
#         self.transform = transform
#         self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#
#         low_quality_folder = os.path.join(self.dataset_dir, 'low-20260203T123949Z-3-001', 'low')
#         self.low_quality_folder_list = natsorted(glob(os.path.join(low_quality_folder, '*.jpg')))
#
#     def __len__(self):
#         return len(self.low_quality_folder_list)
#
#     def __getitem__(self, index):
#         low_quality_image_path = self.low_quality_folder_list[index]
#         low_quality_image = Image.open(low_quality_image_path).convert('RGB')
#
#         seed = random.randint(1, 1000000)
#         seed = np.random.randint(seed) # make a seed with numpy generator
#
#         if self.transform:
#             random.seed(seed) # apply this seed to img tranfsorms
#             torch.manual_seed(seed) # needed for torchvision 0.7
#             low_quality_image = self.transform(low_quality_image)
#
#         data_batch = {
#             "LQ_image": low_quality_image, "HQ_image": low_quality_image,
#         }
#
#         return data_batch

import os
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import numpy as np
from glob import glob
from PIL import Image
from natsort import natsorted


# ============= 저조도 특화 Augmentation =============
class LowLightAugmentation:
    """
    저조도 이미지 향상을 위한 맞춤 Augmentation
    - Geometric: Crop, Flip, Rotation
    - Photometric: Gamma, Noise, Exposure
    """

    def __init__(self,
                 img_size=256,
                 use_geometric=True,
                 use_photometric=True,
                 gamma_range=(0.7, 1.3),
                 noise_prob=0.3,
                 noise_level=(0.01, 0.03),
                 color_jitter_prob=0.3):

        self.img_size = img_size
        self.use_geometric = use_geometric
        self.use_photometric = use_photometric

        # Photometric augmentation parameters
        self.gamma_range = gamma_range
        self.noise_prob = noise_prob
        self.noise_level = noise_level
        self.color_jitter_prob = color_jitter_prob

    def __call__(self, lq_img, hq_img):
        """
        Args:
            lq_img: PIL Image (low quality)
            hq_img: PIL Image (high quality)
        Returns:
            lq_tensor, hq_tensor: Augmented tensors
        """
        # Convert to tensor first
        lq = TF.to_tensor(lq_img)
        hq = TF.to_tensor(hq_img)

        # ===== 1. Geometric Augmentation (동일하게 적용!) =====
        if self.use_geometric:
            lq, hq = self._geometric_augmentation(lq, hq)

        # ===== 2. Photometric Augmentation (LQ에만 선택적으로) =====
        if self.use_photometric:
            lq = self._photometric_augmentation(lq)

        return lq, hq

    def _geometric_augmentation(self, lq, hq):
        """
        기하학적 변환 (LQ, HQ 동일하게!)
        """
        # Random Crop
        if lq.shape[1] > self.img_size and lq.shape[2] > self.img_size:
            i = random.randint(0, lq.shape[1] - self.img_size)
            j = random.randint(0, lq.shape[2] - self.img_size)

            lq = lq[:, i:i + self.img_size, j:j + self.img_size]
            hq = hq[:, i:i + self.img_size, j:j + self.img_size]
        else:
            # Resize if smaller than target
            lq = TF.resize(lq, (self.img_size, self.img_size))
            hq = TF.resize(hq, (self.img_size, self.img_size))

        # Random Horizontal Flip
        if random.random() > 0.5:
            lq = TF.hflip(lq)
            hq = TF.hflip(hq)

        # Random Vertical Flip
        if random.random() > 0.5:
            lq = TF.vflip(lq)
            hq = TF.vflip(hq)

        # Random Rotation (90도 단위)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            lq = TF.rotate(lq, angle)
            hq = TF.rotate(hq, angle)

        return lq, hq

    def _photometric_augmentation(self, img):
        """
        광학적 변환 (LQ 이미지에만)
        """
        # 1. Gamma Correction (밝기 변화)
        if random.random() > 0.3:
            gamma = random.uniform(*self.gamma_range)
            img = torch.pow(img.clamp(1e-6, 1.0), gamma)

        # 2. Gaussian Noise (저조도 노이즈 시뮬레이션)
        if random.random() < self.noise_prob:
            noise_level = random.uniform(*self.noise_level)
            noise = torch.randn_like(img) * noise_level
            img = (img + noise).clamp(0, 1)

        # 3. Color Jitter (약하게)
        if random.random() < self.color_jitter_prob:
            # Brightness
            if random.random() > 0.5:
                factor = random.uniform(0.9, 1.1)
                img = (img * factor).clamp(0, 1)

            # Saturation
            if random.random() > 0.5:
                gray = img.mean(dim=0, keepdim=True)
                factor = random.uniform(0.9, 1.1)
                img = (gray + (img - gray) * factor).clamp(0, 1)

        return img


# ============= 개선된 Train Dataset =============
class NTIRE2026_EfficientLLIETrainDataset(Dataset):
    def __init__(self,
                 data_dir,
                 img_size=256,
                 use_augmentation=True,
                 gamma_range=(0.7, 1.3),
                 noise_prob=0.3,
                 noise_level=(0.01, 0.03)):

        super(NTIRE2026_EfficientLLIETrainDataset, self).__init__()

        self.dataset_dir = os.path.join(data_dir, 'train')
        self.img_size = img_size
        self.use_augmentation = use_augmentation

        low_quality_folder = os.path.join(self.dataset_dir, 'low-20260203T115952Z-3-001', 'low')
        high_quality_folder = os.path.join(self.dataset_dir, 'normal-20260203T115952Z-3-001', 'normal')

        self.low_quality_folder_list = natsorted(glob(os.path.join(low_quality_folder, '*.jpg')))
        self.high_quality_folder_list = natsorted(glob(os.path.join(high_quality_folder, '*.jpg')))

        assert len(self.low_quality_folder_list) == len(self.high_quality_folder_list), \
            f"Mismatch: {len(self.low_quality_folder_list)} LQ vs {len(self.high_quality_folder_list)} HQ"

        # Augmentation 초기화
        if self.use_augmentation:
            self.augmentation = LowLightAugmentation(
                img_size=img_size,
                use_geometric=True,
                use_photometric=True,
                gamma_range=gamma_range,
                noise_prob=noise_prob,
                noise_level=noise_level
            )
        else:
            # Augmentation 없이 resize만
            self.to_tensor = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.low_quality_folder_list)

    def __getitem__(self, index):
        low_quality_image_path = self.low_quality_folder_list[index]
        high_quality_image_path = self.high_quality_folder_list[index]

        low_quality_image = Image.open(low_quality_image_path).convert('RGB')
        high_quality_image = Image.open(high_quality_image_path).convert('RGB')

        # Augmentation 적용
        if self.use_augmentation:
            low_quality_tensor, high_quality_tensor = self.augmentation(
                low_quality_image, high_quality_image
            )
        else:
            low_quality_tensor = self.to_tensor(low_quality_image)
            high_quality_tensor = self.to_tensor(high_quality_image)

        data_batch = {
            "LQ_image": low_quality_tensor,
            "HQ_image": high_quality_tensor,
        }

        return data_batch


# ============= 개선된 Val Dataset =============
class NTIRE2026_EfficientLLIEValDataset(Dataset):
    def __init__(self, data_dir, img_size=256):
        super(NTIRE2026_EfficientLLIEValDataset, self).__init__()

        self.dataset_dir = os.path.join(data_dir, 'val')
        self.img_size = img_size

        low_quality_folder = os.path.join(self.dataset_dir, 'low-20260203T123949Z-3-001', 'low')
        self.low_quality_folder_list = natsorted(glob(os.path.join(low_quality_folder, '*.jpg')))

        # Validation은 augmentation 없이 resize만
        self.transform = transforms.Compose([
           # transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.low_quality_folder_list)

    def __getitem__(self, index):
        low_quality_image_path = self.low_quality_folder_list[index]
        low_quality_image = Image.open(low_quality_image_path).convert('RGB')

        low_quality_tensor = self.transform(low_quality_image)

        data_batch = {
            "LQ_image": low_quality_tensor,
            "HQ_image": low_quality_tensor,  # Validation에는 GT 없음
        }

        return data_batch


# ============= 사용 예시 =============
if __name__ == "__main__":
    # 기본 설정으로 Dataset 생성
    train_dataset = NTIRE2026_EfficientLLIETrainDataset(
        data_dir='./data',
        img_size=256,
        use_augmentation=True
    )

    # DataLoader 생성
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 테스트
    print(f"Total training samples: {len(train_dataset)}")

    batch = next(iter(train_loader))
    print(f"LQ shape: {batch['LQ_image'].shape}")
    print(f"HQ shape: {batch['HQ_image'].shape}")

    # Augmentation 효과 확인
    import matplotlib.pyplot as plt

    sample = train_dataset[0]
    lq = sample['LQ_image'].permute(1, 2, 0).numpy()
    hq = sample['HQ_image'].permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(lq)
    axes[0].set_title('Augmented LQ')
    axes[0].axis('off')

    axes[1].imshow(hq)
    axes[1].set_title('Augmented HQ')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('augmentation_sample.png')
    print("Saved augmentation_sample.png")