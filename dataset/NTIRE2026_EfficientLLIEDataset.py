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
                 num_patches=8,
                 gamma_range=(0.7, 1.3),
                 noise_prob=0.3,
                 noise_level=(0.01, 0.03),
                 color_jitter_prob=0.3):

        self.img_size = img_size
        self.use_geometric = use_geometric
        self.use_photometric = use_photometric
        self.num_patches = num_patches
        self.dark_candidates = 16  # 후보 개수 (8~32 권장)
        self.dark_min_mean = 0.0  # 예: 0.002로 두면 “완전 검정” 패치만 고르는 걸 완화

        # Photometric augmentation parameters
        self.gamma_range = gamma_range
        self.noise_prob = noise_prob
        self.noise_level = noise_level
        self.color_jitter_prob = color_jitter_prob

    def __call__(self, lq_img, hq_img):
        lq = TF.to_tensor(lq_img)  # (C,H,W)
        hq = TF.to_tensor(hq_img)

        if self.use_geometric:
            lq, hq = self._geometric_augmentation(lq, hq)  # ✅ 이제 (K,C,S,S) 또는 (C,S,S)

        if self.use_photometric:
            # LQ에만 적용 (K개면 각 패치별로 적용)
            if lq.dim() == 4:  # (K,C,H,W)
                lq = torch.stack([self._photometric_augmentation(lq[k]) for k in range(lq.shape[0])], dim=0)
            else:
                lq = self._photometric_augmentation(lq)

        return lq, hq

    # def _geometric_augmentation(self, lq, hq):
    #     """
    #     기존 방식 유지:
    #     - 둘 다 img_size보다 크면 random crop
    #     - 아니면 resize(img_size,img_size)
    #     단, K(num_patches)개를 반환하되:
    #       - 앞에서 2개: 완전 랜덤
    #       - 나머지: 어두운 곳 위주(후보 M개 중 가장 어두운 patch 선택)
    #     """
    #     S = self.img_size
    #     K = self.num_patches
    #     assert K >= 1
    #
    #     # === 1) resize 케이스: 기존 로직대로 SxS로 만든 뒤 K개 반복 ===
    #     if not (lq.shape[1] > S and lq.shape[2] > S):
    #         lq = TF.resize(lq, (S, S))
    #         hq = TF.resize(hq, (S, S))
    #         if K == 1:
    #             return lq, hq
    #         return torch.stack([lq] * K, dim=0), torch.stack([hq] * K, dim=0)
    #
    #     # === 2) crop 가능한 케이스 ===
    #     H, W = lq.shape[1], lq.shape[2]
    #
    #     # 2 random + 2 dark가 기본. (K가 4가 아니면 가능한 범위에서 자동 조절)
    #     k_rand = min(2, K)
    #     k_dark = K - k_rand
    #
    #     # dark patch 고를 때 후보 수 (클수록 더 “진짜 어두운” patch를 고름. 8~32 추천)
    #     M = getattr(self, "dark_candidates", 16)
    #
    #     # 너무 완전한 검정(정보없는) patch만 고르는 걸 방지하고 싶으면 아래 임계값 사용
    #     # (필요 없으면 0.0으로 두면 됨)
    #     min_mean = getattr(self, "dark_min_mean", 0.0)  # 예: 0.002 정도
    #
    #     def luma_mean(patch):
    #         # patch: (C,S,S), TF.to_tensor 기준 RGB in [0,1]
    #         r = patch[0:1]
    #         g = patch[1:2]
    #         b = patch[2:3]
    #         y = 0.299 * r + 0.587 * g + 0.114 * b
    #         return float(y.mean())
    #
    #     def random_crop_pair():
    #         i = random.randint(0, H - S)
    #         j = random.randint(0, W - S)
    #         return lq[:, i:i + S, j:j + S], hq[:, i:i + S, j:j + S]
    #
    #     def dark_biased_crop_pair():
    #         # 후보 M개 중 "가장 어두운(luma mean 최소)" patch 선택
    #         best = None
    #         best_score = 1e9
    #
    #         for _ in range(M):
    #             i = random.randint(0, H - S)
    #             j = random.randint(0, W - S)
    #             lq_p = lq[:, i:i + S, j:j + S]
    #             score = luma_mean(lq_p)
    #
    #             # 너무 완전한 검정만 고르는 걸 피하고 싶으면 min_mean로 컷
    #             if score < min_mean:
    #                 continue
    #
    #             if score < best_score:
    #                 best_score = score
    #                 best = (lq_p, hq[:, i:i + S, j:j + S])
    #
    #         # min_mean 때문에 후보가 전부 걸러지는 예외 대비: 그냥 랜덤 하나 반환
    #         if best is None:
    #             return random_crop_pair()
    #         return best
    #
    #     # === 3) 패치 생성 ===
    #     lq_patches, hq_patches = [], []
    #
    #     # (a) 2장 랜덤
    #     for _ in range(k_rand):
    #         lp, hp = random_crop_pair()
    #         lq_patches.append(lp)
    #         hq_patches.append(hp)
    #
    #     # (b) 나머지 어두운 곳 위주
    #     for _ in range(k_dark):
    #         lp, hp = dark_biased_crop_pair()
    #         lq_patches.append(lp)
    #         hq_patches.append(hp)
    #
    #     # === 4) 반환 ===
    #     if K == 1:
    #         return lq_patches[0], hq_patches[0]
    #     return torch.stack(lq_patches, dim=0), torch.stack(hq_patches, dim=0)

    def _geometric_augmentation(self, lq, hq):
        """
        기존 방식 유지:
        - 둘 다 img_size보다 크면 random crop
        - 아니면 resize(img_size,img_size)

        단, K(num_patches)개를 만들 때:
        1) 후보 패치 M개를 랜덤 crop으로 생성
        2) 후보의 mean luma 분포를 K개의 구간으로 나눔(quantile 기반 -> bin이 비는 문제 최소화)
        3) 각 구간에서 1개씩 선택하여 총 K개 구성
        """
        S = self.img_size
        K = self.num_patches

        # ===== resize 케이스(기존 로직) =====
        if not (lq.shape[1] > S and lq.shape[2] > S):
            lq = TF.resize(lq, (S, S))
            hq = TF.resize(hq, (S, S))
            if K == 1:
                return lq, hq
            return torch.stack([lq] * K, dim=0), torch.stack([hq] * K, dim=0)

        H, W = lq.shape[1], lq.shape[2]

        # 후보 개수: K보다 충분히 크게 (너무 작으면 구간 나눠도 의미가 약함)
        # 보통 8K~16K 추천. K=64면 512~1024 정도.
        M = getattr(self, "brightness_candidates", max(8 * K, 64))

        # ---- luma mean ----
        def luma_mean(patch):
            # patch: (3,S,S) in [0,1]
            r = patch[0:1]
            g = patch[1:2]
            b = patch[2:3]
            y = 0.299 * r + 0.587 * g + 0.114 * b
            return y.mean()

        # ===== 1) 후보 패치 M개 생성 =====
        cand_i, cand_j = [], []
        mus = []

        for _ in range(M):
            i = random.randint(0, H - S)
            j = random.randint(0, W - S)
            p = lq[:, i:i + S, j:j + S]
            cand_i.append(i)
            cand_j.append(j)
            mus.append(luma_mean(p))

        mus = torch.stack(mus)  # (M,)

        # ===== 2) 밝기 분포를 K개 구간으로 나누기 (quantile bin) =====
        # torch.quantile이 없을 수도 있어서 안전하게 처리
        qs = torch.linspace(0.0, 1.0, steps=K + 1, device=mus.device)

        try:
            edges = torch.quantile(mus, qs)  # (K+1,)
        except Exception:
            # fallback: 정렬 후 index로 근사 quantile
            sorted_mus, _ = torch.sort(mus)
            idx = torch.clamp((qs * (M - 1)).round().long(), 0, M - 1)
            edges = sorted_mus[idx]

        # edges가 모두 같은 경우(후보가 거의 동일한 밝기): 그냥 랜덤 K개
        if torch.allclose(edges, edges[0]):
            pick = random.sample(range(M), k=min(K, M))
            lq_patches = [lq[:, cand_i[t]:cand_i[t] + S, cand_j[t]:cand_j[t] + S] for t in pick]
            hq_patches = [hq[:, cand_i[t]:cand_i[t] + S, cand_j[t]:cand_j[t] + S] for t in pick]
            # 부족하면 중복 허용해서 채움
            while len(lq_patches) < K:
                t = random.randrange(M)
                lq_patches.append(lq[:, cand_i[t]:cand_i[t] + S, cand_j[t]:cand_j[t] + S])
                hq_patches.append(hq[:, cand_i[t]:cand_i[t] + S, cand_j[t]:cand_j[t] + S])
            if K == 1:
                return lq_patches[0], hq_patches[0]
            return torch.stack(lq_patches, 0), torch.stack(hq_patches, 0)

        # bucketize로 각 후보를 0..K-1 bin에 할당
        # 경계는 edges[1:-1] 사용
        bins = torch.bucketize(mus, edges[1:-1], right=False)  # (M,), in [0, K-1]

        # ===== 3) 각 bin에서 1개씩 선택 =====
        selected = []
        used = torch.zeros((M,), dtype=torch.bool)

        for b in range(K):
            idxs = torch.nonzero((bins == b) & (~used), as_tuple=True)[0]
            if idxs.numel() == 0:
                selected.append(None)
                continue
            # bin 내부에서는 랜덤 1개 (네가 말한 "각 구간 별로 선택" 그대로)
            t = idxs[random.randrange(idxs.numel())].item()
            selected.append(t)
            used[t] = True

        # ===== 4) 빈 bin 채우기 (fallback) =====
        # 빈 bin이 있다면, 아직 안 쓴 후보 중에서 "그 bin의 대표 밝기"에 가장 가까운 걸 넣음
        # 대표 밝기: (edges[b] + edges[b+1]) / 2
        remain = torch.nonzero(~used, as_tuple=True)[0]

        for b in range(K):
            if selected[b] is not None:
                continue
            if remain.numel() == 0:
                # 후보가 다 소진되면 그냥 아무거나 중복 허용
                selected[b] = random.randrange(M)
                continue
            target = 0.5 * (edges[b] + edges[b + 1])
            # remain 중 target에 가장 가까운 후보 선택
            diffs = (mus[remain] - target).abs()
            best_idx = torch.argmin(diffs).item()
            t = remain[best_idx].item()
            selected[b] = t
            used[t] = True
            remain = torch.nonzero(~used, as_tuple=True)[0]

        # ===== 5) crop해서 반환 =====
        lq_patches, hq_patches = [], []
        for t in selected[:K]:
            i, j = cand_i[t], cand_j[t]
            lq_patches.append(lq[:, i:i + S, j:j + S])
            hq_patches.append(hq[:, i:i + S, j:j + S])

        if K == 1:
            return lq_patches[0], hq_patches[0]
        return torch.stack(lq_patches, dim=0), torch.stack(hq_patches, dim=0)

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
                 validation=False,
                 use_augmentation=True,
                 gamma_range=(0.7, 1.3),
                 noise_prob=0.3,
                 noise_level=(0.01, 0.03)):

        super(NTIRE2026_EfficientLLIETrainDataset, self).__init__()

        self.dataset_dir = os.path.join(data_dir, 'train')
        self.img_size = img_size
        self.use_augmentation = use_augmentation
        self.validation_index = 300
        self.validation = validation

        low_quality_folder = os.path.join(self.dataset_dir, 'low-20260203T115952Z-3-001', 'low')
        high_quality_folder = os.path.join(self.dataset_dir, 'normal-20260203T115952Z-3-001', 'normal')

        self.low_quality_folder_list = natsorted(glob(os.path.join(low_quality_folder, '*.jpg')))
        self.high_quality_folder_list = natsorted(glob(os.path.join(high_quality_folder, '*.jpg')))

        if validation:
            self.low_quality_folder_list = self.low_quality_folder_list[self.validation_index:]
            self.high_quality_folder_list = self.high_quality_folder_list[self.validation_index:]
        else:
            self.low_quality_folder_list = self.low_quality_folder_list[:self.validation_index]
            self.high_quality_folder_list = self.high_quality_folder_list[:self.validation_index]

        assert len(self.low_quality_folder_list) == len(self.high_quality_folder_list), \
            f"Mismatch: {len(self.low_quality_folder_list)} LQ vs {len(self.high_quality_folder_list)} HQ"

        # Augmentation 초기화
        if validation:
            self.to_tensor = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            if self.use_augmentation:
                self.augmentation = LowLightAugmentation(
                    img_size=img_size,
                    use_geometric=True,
                    use_photometric=False,
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

        # if validation:
        #     self.to_tensor = transforms.Compose([
        #         transforms.ToTensor()
        #     ])

    def __len__(self):
        return len(self.low_quality_folder_list)

    def __getitem__(self, index):
        low_quality_image_path = self.low_quality_folder_list[index]
        high_quality_image_path = self.high_quality_folder_list[index]

        low_quality_image = Image.open(low_quality_image_path).convert('RGB')
        high_quality_image = Image.open(high_quality_image_path).convert('RGB')

        # Augmentation 적용
        if self.validation:
            low_quality_tensor = self.to_tensor(low_quality_image)
            high_quality_tensor = self.to_tensor(high_quality_image)
        else:
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