from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset.LOLDataset import LOLDatasetFromFolderTrain, LOLDatasetFromFolderTest
from dataset.NTIRE2026_EfficientLLIEDataset import NTIRE2026_EfficientLLIETrainDataset, NTIRE2026_EfficientLLIEValDataset
DATASET_ZOO = {
    'lol_dataset': {'train_dataset': LOLDatasetFromFolderTrain,
                    'test_dataset' : LOLDatasetFromFolderTest},
    'NTIRE2026_Challenge_Dataset': {'train_dataset': NTIRE2026_EfficientLLIETrainDataset,
                                    'test_dataset': NTIRE2026_EfficientLLIEValDataset}
}

def train_transform(size=256):
    return transforms.Compose([
        transforms.RandomCrop((size, size)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor()])

def test_transform():
    return transforms.Compose([
        transforms.ToTensor()])

def dataloader_generator(args):
    print(f'===> Loading data_type: {args.data_type}')

    if args.data_type not in DATASET_ZOO:
        raise ValueError(f"Unknown data_type: {args.data_type}. Available: {list(DATASET_ZOO.keys())}")

    train_dataset = DATASET_ZOO[args.data_type]['train_dataset'](args.data_path, train_transform())
    test_dataset = DATASET_ZOO[args.data_type]['test_dataset'](args.data_path, test_transform())

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return train_loader, test_loader