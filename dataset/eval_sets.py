from torch.utils.data import Dataset

class DatasetFromFolderEval(Dataset):
    def __init__(self, data_dir, transform=None):
        super(DatasetFromFolderEval, self).__init__()