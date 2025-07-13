from torch.utils.data import Dataset
from pathlib import Path


class WatermelonDataset(Dataset):
    def __init__(self, root: Path, split: str = "train"):
        self.root = Path(root)
        self.split = split

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise NotImplementedError
