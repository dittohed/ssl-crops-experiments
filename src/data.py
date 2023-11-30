from pathlib import Path
from typing import Callable

import PIL

from torch.utils.data import Dataset


class FlatImageFolder(Dataset):
    """
    Custom PyTorch dataset for loading images from a single directory 
    (without class-wise subdirectories).
    """

    def __init__(self, data_dir: str, transform: Callable = None):
        super().__init__()

        self.paths = list(Path(data_dir).glob('*'))
        self.transform = transform

    def __getitem__(self, index: int):
        img = PIL.Image.open(self.paths[index]).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img 
    
    def __len__(self):
        return len(self.paths)