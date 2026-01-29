"""
dataset.py
Author: Lennin Abrão Sousa Santos
Data criação: 2026-01-29
Date update: 2026-01-29
Explicação: PyTorch Dataset class for loading the shapes images.
How to use: Used internally by train.py
Licença: AGPL3
"""

from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torchvision.transforms as T


class ShapesDataset(Dataset):
    def __init__(self, root_dir=None, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if root_dir is None:
            # Assume ../data/raw relative to this file
            root_dir = Path(__file__).parent.parent / "data" / "raw"
        else:
            root_dir = Path(root_dir)

        self.root_dir = root_dir

        if transform is None:
            # Default transform if not provided: ToTensor and Normalize for grayscale
            self.transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
        else:
            self.transform = transform

        self.image_paths = []
        self.labels = []

        # Load circle images (label 0)
        circle_dir = root_dir / "circle"
        if circle_dir.exists():
            for img_path in circle_dir.glob("*.png"):
                self.image_paths.append(img_path)
                self.labels.append(0)

        # Load square images (label 1)
        square_dir = root_dir / "square"
        if square_dir.exists():
            for img_path in square_dir.glob("*.png"):
                self.image_paths.append(img_path)
                self.labels.append(1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Open image and convert to grayscale ('L') just in case
        image = Image.open(img_path).convert("L")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
