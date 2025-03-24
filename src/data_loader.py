import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImage, Resize, EnsureChannelFirst, ScaleIntensity
import os

class LungNoduleDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = LoadImage()(self.image_paths[idx])  # Load image
        image = EnsureChannelFirst()(image)  # Convert shape to (C, H, W)
        image = ScaleIntensity()(image)  # Normalize pixel values

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label