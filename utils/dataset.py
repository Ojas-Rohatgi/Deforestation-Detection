import os
import glob
from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image

class DeforestationDataset(Dataset):
    def __init__(self, rgb_dir, ndvi_dir, mask_dir, transform=None):
        self.rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
        self.ndvi_paths = sorted(glob.glob(os.path.join(ndvi_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb = Image.open(self.rgb_paths[idx]).convert("RGB")
        ndvi = Image.open(self.ndvi_paths[idx]).convert("L")  # NDVI as grayscale

        rgb = np.array(rgb) / 255.0
        ndvi = np.array(ndvi)[..., None] / 255.0  # (H, W, 1)

        image = np.concatenate([rgb, ndvi], axis=-1)  # (H, W, 4)
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L")) / 255.0
        mask = mask[None, ...]  # Shape: (1, H, W)

        image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)  # (4, H, W)
        mask = torch.tensor(mask, dtype=torch.float)

        return image, mask
