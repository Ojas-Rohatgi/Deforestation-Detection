import os
from utils.config import *
from utils.resize_transform import resize_transform
from torch.utils.data import DataLoader
from utils.dataset import DeforestationDataset

def get_loaders():
    def make_loader(split):
        return DataLoader(
            DeforestationDataset(
                rgb_dir=os.path.join(DATA_DIR, split, "images", "rgb"),
                ndvi_dir=os.path.join(DATA_DIR, split, "images", "ndvi"),
                mask_dir=os.path.join(DATA_DIR, split, "masks"),
                transform=resize_transform
            ),
            batch_size=BATCH_SIZE,
            shuffle=(split == "train")
        )

    return make_loader("train"), make_loader("val"), make_loader("test")
