import torch

DATA_DIR = "data/processed_patches"
ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 500
LR = 1e-4