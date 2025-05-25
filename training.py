import segmentation_models_pytorch as smp
from utils.config import *
from utils.loader import get_loaders
from utils.model import model
from utils.train import train

# Loaders
train_loader, val_loader, test_loader = get_loaders()

# Loss & Optimizer
loss_fn = smp.losses.DiceLoss(mode='binary')
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Run
if __name__ == "__main__":
    # Sanity check
    sample_image, sample_mask = next(iter(train_loader))
    print(f"Sample image shape: {sample_image.shape}")
    print(f"Sample mask shape: {sample_mask.shape}")


    print("Starting training...")
    # train(train_loader, val_loader, loss_fn, optimizer)

