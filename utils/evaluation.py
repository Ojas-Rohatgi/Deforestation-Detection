import torch
from utils.model import model
from utils.config import DEVICE

def evaluate(loader, return_score=False):
    model.eval()
    total_iou = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            preds = torch.sigmoid(model(images)) > 0.5  # Boolean
            masks = masks > 0.5  # Boolean

            intersection = (preds & masks).sum(dim=(1, 2, 3)).float()
            union = (preds | masks).sum(dim=(1, 2, 3)).float()
            iou = (intersection / (union + 1e-6)).mean()
            total_iou += iou.item()

    mean_iou = total_iou / len(loader)
    if return_score:
        return mean_iou
    else:
        print(f"Validation IoU: {mean_iou:.4f}")
        return None

