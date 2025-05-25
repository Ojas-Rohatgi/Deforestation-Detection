import segmentation_models_pytorch as smp
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load fresh model structure
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=4,
    classes=1,
    activation=None,
).to(DEVICE)

# Load trained weights
model = torch.load("deforestation_unet_full_model.pt", map_location=DEVICE, weights_only=False)
model.eval()

def predict_single(rgb_path, ndvi_path):
    # Load images
    rgb = Image.open(rgb_path).convert("RGB").resize((256, 256))
    ndvi = Image.open(ndvi_path).convert("L").resize((256, 256))

    # Preprocess
    rgb = np.array(rgb) / 255.0
    ndvi = np.array(ndvi)[..., None] / 255.0

    image = np.concatenate([rgb, ndvi], axis=-1)  # (H, W, 4)
    image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)  # (1, 4, H, W)

    # Move to device
    image = image.to(DEVICE)

    # Predict
    with torch.no_grad():
        pred = torch.sigmoid(model(image))  # (1, 1, H, W)
        pred_mask = (pred > 0.5).float()

    return pred_mask.squeeze().cpu().numpy()


pred_mask = predict_single("data/processed_patches/test/images/rgb/amazon_-8.56_-50.00_0_rgb.png", "data/processed_patches/test/images/ndvi/amazon_-8.56_-50.00_0_ndvi.png")
ground_truth = "data/processed_patches/test/masks/amazon_-8.56_-50.00_0_mask.png"

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(Image.open(ground_truth).convert("L"), cmap="gray")
plt.title("Ground Truth Deforested Area")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(pred_mask, cmap="gray")
plt.title("Predicted Deforested Area")
plt.axis("off")
plt.show()
plt.savefig("Ground Truth vs Predicted Area.png")
