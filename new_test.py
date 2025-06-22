import segmentation_models_pytorch as smp
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model structure
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=4,
    classes=1,
    activation=None,
).to(DEVICE)

# Load trained full model
model = torch.load("deforestation_unet_full_model.pt", map_location=DEVICE, weights_only=False)
model.eval()

def predict_single(rgb_path, ndvi_path):
    rgb = Image.open(rgb_path).convert("RGB").resize((256, 256))
    ndvi = Image.open(ndvi_path).convert("L").resize((256, 256))

    rgb = np.array(rgb) / 255.0
    ndvi = np.expand_dims(np.array(ndvi) / 255.0, axis=-1)

    image = np.concatenate([rgb, ndvi], axis=-1)  # (H, W, 4)
    image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = torch.sigmoid(model(image))
        pred_mask = (pred > 0.5).float()

    return pred_mask.squeeze().cpu().numpy()


def get_deforestation_color_map(mask_t0, mask_t1):
    H, W = mask_t0.shape
    color_map = np.zeros((H, W, 3), dtype=np.uint8)

    retained = (mask_t0 == 1) & (mask_t1 == 1)
    lost     = (mask_t0 == 1) & (mask_t1 == 0)
    none     = (mask_t0 == 0)  # non-vegetated in T0

    color_map[retained] = [0, 255, 0]    # Green
    color_map[lost]     = [255, 0, 0]    # Red
    color_map[none]     = [128, 128, 128]  # Gray

    return color_map


# ---- Input paths ----
base_name = "amazon_-8.56_-50.00"
rgb_t0 = f"data/processed_patches/train/images/rgb/{base_name}_0_rgb.png"
ndvi_t0 = f"data/processed_patches/train/images/ndvi/{base_name}_0_ndvi.png"
rgb_t1 = f"data/processed_patches/train/images/rgb/{base_name}_1_rgb.png"
ndvi_t1 = f"data/processed_patches/train/images/ndvi/{base_name}_1_ndvi.png"

# Predict vegetation masks
mask_t0 = predict_single(rgb_t0, ndvi_t0)
mask_t1 = predict_single(rgb_t1, ndvi_t1)

# Calculate deforestation: vegetation present in T0 but gone in T1
deforestation_mask = ((mask_t0 == 1) & (mask_t1 == 0)).astype(np.uint8) * 255

# Compute % deforestation
deforested_pixels = ((mask_t0 == 1) & (mask_t1 == 0)).sum()
total_vegetation_t0 = (mask_t0 == 1).sum()

if total_vegetation_t0 > 0:
    percent_deforested = (deforested_pixels / total_vegetation_t0) * 100
else:
    percent_deforested = 0

print(f"🌍 Deforestation in {base_name}: {percent_deforested:.2f}% of vegetation lost.")


# ---- Visualization ----
color_mask = get_deforestation_color_map(mask_t0, mask_t1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(mask_t0, cmap="Greens")
plt.title("Vegetation at T0")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask_t1, cmap="Greens")
plt.title("Vegetation at T1")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(color_mask)
plt.title(f"Change Map\n{percent_deforested:.2f}% loss")
plt.axis("off")

plt.tight_layout()
plt.savefig(f"{base_name}_deforestation_colormap.png", dpi=150)
plt.show()
