import os
import ee
import numpy as np
import requests
import io
import base64
from rasterio.io import MemoryFile
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

ee.Initialize(project='deforestation-detection-459814')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model once
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=4,
    classes=1,
    activation=None,
).to(DEVICE)
model = torch.load("deforestation_unet_full_model.pt", map_location=DEVICE, weights_only=False)
model.eval()

def apply_scale_factors(image):
    optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
    return image.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)

def fetch_rgb_ndvi(region, year, scale=30):
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year, 12, 31)
    col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
           .filterBounds(region)
           .filterDate(start, end)
           .filterMetadata('CLOUD_COVER', 'less_than', 10)
           .map(apply_scale_factors))
    image = col.median().clip(region)
    ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    image = image.addBands(ndvi)
    return image.select(['SR_B4', 'SR_B3', 'SR_B2']), image.select('NDVI')

def download_geotiff_array(img, region, bands, scale=30):
    url = img.getThumbURL({
        'scale': scale,
        'region': region,
        'format': 'GeoTIFF',
        'bands': bands
    })
    response = requests.get(url)
    with MemoryFile(response.content) as memfile:
        with memfile.open() as src:
            arr = src.read().astype(np.float32)
            if arr.max() > 1.5:
                arr /= 255.0
    return arr

def predict_from_arrays(rgb_arr, ndvi_arr):
    rgb_arr = rgb_arr[:3, :, :]
    ndvi_arr = ndvi_arr[:1, :, :]
    input_arr = np.concatenate([rgb_arr, ndvi_arr], axis=0)
    input_tensor = torch.tensor(input_arr).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = torch.sigmoid(model(input_tensor))
        return (pred > 0.5).float().squeeze().cpu().numpy()

def get_deforestation_color_map(mask_t0, mask_t1):
    H, W = mask_t0.shape
    color_map = np.zeros((H, W, 3), dtype=np.uint8)

    retained = (mask_t0 == 1) & (mask_t1 == 1)
    lost     = (mask_t0 == 1) & (mask_t1 == 0)
    gained   = (mask_t0 == 0) & (mask_t1 == 1)
    none     = (mask_t0 == 0) & (mask_t1 == 0)

    color_map[retained] = [0, 255, 0]         # Green
    color_map[lost]     = [255, 0, 0]         # Red
    color_map[gained]   = [65, 168, 255]      # Blue (gain)
    color_map[none]     = [255, 255, 255]     # White (no change)

    return color_map

def run_deforestation_pipeline(lat_min, lat_max, lon_min, lon_max, start_year, end_year):
    region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

    rgb_t0_ee, ndvi_t0_ee = fetch_rgb_ndvi(region, start_year)
    rgb_t0 = download_geotiff_array(rgb_t0_ee, region, ['SR_B4', 'SR_B3', 'SR_B2'])
    ndvi_t0 = download_geotiff_array(ndvi_t0_ee, region, ['NDVI'])

    rgb_t1_ee, ndvi_t1_ee = fetch_rgb_ndvi(region, end_year)
    rgb_t1 = download_geotiff_array(rgb_t1_ee, region, ['SR_B4', 'SR_B3', 'SR_B2'])
    ndvi_t1 = download_geotiff_array(ndvi_t1_ee, region, ['NDVI'])

    mask_t0 = predict_from_arrays(rgb_t0, ndvi_t0)
    mask_t1 = predict_from_arrays(rgb_t1, ndvi_t1)

    deforested_pixels = ((mask_t0 == 1) & (mask_t1 == 0)).sum()
    gained_pixels = ((mask_t0 == 0) & (mask_t1 == 1)).sum()
    total_vegetation_t0 = (mask_t0 == 1).sum()

    percent_loss = (deforested_pixels / total_vegetation_t0) * 100 if total_vegetation_t0 > 0 else 0
    percent_gain = (gained_pixels / mask_t0.size) * 100  # relative to total area

    color_mask = get_deforestation_color_map(mask_t0, mask_t1)

    # Generate figure in memory
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(mask_t0, cmap="Greens")
    axes[0].set_title(f"Vegetation in {start_year}")
    axes[0].axis("off")

    axes[1].imshow(mask_t1, cmap="Greens")
    axes[1].set_title(f"Vegetation in {end_year}")
    axes[1].axis("off")

    axes[2].imshow(color_mask)
    axes[2].set_title(f"Vegetation Change")
    axes[2].axis("off")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return {
        "percent_deforested": round(percent_loss, 2),
        "percent_regrowth": round(percent_gain, 2),
        "image_base64": img_base64
    }

