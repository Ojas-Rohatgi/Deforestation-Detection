import os
import numpy as np
from skimage.io import imread, imsave
from glob import glob
from tqdm import tqdm
import shutil

def generate_deforestation_mask(before_path, after_path, masks_dir, aoi_name):
    """
    Generates and saves the deforestation mask for a given AOI pair.
    """
    # print(f"Generating mask for AOI: {aoi_name}")
    try:
        before_img = imread(before_path)
        after_img = imread(after_path)

        # Calculate NDVI (using bands 3 and 2 for NIR and Red as per Landsat 8)
        # Ensure images have enough bands
        if before_img.shape[-1] < 4 or after_img.shape[-1] < 4:
             print(f"Skipping {aoi_name}: Image does not have enough bands for NDVI calculation.")
             return False

        before_ndvi = (before_img[:,:,3] - before_img[:,:,2]) / (before_img[:,:,3] + before_img[:,:,2] + 1e-8)
        after_ndvi = (after_img[:,:,3] - after_img[:,:,2]) / (after_img[:,:,3] + after_img[:,:,2] + 1e-8)

        ndvi_diff = after_ndvi - before_ndvi

        # Create deforestation mask (values < -0.1 indicate potential deforestation)
        deforestation_mask = (ndvi_diff < -0.015).astype(np.uint8) * 255

        # Save the mask
        mask_filename = f"{aoi_name}_mask.png" # Save as PNG for simplicity
        mask_path = os.path.join(masks_dir, mask_filename)
        imsave(mask_path, deforestation_mask)

        # print(f"Mask saved to: {mask_path}")
        return True

    except Exception as e:
        print(f"Error generating mask for AOI {aoi_name}: {e}")
        return False

def main():
    output_dir = "output" # Directory containing the original TIFF images
    masks_dir = os.path.join("data", "masks") # Directory to save the generated masks

    # Clear existing masks
    if os.path.exists(masks_dir):
        shutil.rmtree(masks_dir)
    os.makedirs(masks_dir, exist_ok=True)

    # Get all before image files
    before_images = glob(os.path.join(output_dir, "before_2015_amazon_*.tif"))

    if not before_images:
        print(f"No 'before_2015_amazon_*.tif' files found in '{output_dir}'.")
        return

    processed_count = 0
    print("Starting mask generation:")

    # Add progress bar for AOI processing
    with tqdm(before_images, desc="Generating Masks") as pbar:
        for before_path in pbar:
            # Get corresponding after image path
            aoi_name = os.path.basename(before_path).replace("before_2015_", "").replace(".tif", "")
            after_path = os.path.join(output_dir, f"after_2020_{aoi_name}.tif")

            # Check if the corresponding after file exists
            if os.path.exists(after_path):
                pbar.set_description(f"Generating Mask for AOI: {aoi_name}")
                if generate_deforestation_mask(before_path, after_path, masks_dir, aoi_name):
                    processed_count += 1
                pbar.set_postfix(generated_masks=processed_count)
            else:
                print(f"Skipping {aoi_name}: Corresponding after file '{os.path.basename(after_path)}' not found.")


    print(f"\nFinished mask generation. Total masks generated: {processed_count}")
    print(f"Masks saved to: {masks_dir}")

if __name__ == "__main__":
    main()