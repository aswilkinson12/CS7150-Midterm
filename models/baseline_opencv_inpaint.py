import os
import numpy as np
import rasterio as rio
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ========= CONFIG =========
IN_DIR = "../data/raw"
OUT_DIR = "../data/out"
PNG_DIR = os.path.join(OUT_DIR, "opencv_inpaint")

METHODS = ("telea", "ns")  # Inpainting algorithms
RADII = (3, 5)  # Inpainting neighborhood sizes
ERODE_PX = 1  # Shrink mask by 1 pixel to reduce artifacts

# Lake Erie crop (1250 wide x 750 tall, from bottom-left corner)
CROP_LAKE_ERIE = True
CROP_WIDTH = 1250
CROP_HEIGHT = 750
# ==========================

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)

# CyAN colormap: blue -> green -> yellow -> red
CYAN_CMAP = LinearSegmentedColormap.from_list(
    "cyan_like",
    [(0.00, (26 / 255, 0 / 255, 102 / 255)),  # deep blue
     (0.25, (0 / 255, 128 / 255, 255 / 255)),  # blue
     (0.50, (0 / 255, 255 / 255, 102 / 255)),  # green
     (0.75, (255 / 255, 255 / 255, 0 / 255)),  # yellow
     (1.00, (255 / 255, 51 / 255, 0 / 255))]  # red
)


def colorize_dn(dn: np.ndarray) -> np.ndarray:
    """
    Convert DN values to RGB image matching CyAN legend:
    - 255 (cloud/no-data) -> black
    - 254 (land) -> light brown
    - 0 (below detection) -> gray
    - 1-253 (cyanobacteria) -> blue to red gradient
    """
    dn = dn.astype(np.uint16)
    H, W = dn.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # Create masks for each category
    mask_cloud = (dn == 255)
    mask_land = (dn == 254)
    mask_dn0 = (dn == 0)
    mask_val = (dn >= 1) & (dn <= 253)

    # Assign colors
    rgb[mask_cloud] = (0, 0, 0)  # black
    rgb[mask_land] = (205, 170, 125)  # tan/brown
    rgb[mask_dn0] = (160, 160, 160)  # gray

    # Apply colormap to valid data
    if np.any(mask_val):
        normalized = (dn[mask_val] - 1) / 252.0  # 0 to 1 range
        colors = (CYAN_CMAP(normalized)[:, :3] * 255.0).astype(np.uint8)
        rgb[mask_val] = colors

    return rgb


def save_png(img: np.ndarray, path: str, title: str = None):
    """Save array as PNG image"""
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    if title:
        plt.title(title, fontsize=10)
    plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close()


def inpaint_and_save(tif_path: str):
    """Main processing function"""
    base = os.path.splitext(os.path.basename(tif_path))[0]

    # Read the GeoTIFF; raster is used to read  tif files
    with rio.open(tif_path) as ds:
        dn_full = ds.read(1)  # Read band 1
        profile = ds.profile  # Metadata
        img_height = dn_full.shape[0]

    # Crop to Lake Erie region
    if CROP_LAKE_ERIE:
        y_start = img_height - CROP_HEIGHT  # Start from bottom
        y_end = img_height  # To bottom edge
        x_start = 0
        x_end = CROP_WIDTH

        dn = dn_full[y_start:y_end, x_start:x_end].copy()

        # Update metadata for new dimensions
        profile.update({
            'height': dn.shape[0],
            'width': dn.shape[1]
        })
    else:
        dn = dn_full.copy()

    # Identify pixel categories
    land = (dn == 254)
    cloud = (dn == 255)

    # clouds
    mask_to_inpaint = cloud.copy()

    # Erode mask slightly to avoid edge artifacts
    if ERODE_PX > 0 and np.any(mask_to_inpaint):
        kernel = np.ones((ERODE_PX * 2 + 1, ERODE_PX * 2 + 1), np.uint8)
        mask_eroded = cv2.erode(mask_to_inpaint.astype(np.uint8),
                                kernel, iterations=1).astype(bool)
    else:
        mask_eroded = mask_to_inpaint

    # Skip if no clouds to fix
    if not np.any(mask_eroded):
        print(f" No clouds found ")
        return

    cloud_percent = 100 * np.sum(mask_eroded) / mask_eroded.size
    print(f"  Cloud coverage: {cloud_percent:.1f}% ({np.sum(mask_eroded)} pixels)")

    dn_original = dn.copy()

    # Prepare image for inpainting
    # Set land/cloud to median water value for stable neighborhoods
    img = dn.copy().astype(np.float32)
    water_pixels = dn[(dn < 254)]
    median_water = float(np.median(water_pixels)) if water_pixels.size else 0.0
    img[land | cloud] = median_water

    # Save visualization of original and mask
    save_png(colorize_dn(dn),
             os.path.join(PNG_DIR, f"{base}_01_original.png"),
             "Original")
    save_png(mask_eroded.astype(np.uint8) * 255,
             os.path.join(PNG_DIR, f"{base}_02_mask.png"),
             f"Cloud Mask ({cloud_percent:.1f}%)")

    # Try each combination of method and radius
    for method in METHODS:
        # Select OpenCV algorithm
        flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS

        for radius in RADII:
            # Convert mask to uint8 format required by OpenCV
            mask_u8 = (mask_eroded.astype(np.uint8) * 255)

            # INPAINT!
            repaired = cv2.inpaint(img.astype(np.uint8), mask_u8, radius, flag)

            # Build output: keep original except where we inpainted
            out = dn_original.copy()
            out[mask_eroded] = repaired[mask_eroded]

            # Ensure valid range
            out = np.clip(out, 0, 255).astype(np.uint8)

            # Create binary mask showing what changed
            changed = mask_eroded.astype(np.uint8)

            # Generate filenames
            tag = f"{method}_r{radius}"
            tif_out = os.path.join(OUT_DIR, f"{base}_fixed_{tag}.tif")
            tif_chg = os.path.join(OUT_DIR, f"{base}_changed_{tag}.tif")

            # Save GeoTIFFs
            with rio.open(tif_out, "w", **profile) as ds:
                ds.write(out, 1)

            with rio.open(tif_chg, "w", **profile) as ds:
                ds.write(changed, 1)

            # Save PNGs
            save_png(colorize_dn(out),
                     os.path.join(PNG_DIR, f"{base}_fixed_{tag}.png"),
                     f"Inpainted: {tag}")
            save_png(changed * 255,
                     os.path.join(PNG_DIR, f"{base}_mask_{tag}.png"),
                     f"Mask: {tag}")

            print(f"    ✓ {tag}")


def main():
    tifs = [f for f in sorted(os.listdir(IN_DIR))
            if f.lower().endswith(".tif")
            and "_fixed_" not in f
            and "_changed_" not in f]

    if not tifs:
        print(f"No .tif files found in {IN_DIR}")
        return

    print("=" * 70)

    for fn in tifs:
        path = os.path.join(IN_DIR, fn)
        print(f"Processing: {fn}")
        try:
            inpaint_and_save(path)
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
        print()


if __name__ == "__main__":
    main()