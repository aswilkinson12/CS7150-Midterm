"""
evaluate.py - Full-image inpainting with REAL temporal neighbors
Fixed to use actual past frames instead of repeating the same frame
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import rasterio as rio
from tqdm import tqdm
import pandas as pd
from datetime import timedelta
import warnings
from rasterio.errors import NotGeoreferencedWarning

import config as cfg
from models.convlstm import HABInpaintModel

warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

# CyAN colormap
CYAN_CMAP = LinearSegmentedColormap.from_list(
    "cyan",
    [(0.00, (26 / 255, 0 / 255, 102 / 255)),
     (0.25, (0 / 255, 128 / 255, 255 / 255)),
     (0.50, (0 / 255, 255 / 255, 102 / 255)),
     (0.75, (255 / 255, 255 / 255, 0 / 255)),
     (1.00, (255 / 255, 51 / 255, 0 / 255))]
)


def colorize_dn(dn):
    """Convert DN to RGB using CyAN colormap."""
    dn = dn.astype(np.uint16)
    H, W = dn.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    cloud = (dn == 255)
    land = (dn == 254)
    dn0 = (dn == 0)
    valid = (dn >= 1) & (dn <= 253)

    rgb[cloud] = (0, 0, 0)
    rgb[land] = (205, 170, 125)
    rgb[dn0] = (160, 160, 160)

    if np.any(valid):
        norm = (dn[valid] - 1) / 252.0
        rgb[valid] = (CYAN_CMAP(norm)[:, :3] * 255).astype(np.uint8)

    return rgb


def read_image(path):
    """Read GeoTIFF."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
        with rio.open(path) as src:
            dn = src.read(1).astype(np.float32)
    return dn


def normalize_patch(patch, water_mask):
    """P2-P98 normalization on water pixels."""
    water_vals = patch[water_mask]
    if len(water_vals) < 10:
        return np.zeros_like(patch, dtype=np.float32), 0, 1

    p2, p98 = np.percentile(water_vals, [2, 98])
    if p98 - p2 < 1e-6:
        p2, p98 = 0, 253

    normalized = np.clip((patch - p2) / (p98 - p2), 0, 1)
    return normalized.astype(np.float32), p2, p98


def denormalize_patch(normalized, p2, p98):
    """Inverse normalization."""
    dn = normalized * (p98 - p2) + p2
    return np.clip(dn, 0, 253).astype(np.uint8)


def find_temporal_neighbors(target_date, index_df, dates_dict):
    """
    Find 5 past neighbors (t-5 to t-1) for target date.
    Returns: (neighbor_paths, present_bits)
    """
    neighbors = []
    present_bits = []

    for offset in range(-cfg.LOOKBACK, 0):  # -5, -4, -3, -2, -1
        neighbor_date = target_date + timedelta(days=offset)

        # Check if exact date exists
        if neighbor_date in dates_dict:
            neighbors.append(dates_dict[neighbor_date])
            present_bits.append(1.0)
        else:
            # Find closest available date within MAX_GAP_DAYS
            closest_date = None
            min_gap = float('inf')

            for available_date in dates_dict.keys():
                gap = abs((available_date - neighbor_date).days)
                if gap < min_gap and gap <= cfg.MAX_GAP_DAYS:
                    min_gap = gap
                    closest_date = available_date

            if closest_date:
                neighbors.append(dates_dict[closest_date])
                present_bits.append(1.0)
            else:
                # No suitable neighbor found
                neighbors.append(None)
                present_bits.append(0.0)

    return neighbors, present_bits


def load_temporal_sequence(target_path, index_df, dates_dict, target_date):
    """
    Load target image and its 5 temporal neighbors.
    Returns: (target_dn, neighbor_frames, present_bits)
    """
    # Load target
    target_dn = read_image(os.path.join(cfg.DATA_DIR, target_path))
    H, W = target_dn.shape

    # Find neighbors
    neighbor_paths, present_bits = find_temporal_neighbors(target_date, index_df, dates_dict)

    # Load neighbor frames
    neighbor_frames = []
    for neighbor_path, present in zip(neighbor_paths, present_bits):
        if present and neighbor_path:
            try:
                neighbor_dn = read_image(os.path.join(cfg.DATA_DIR, neighbor_path))
                # Ensure same size
                if neighbor_dn.shape != (H, W):
                    neighbor_dn = np.zeros((H, W), dtype=np.float32)
                neighbor_frames.append(neighbor_dn)
            except Exception as e:
                print(f"  Warning: Failed to load neighbor {neighbor_path}: {e}")
                neighbor_frames.append(np.zeros((H, W), dtype=np.float32))
        else:
            # Missing neighbor - use zeros
            neighbor_frames.append(np.zeros((H, W), dtype=np.float32))

    return target_dn, neighbor_frames, present_bits


def inpaint_full_image_with_temporal(model, target_dn, neighbor_frames, present_bits,
                                      device, patch_size=128, stride=128):
    """
    Inpaint full image using sliding window with REAL temporal neighbors.
    """
    H, W = target_dn.shape
    output_dn = target_dn.copy()

    inpaint_count = np.zeros((H, W), dtype=np.float32)
    inpaint_sum = np.zeros((H, W), dtype=np.float32)

    cloud_mask_full = (target_dn == 255)
    water_mask_full = (target_dn < 254)

    patches_processed = 0

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            # Extract patches
            patch_dn = target_dn[y:y + patch_size, x:x + patch_size]
            water_mask_patch = water_mask_full[y:y + patch_size, x:x + patch_size]
            cloud_mask_patch = cloud_mask_full[y:y + patch_size, x:x + patch_size]

            # Skip if no clouds in patch
            if not cloud_mask_patch.any():
                continue

            # Skip if not enough water
            if water_mask_patch.sum() < 100:
                continue

            # Normalize target patch
            patch_norm, p2, p98 = normalize_patch(patch_dn, water_mask_patch)

            # Extract and normalize neighbor patches
            neighbor_patches = []
            for neighbor_frame in neighbor_frames:
                neighbor_patch = neighbor_frame[y:y + patch_size, x:x + patch_size]
                neighbor_norm, _, _ = normalize_patch(neighbor_patch, water_mask_patch)
                neighbor_patches.append(neighbor_norm)

            # Build input [1, T=5, C=7, H, W]
            x_input = np.zeros((1, 5, 7, patch_size, patch_size), dtype=np.float32)

            for t in range(5):
                # Frame from temporal neighbor (NOT target frame!)
                x_input[0, t, 0] = neighbor_patches[t]

                # Mask (same for all time steps - shows where target has clouds)
                x_input[0, t, 1] = cloud_mask_patch.astype(np.float32)

                # Present bits (which neighbors are available)
                for pb_idx, pb_val in enumerate(present_bits):
                    x_input[0, t, 2 + pb_idx] = pb_val

            x_tensor = torch.from_numpy(x_input).to(device)

            # Predict
            with torch.no_grad():
                pred_norm = model(x_tensor).cpu().numpy()[0, 0]

            # Denormalize
            pred_dn = denormalize_patch(pred_norm, p2, p98)

            # Accumulate only in cloud regions
            inpaint_sum[y:y + patch_size, x:x + patch_size][cloud_mask_patch] += pred_dn[cloud_mask_patch]
            inpaint_count[y:y + patch_size, x:x + patch_size][cloud_mask_patch] += 1

            patches_processed += 1

    # Average overlapping predictions
    mask_inpainted = (inpaint_count > 0)
    output_dn[mask_inpainted] = (inpaint_sum[mask_inpainted] / inpaint_count[mask_inpainted]).astype(np.uint8)

    return output_dn, patches_processed


def evaluate_full_images():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    # Load model
    model = HABInpaintModel(
        input_channels=cfg.INPUT_CHANNELS,
        hidden_dims=cfg.HIDDEN_DIMS,
        kernel_size=cfg.KERNEL_SIZE
    ).to(cfg.DEVICE)

    checkpoint = torch.load(f"{cfg.CHECKPOINT_DIR}/model_best.pt", map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
    print(f"  Best val L1: {checkpoint['metrics'].get('l1', 'N/A')}")

    # Load index and build date dictionary
    index = pd.read_csv(cfg.INDEX_CSV)
    index['date'] = pd.to_datetime(index['date'])

    # Build date->path lookup
    dates_dict = dict(zip(index['date'], index['path']))

    # Get test files
    test_df = index[index['split'] == 'test'].reset_index(drop=True)

    print(f"\nProcessing {len(test_df)} test images...")
    print(f"Using temporal window: last {cfg.LOOKBACK} days (within {cfg.MAX_GAP_DAYS} day gaps)\n")

    saved = 0
    max_save = 10
    skipped = 0

    for idx, row in tqdm(test_df.iterrows(), total=min(len(test_df), max_save),
                         desc="Full-image inpainting"):
        if saved >= max_save:
            break

        target_path = row['path']
        target_date = row['date']
        filepath = os.path.join(cfg.DATA_DIR, target_path)

        if not os.path.exists(filepath):
            skipped += 1
            continue

        # Load temporal sequence
        try:
            target_dn, neighbor_frames, present_bits = load_temporal_sequence(
                target_path, index, dates_dict, target_date
            )
        except Exception as e:
            print(f"\n  Error loading {target_path}: {e}")
            skipped += 1
            continue

        # Check clouds
        cloud_pixels = (target_dn == 255).sum()
        if cloud_pixels == 0:
            skipped += 1
            continue

        # Check temporal availability
        n_available = sum(present_bits)
        if n_available < 3:
            print(f"\n  Skipping {target_path}: only {n_available}/5 neighbors available")
            skipped += 1
            continue

        print(f"\n{target_path} ({target_date.strftime('%Y-%m-%d')}):")
        print(f"  Clouds: {cloud_pixels} pixels ({100 * cloud_pixels / target_dn.size:.2f}%)")
        print(f"  Temporal neighbors: {n_available}/5 available")

        # Inpaint with real temporal context
        inpainted_dn, n_patches = inpaint_full_image_with_temporal(
            model, target_dn, neighbor_frames, present_bits, cfg.DEVICE
        )
        print(f"  Processed {n_patches} patches")

        # Colorize
        output_rgb = colorize_dn(inpainted_dn)
        original_rgb = colorize_dn(target_dn)

        # Save comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].imshow(original_rgb)
        axes[0].set_title(f"Original ({target_date.strftime('%Y-%m-%d')})\n{cloud_pixels} cloud pixels",
                         fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(output_rgb)
        axes[1].set_title(f"ConvLSTM Inpainted\n{n_available}/5 temporal neighbors", fontsize=12)
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(cfg.RESULTS_DIR, f"comparison_{saved:03d}.png"), dpi=150)
        plt.close()

        # Save individual inpainted image
        plt.figure(figsize=(10, 6))
        plt.imshow(output_rgb)
        plt.title(f"Lake Okeechobee - Inpainted ({target_date.strftime('%Y-%m-%d')})",
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.RESULTS_DIR, f"inpainted_{saved:03d}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # Save DN array
        np.save(os.path.join(cfg.RESULTS_DIR, f"inpainted_{saved:03d}.npy"), inpainted_dn)

        saved += 1

    print(f"\n{'='*60}")
    print(f"✓ Saved {saved} results to {cfg.RESULTS_DIR}/")
    print(f"  Skipped {skipped} images (no clouds or insufficient temporal data)")
    print(f"{'='*60}")


if __name__ == "__main__":
    evaluate_full_images()