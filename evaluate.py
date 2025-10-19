"""
evaluate.py - Full-image inpainting evaluation (updated for 5 past frames)
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import rasterio as rio
from tqdm import tqdm
import pandas as pd

import config as cfg
from models.hab_inpaint import HABInpaintModel

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


def normalize_patch(patch, water_mask):
    """P2-P98 normalization."""
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


def inpaint_full_image(model, image_dn, device, patch_size=128, stride=128):
    """
    Inpaint full image using sliding window.

    Note: This version doesn't use real temporal neighbors (they're not available
    during evaluation on single images). It fills all time steps with the same frame.
    For proper temporal inpainting, use the temporal evaluation script.
    """
    H, W = image_dn.shape
    output_dn = image_dn.copy()

    inpaint_count = np.zeros((H, W), dtype=np.float32)
    inpaint_sum = np.zeros((H, W), dtype=np.float32)

    cloud_mask_full = (image_dn == 255)
    water_mask_full = (image_dn < 254)

    patches_processed = 0

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch_dn = image_dn[y:y + patch_size, x:x + patch_size]
            water_mask_patch = water_mask_full[y:y + patch_size, x:x + patch_size]
            cloud_mask_patch = cloud_mask_full[y:y + patch_size, x:x + patch_size]

            if not cloud_mask_patch.any():
                continue

            if water_mask_patch.sum() < 100:
                continue

            # Normalize
            patch_norm, p2, p98 = normalize_patch(patch_dn, water_mask_patch)

            # Create input [1, T=5, C=7, H, W]
            x_input = np.zeros((1, 5, 7, patch_size, patch_size), dtype=np.float32)

            # Fill all 5 past time steps with current frame (no real temporal data)
            for t in range(5):
                x_input[0, t, 0] = patch_norm  # Frame
                x_input[0, t, 1] = cloud_mask_patch.astype(np.float32)  # Mask
                # Present bits - all 5 marked as available
                for pb in range(5):
                    x_input[0, t, 2 + pb] = 1.0

            x_tensor = torch.from_numpy(x_input).to(device)

            # Predict
            with torch.no_grad():
                pred_norm = model(x_tensor).cpu().numpy()[0, 0]

            # Denormalize
            pred_dn = denormalize_patch(pred_norm, p2, p98)

            # Accumulate
            inpaint_sum[y:y + patch_size, x:x + patch_size][cloud_mask_patch] += pred_dn[cloud_mask_patch]
            inpaint_count[y:y + patch_size, x:x + patch_size][cloud_mask_patch] += 1

            patches_processed += 1

    # Average
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

    # Load test files
    index = pd.read_csv(cfg.INDEX_CSV)
    test_files = index[index['split'] == 'test']['path'].tolist()

    print(f"\nProcessing {len(test_files)} test images...")

    saved = 0
    max_save = 10

    for file_idx, filename in enumerate(tqdm(test_files[:max_save], desc="Full-image inpainting")):
        filepath = os.path.join(cfg.DATA_DIR, filename)

        if not os.path.exists(filepath):
            continue

        # Load full image
        with rio.open(filepath) as src:
            image_dn = src.read(1).astype(np.uint8)

        # Check clouds
        cloud_pixels = (image_dn == 255).sum()
        if cloud_pixels == 0:
            continue

        print(f"\n{filename}:")
        print(f"  Clouds: {cloud_pixels} pixels ({100 * cloud_pixels / image_dn.size:.2f}%)")

        # Inpaint
        inpainted_dn, n_patches = inpaint_full_image(model, image_dn, cfg.DEVICE)
        print(f"  Processed {n_patches} patches")

        # Colorize
        output_rgb = colorize_dn(inpainted_dn)
        original_rgb = colorize_dn(image_dn)

        # Save comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].imshow(original_rgb)
        axes[0].set_title(f"Original\n{cloud_pixels} cloud pixels", fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(output_rgb)
        axes[1].set_title("ConvLSTM Inpainted", fontsize=12)
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(cfg.RESULTS_DIR, f"comparison_{saved:03d}.png"), dpi=150)
        plt.close()

        # Save individual
        plt.figure(figsize=(10, 6))
        plt.imshow(output_rgb)
        plt.title("Lake Erie - Inpainted", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.RESULTS_DIR, f"inpainted_{saved:03d}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # Save DN
        np.save(os.path.join(cfg.RESULTS_DIR, f"inpainted_{saved:03d}.npy"), inpainted_dn)

        saved += 1

    print(f"\n✓ Saved {saved} results to {cfg.RESULTS_DIR}/")


if __name__ == "__main__":
    evaluate_full_images()