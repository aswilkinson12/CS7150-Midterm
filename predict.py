"""
predict.py - Predict bloom for a specific date
Usage: python predict.py --date 2023-01-15
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from datetime import datetime, timedelta
import warnings
from rasterio.errors import NotGeoreferencedWarning

import config as cfg
from models.convlstm import HABInpaintModel
from utils.data_loader import read_image, normalize_robust

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
    """Convert DN to RGB."""
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


def denormalize(normalized, p2, p98):
    """Inverse normalization."""
    dn = normalized * (p98 - p2) + p2
    return np.clip(dn, 0, 253).astype(np.uint8)


def predict_date(model, target_date, index_df, device):
    """Predict bloom for target_date using past 5 days."""

    # Build date->path lookup
    dates_dict = dict(zip(index_df['date'], index_df['path']))

    # Load past 5 frames
    past_frames = []
    present_bits = []
    dates_used = []

    for offset in range(-5, 0):
        past_date = target_date + timedelta(days=offset)

        if past_date in dates_dict:
            past_path = dates_dict[past_date]
            try:
                past_dn = read_image(f"{cfg.DATA_DIR}/{past_path}")
                past_frames.append(past_dn)
                present_bits.append(1.0)
                dates_used.append(past_date.strftime('%Y-%m-%d'))
            except:
                past_frames.append(np.zeros((200, 200)))
                present_bits.append(0.0)
                dates_used.append(f"{past_date.strftime('%Y-%m-%d')} (missing)")
        else:
            past_frames.append(np.zeros((200, 200)))
            present_bits.append(0.0)
            dates_used.append(f"{past_date.strftime('%Y-%m-%d')} (missing)")

    # Check availability
    n_available = sum(present_bits)
    if n_available < 3:
        raise ValueError(f"Insufficient data: only {n_available}/5 past frames available")

    print(f"Using past frames:")
    for date, avail in zip(dates_used, present_bits):
        status = "✓" if avail else "✗"
        print(f"  {status} {date}")

    # Normalize
    water_mask = (past_frames[0] < 254)

    # Collect all water values from all frames
    all_water_vals = []
    for frame in past_frames:
        water_vals = frame[water_mask]
        if len(water_vals) > 0:
            all_water_vals.extend(water_vals)

    # Compute global p2/p98
    if len(all_water_vals) > 10:
        p2_global = np.percentile(all_water_vals, 2)
        p98_global = np.percentile(all_water_vals, 98)
        if p98_global - p2_global < 1e-6:
            p98_global = p2_global + 1.0
    else:
        p2_global, p98_global = 0, 253

    # Normalize all frames with same p2/p98
    normalized = []
    p2_vals = []
    p98_vals = []

    for frame in past_frames:
        norm, p2, p98 = normalize_robust(frame, water_mask)
        normalized.append(norm)
        p2_vals.append(p2)
        p98_vals.append(p98)

    # Use last frame's p2/p98 for denormalization (or mean)
    p2_global = np.mean(p2_vals)
    p98_global = np.mean(p98_vals)

    # Build input [1, 5, 6, H, W]
    H, W = past_frames[0].shape
    x = np.zeros((1, 5, 2, H, W), dtype=np.float32)

    for t in range(5):
        x[0, t, 0] = normalized[t]
        x[0, t, 1] = present_bits[t]

        # Predict
    x_tensor = torch.from_numpy(x).to(device)

    with torch.no_grad():
        pred_norm = model(x_tensor).cpu().numpy()[0, 0]

    # Denormalize
    pred_dn = denormalize(pred_norm, p2_global, p98_global)
    pred_dn = np.clip(pred_dn, 0, 253)

    return pred_dn


def main():
    parser = argparse.ArgumentParser(description='Predict HAB for specific date')
    parser.add_argument('--date', type=str, required=True,
                        help='Target date (YYYY-MM-DD)')
    parser.add_argument('--checkpoint', type=str, default='data/checkpoints/model_best.pt',
                        help='Model checkpoint path')
    args = parser.parse_args()

    # Parse date
    try:
        target_date = pd.to_datetime(args.date)
    except:
        print(f"Error: Invalid date format. Use YYYY-MM-DD")
        return

    print(f"\n{'=' * 60}")
    print(f"Predicting HAB for: {target_date.strftime('%Y-%m-%d')}")
    print(f"{'=' * 60}\n")

    # Load model
    print("Loading model...")
    model = HABInpaintModel(
        input_channels=cfg.INPUT_CHANNELS,
        hidden_dims=cfg.HIDDEN_DIMS,
        kernel_size=cfg.KERNEL_SIZE
    ).to(cfg.DEVICE)

    checkpoint = torch.load(args.checkpoint, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
    print(f"  Val loss: {checkpoint['metrics'].get('loss', 'N/A'):.4f}\n")

    # Load index
    index = pd.read_csv(cfg.INDEX_CSV)
    index['date'] = pd.to_datetime(index['date'])

    # Predict
    try:
        pred_dn = predict_date(model, target_date, index, cfg.DEVICE)

        print(f"\n✓ Prediction complete!")

        # Visualize
        plt.figure(figsize=(10, 8))
        plt.imshow(colorize_dn(pred_dn))
        plt.title(f"Predicted HAB - {target_date.strftime('%Y-%m-%d')}",
                  fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()

        # Save
        output_path = f"data/results/prediction_{args.date}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")

        plt.show()

        # Save DN array
        np.save(f"data/results/prediction_{args.date}.npy", pred_dn)

    except ValueError as e:
        print(f"\n✗ Error: {e}")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")


if __name__ == "__main__":
    main()