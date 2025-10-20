"""
evaluate.py - Comprehensive evaluation with metrics and visualizations
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import pandas as pd
from datetime import timedelta
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import json

import config as cfg
from models.convlstm import HABInpaintModel
from utils.data_loader import read_image, normalize_robust

# CyAN colormap
CYAN_CMAP = LinearSegmentedColormap.from_list(
    "cyan",
    [(0.00, (26/255, 0/255, 102/255)),
     (0.25, (0/255, 128/255, 255/255)),
     (0.50, (0/255, 255/255, 102/255)),
     (0.75, (255/255, 255/255, 0/255)),
     (1.00, (255/255, 51/255, 0/255))]
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


def compute_metrics(pred, target, water_mask):
    """Comprehensive metrics."""
    # Extract water pixels
    pred_water = pred[water_mask]
    target_water = target[water_mask]

    if len(pred_water) == 0:
        return {
            'mae': 0.0,
            'rmse': 0.0,
            'correlation': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'pred_mean_intensity': 0.0,
            'target_mean_intensity': 0.0,
            'pred_std_intensity': 0.0,
            'target_std_intensity': 0.0,
            'mean_bias': 0.0
        }

    # Pixel-wise metrics
    mae = mean_absolute_error(target_water, pred_water)
    mse = mean_squared_error(target_water, pred_water)
    rmse = np.sqrt(mse)

    # Correlation
    corr, _ = pearsonr(target_water.flatten(), pred_water.flatten()) if len(pred_water) > 1 else (0, 1)

    # Bloom detection metrics (DN > 10 = bloom)
    bloom_threshold = 10
    pred_bloom = pred_water > bloom_threshold
    target_bloom = target_water > bloom_threshold

    if pred_bloom.any() or target_bloom.any():
        tp = (pred_bloom & target_bloom).sum()
        fp = (pred_bloom & ~target_bloom).sum()
        fn = (~pred_bloom & target_bloom).sum()
        tn = (~pred_bloom & ~target_bloom).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        precision = recall = f1 = 0

    # Intensity distribution
    pred_mean = pred_water.mean()
    target_mean = target_water.mean()
    pred_std = pred_water.std()
    target_std = target_water.std()

    return {
        'mae': mae,
        'rmse': rmse,
        'correlation': corr,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'pred_mean_intensity': pred_mean,
        'target_mean_intensity': target_mean,
        'pred_std_intensity': pred_std,
        'target_std_intensity': target_std,
        'mean_bias': pred_mean - target_mean
    }


def extract_features(model, x, layer_name='convlstm'):
    """Extract intermediate features for visualization."""
    features = {}

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, list):
                # ConvLSTM returns list
                features[name] = output[-1].detach().cpu().numpy()
            else:
                features[name] = output.detach().cpu().numpy()
        return hook

    # Register hook
    if hasattr(model, 'convlstm'):
        model.convlstm.register_forward_hook(hook_fn('convlstm'))

    # Forward pass
    with torch.no_grad():
        _ = model(x)

    return features


def predict_single_frame(model, past_frames, present_bits, water_mask, device, target_frame=None,
                         return_features=False):
    """Predict with correct denormalization.

    Args:
        model: Trained HAB model
        past_frames: List of 5 past DN arrays
        present_bits: List of 5 presence indicators
        water_mask: Binary mask for water pixels
        device: torch device
        target_frame: Ground truth DN array (for correct denormalization stats)
        return_features: Whether to extract intermediate features

    Returns:
        pred_dn: Predicted DN array
        features: (optional) Extracted feature maps
    """
    H, W = past_frames[0].shape

    # Normalize each input frame independently (matches training)
    normalized_frames = []
    for frame in past_frames:
        norm, _, _ = normalize_robust(frame, water_mask)
        normalized_frames.append(norm)

    # Build input [1, T=5, C=2, H, W]
    x = np.zeros((1, 5, 2, H, W), dtype=np.float32)
    for t in range(5):
        x[0, t, 0] = normalized_frames[t]  # Frame
        x[0, t, 1] = present_bits[t]  # Present bit

    x_tensor = torch.from_numpy(x).to(device)

    # Predict
    with torch.no_grad():
        pred_norm = model(x_tensor).cpu().numpy()[0, 0]

    # Denormalize using appropriate stats
    if target_frame is not None:
        # Use target frame's stats (ground truth available - for evaluation)
        _, p2_denorm, p98_denorm = normalize_robust(target_frame, water_mask)
    else:
        # Use mean of input frames (inference mode - no ground truth)
        p2_vals, p98_vals = [], []
        for frame in past_frames:
            _, p2, p98 = normalize_robust(frame, water_mask)
            p2_vals.append(p2)
            p98_vals.append(p98)
        p2_denorm = np.mean(p2_vals)
        p98_denorm = np.mean(p98_vals)

    pred_dn = denormalize(pred_norm, p2_denorm, p98_denorm)

    if return_features:
        features = extract_features(model, x_tensor)
        return pred_dn, features
    else:
        return pred_dn

def visualize_features(features, save_path):
    """Visualize learned features."""
    if 'convlstm' not in features:
        return

    feat = features['convlstm'][0]  # [C, H, W]
    n_channels = min(16, feat.shape[0])

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(n_channels):
        ax = axes[i // 4, i % 4]
        ax.imshow(feat[i], cmap='viridis')
        ax.set_title(f'Channel {i}', fontsize=8)
        ax.axis('off')

    plt.suptitle('Learned Feature Maps (ConvLSTM)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_intensity_distribution(all_preds, all_targets, save_path):
    """Plot prediction vs target intensity distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(all_targets, bins=50, alpha=0.5, label='Ground Truth', color='blue')
    axes[0].hist(all_preds, bins=50, alpha=0.5, label='Predicted', color='red')
    axes[0].set_xlabel('DN Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Intensity Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Scatter plot
    axes[1].scatter(all_targets, all_preds, alpha=0.1, s=1)
    axes[1].plot([0, 253], [0, 253], 'r--', label='Perfect Prediction')
    axes[1].set_xlabel('Ground Truth DN')
    axes[1].set_ylabel('Predicted DN')
    axes[1].set_title('Prediction vs Ground Truth')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0, 253)
    axes[1].set_ylim(0, 253)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_summary(metrics_list, save_path):
    """Plot summary of all metrics."""
    df = pd.DataFrame(metrics_list)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # MAE
    axes[0, 0].bar(range(len(df)), df['mae'])
    axes[0, 0].set_title('Mean Absolute Error')
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].axhline(df['mae'].mean(), color='r', linestyle='--', label=f'Mean: {df["mae"].mean():.2f}')
    axes[0, 0].legend()

    # Correlation
    axes[0, 1].bar(range(len(df)), df['correlation'])
    axes[0, 1].set_title('Correlation')
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('Pearson r')
    axes[0, 1].axhline(df['correlation'].mean(), color='r', linestyle='--', label=f'Mean: {df["correlation"].mean():.2f}')
    axes[0, 1].legend()

    # F1 Score
    axes[0, 2].bar(range(len(df)), df['f1_score'])
    axes[0, 2].set_title('F1 Score (Bloom Detection)')
    axes[0, 2].set_xlabel('Sample')
    axes[0, 2].set_ylabel('F1')
    axes[0, 2].axhline(df['f1_score'].mean(), color='r', linestyle='--', label=f'Mean: {df["f1_score"].mean():.2f}')
    axes[0, 2].legend()

    # Mean bias
    axes[1, 0].bar(range(len(df)), df['mean_bias'])
    axes[1, 0].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].set_title('Prediction Bias')
    axes[1, 0].set_xlabel('Sample')
    axes[1, 0].set_ylabel('Bias (Pred - Target)')

    # RMSE
    axes[1, 1].bar(range(len(df)), df['rmse'])
    axes[1, 1].set_title('Root Mean Square Error')
    axes[1, 1].set_xlabel('Sample')
    axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].axhline(df['rmse'].mean(), color='r', linestyle='--', label=f'Mean: {df["rmse"].mean():.2f}')
    axes[1, 1].legend()

    # Summary stats table
    axes[1, 2].axis('off')
    summary_text = f"""
    Summary Statistics:
    
    MAE: {df['mae'].mean():.2f} ± {df['mae'].std():.2f}
    RMSE: {df['rmse'].mean():.2f} ± {df['rmse'].std():.2f}
    Correlation: {df['correlation'].mean():.3f} ± {df['correlation'].std():.3f}
    
    Bloom Detection:
    Precision: {df['precision'].mean():.3f}
    Recall: {df['recall'].mean():.3f}
    F1 Score: {df['f1_score'].mean():.3f}
    
    Mean Bias: {df['mean_bias'].mean():.2f}
    """
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                   verticalalignment='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_comprehensive():
    """Comprehensive evaluation with all metrics and visualizations."""
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

    print(f" Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
    print(f"  Best val loss: {checkpoint['metrics'].get('loss', 'N/A'):.4f}")

    # Load test files
    index = pd.read_csv(cfg.INDEX_CSV)
    index['date'] = pd.to_datetime(index['date'])
    dates_dict = dict(zip(index['date'], index['path']))
    test_df = index[index['split'] == 'test'].reset_index(drop=True)

    print(f"\nProcessing {len(test_df)} test images...")

    # Storage
    all_metrics = []
    all_preds = []
    all_targets = []
    saved = 0
    max_save = 20  # More samples for better stats

    for idx, row in tqdm(test_df.iterrows(), total=min(len(test_df), max_save), desc="Evaluating"):
        if saved >= max_save:
            break

        target_path = row['path']
        target_date = row['date']

        # Load target
        target_dn = read_image(os.path.join(cfg.DATA_DIR, target_path))
        water_mask = (target_dn < 254)

        # Skip heavily clouded targets
        bloom_mask = (target_dn >= 1) & (target_dn <= 253)
        bloom_coverage = bloom_mask.sum() / water_mask.sum() if water_mask.sum() > 0 else 0

        if bloom_coverage < 0.05:  # Less than 5% bloom data
            continue

        # Load past frames
        past_frames = []
        present_bits = []

        for offset in range(-5, 0):
            past_date = target_date + timedelta(days=offset)
            if past_date in dates_dict:
                past_path = dates_dict[past_date]
                past_dn = read_image(os.path.join(cfg.DATA_DIR, past_path))
                past_frames.append(past_dn)
                present_bits.append(1.0)
            else:
                past_frames.append(np.zeros_like(target_dn))
                present_bits.append(0.0)

        if sum(present_bits) < 3:
            continue

        # Predict (with features for first sample)
        if saved == 0:
            pred_dn, features = predict_single_frame(
                model, past_frames, present_bits, water_mask, cfg.DEVICE, return_features=True
            )
            visualize_features(features, os.path.join(cfg.RESULTS_DIR, 'learned_features.png'))
        else:
            pred_dn = predict_single_frame(
                model, past_frames, present_bits, water_mask, cfg.DEVICE
            )

        # Compute metrics
        metrics = compute_metrics(pred_dn, target_dn, water_mask)
        metrics['date'] = target_date.strftime('%Y-%m-%d')
        metrics['n_available'] = sum(present_bits)
        all_metrics.append(metrics)

        # Store for distribution plots
        all_preds.extend(pred_dn[water_mask].flatten())
        all_targets.extend(target_dn[water_mask].flatten())

        # Visualize sample
        if saved < 10:  # Save first 10 visualizations
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            axes[0].imshow(colorize_dn(target_dn))
            axes[0].set_title(f"Ground Truth\n{target_date.strftime('%Y-%m-%d')}", fontsize=12)
            axes[0].axis('off')

            axes[1].imshow(colorize_dn(pred_dn))
            axes[1].set_title(f"Prediction\nMAE: {metrics['mae']:.2f}, Corr: {metrics['correlation']:.3f}", fontsize=12)
            axes[1].axis('off')

            error = np.abs(pred_dn.astype(float) - target_dn.astype(float))
            error_vis = np.zeros_like(target_dn, dtype=float)
            error_vis[water_mask] = error[water_mask]
            im = axes[2].imshow(error_vis, cmap='hot', vmin=0, vmax=50)
            axes[2].set_title("Prediction Error", fontsize=12)
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2], fraction=0.046)

            plt.tight_layout()
            plt.savefig(os.path.join(cfg.RESULTS_DIR, f"prediction_{saved:03d}.png"), dpi=150)
            plt.close()

        saved += 1

    # Summary plots
    print("\nGenerating summary visualizations...")

    plot_intensity_distribution(
        np.array(all_preds),
        np.array(all_targets),
        os.path.join(cfg.RESULTS_DIR, 'intensity_distribution.png')
    )

    plot_metrics_summary(
        all_metrics,
        os.path.join(cfg.RESULTS_DIR, 'metrics_summary.png')
    )

    # Save metrics to CSV and JSON
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(os.path.join(cfg.RESULTS_DIR, 'metrics.csv'), index=False)

    summary_stats = {
        'n_samples': len(all_metrics),
        'mae_mean': float(df_metrics['mae'].mean()),
        'mae_std': float(df_metrics['mae'].std()),
        'rmse_mean': float(df_metrics['rmse'].mean()),
        'correlation_mean': float(df_metrics['correlation'].mean()),
        'f1_score_mean': float(df_metrics['f1_score'].mean()),
        'mean_bias': float(df_metrics['mean_bias'].mean()),
    }

    with open(os.path.join(cfg.RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary_stats, f, indent=2)

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Samples evaluated: {len(all_metrics)}")
    print(f"\nPixel-wise Metrics:")
    print(f"  MAE:  {summary_stats['mae_mean']:.2f} ± {summary_stats['mae_std']:.2f}")
    print(f"  RMSE: {summary_stats['rmse_mean']:.2f}")
    print(f"  Correlation: {summary_stats['correlation_mean']:.3f}")
    print(f"\nBloom Detection:")
    print(f"  F1 Score: {summary_stats['f1_score_mean']:.3f}")
    print(f"\nBias:")
    print(f"  Mean Bias: {summary_stats['mean_bias']:.2f}")
    print(f"\nResults saved to: {cfg.RESULTS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    evaluate_comprehensive()