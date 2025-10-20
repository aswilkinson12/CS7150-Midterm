import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

from config import *
from train.data_loader import get_dataloader
from models.convlstm import ConvLSTM_Predictor


def masked_rmse(y_true, y_pred, mask):
    diff = (y_true - y_pred)[mask]
    return np.sqrt(np.mean(diff ** 2)) if diff.size > 0 else 0.0


def evaluate_metrics(y_true, y_pred):
    y_true_np = y_true.detach().cpu().numpy().transpose(1, 2, 0)
    y_pred_np = y_pred.detach().cpu().numpy().transpose(1, 2, 0)

    # Mask out clouds/land (bright or near-constant pixels)
    mask = (y_true_np < 0.99).all(axis=2)

    ssim_val = ssim(y_true_np, y_pred_np, channel_axis=2, data_range=1.0)
    psnr_val = psnr(y_true_np, y_pred_np, data_range=1.0)
    rmse_val = masked_rmse(y_true_np, y_pred_np, mask)

    return ssim_val, psnr_val, rmse_val, mask, y_true_np, y_pred_np


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(RESULTS_DIR).mkdir(exist_ok=True, parents=True)

    loader = get_dataloader(DATA_DIR, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, num_workers=NUM_WORKERS)
    model = ConvLSTM_Predictor(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                               kernel_size=KERNEL_SIZE, n_layers=N_LAYERS).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    total_ssim, total_psnr, total_rmse, n = 0, 0, 0, 0

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            ssim_val, psnr_val, rmse_val, mask, y_true_np, y_pred_np = evaluate_metrics(y[0], y_pred[0])
            total_ssim += ssim_val
            total_psnr += psnr_val
            total_rmse += rmse_val
            n += 1

            if idx < 5:
                x_last = x[0, -1].cpu().permute(1, 2, 0)
                err = np.abs(y_true_np - y_pred_np)

                # 4-panel comparison
                fig, axes = plt.subplots(1, 4, figsize=(14, 3))
                axes[0].imshow(x_last)
                axes[0].set_title("Input (t)")
                axes[1].imshow(y_pred_np)
                axes[1].set_title("Predicted (t+1)")
                axes[2].imshow(y_true_np)
                axes[2].set_title("Ground Truth")
                axes[3].imshow(err, cmap="magma")
                axes[3].set_title("Error Map")
                for a in axes: a.axis("off")
                plt.tight_layout()
                plt.savefig(RESULTS_DIR / f"sample_{idx+1}.png", dpi=150)
                plt.close()

                # Heatmap visualization (masked)
                fig, ax = plt.subplots(figsize=(5, 4))
                masked_err = np.where(mask[..., None], err, np.nan)
                im = ax.imshow(masked_err.mean(axis=2), cmap="plasma", vmin=0, vmax=0.5)
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Absolute Error")
                ax.set_title(f"Heatmap Error | SSIM={ssim_val:.3f}, PSNR={psnr_val:.2f}")
                ax.axis("off")
                plt.tight_layout()
                plt.savefig(RESULTS_DIR / f"heatmap_{idx+1}.png", dpi=150)
                plt.close()

                print(f"[Saved] sample_{idx+1}.png + heatmap_{idx+1}.png | "
                      f"SSIM={ssim_val:.3f}, PSNR={psnr_val:.2f}, RMSE={rmse_val:.4f}")

    print("\n=== Average Metrics ===")
    print(f"SSIM: {total_ssim/n:.3f}")
    print(f"PSNR: {total_psnr/n:.2f} dB")
    print(f"Masked RMSE: {total_rmse/n:.4f}")
    print(f"Saved results to {Path(RESULTS_DIR).resolve()}/")


if __name__ == "__main__":
    main()
