from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from train.data_loader import get_dataloader
from models.convlstm import ConvLSTM_Predictor

def evaluate(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy().transpose(1,2,0)
    y_pred = y_pred.detach().cpu().numpy().transpose(1,2,0)
    return ssim(y_true, y_pred, channel_axis=2, data_range=1.0), psnr(y_true, y_pred, data_range=1.0)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path("data_imgs/habs_month_images")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    loader = get_dataloader(data_dir, batch_size=1, seq_len=3, num_workers=0)

    model = ConvLSTM_Predictor(input_dim=3, hidden_dim=32, kernel_size=(3,3), n_layers=2).to(device)
    model.load_state_dict(torch.load("checkpoints/convlstm_epoch3.pth", map_location=device))
    model.eval()

    total_ssim, total_psnr, n = 0.0, 0.0, 0

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            ssim_val, psnr_val = evaluate(y[0], y_pred[0])
            total_ssim += ssim_val
            total_psnr += psnr_val
            n += 1

            # save visual
            if idx < 5:  # save first 5
                x_last = x[0, -1].cpu().permute(1, 2, 0)
                y_pred_img = y_pred[0].cpu().permute(1, 2, 0).clamp(0, 1)
                y_true_img = y[0].cpu().permute(1, 2, 0)

                plt.figure(figsize=(10, 3))
                plt.subplot(1, 3, 1)
                plt.imshow(x_last)
                plt.title("Input (t)")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(y_pred_img)
                plt.title("Predicted (t+1)")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.imshow(y_true_img)
                plt.title("Ground Truth")
                plt.axis("off")

                plt.tight_layout()
                save_path = results_dir / f"sample_{idx+1}.png"
                plt.savefig(save_path, dpi=150)
                plt.close()

    print(f"Avg SSIM: {total_ssim/n:.3f}, Avg PSNR: {total_psnr/n:.2f} dB")
    print(f"Saved example images to {results_dir.resolve()}/")
    

if __name__ == "__main__":
    main()