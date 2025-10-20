from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
from pathlib import Path
from train.data_loader import get_dataloader
from models.convlstm import ConvLSTM_Predictor

def evaluate(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy().transpose(1,2,0)
    y_pred = y_pred.detach().cpu().numpy().transpose(1,2,0)
    return ssim(y_true, y_pred, channel_axis=2), psnr(y_true, y_pred)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path("data_imgs/habs_month_images")
    loader = get_dataloader(data_dir, batch_size=1, seq_len=3, num_workers=0)

    model = ConvLSTM_Predictor(input_dim=3, hidden_dim=32, kernel_size=(3,3), n_layers=2).to(device)
    model.load_state_dict(torch.load("checkpoints/convlstm_epoch3.pth", map_location=device))
    model.eval()

    total_ssim, total_psnr, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            ssim_val, psnr_val = evaluate(y[0], y_pred[0])
            total_ssim += ssim_val
            total_psnr += psnr_val
            n += 1
    print(f"Avg SSIM: {total_ssim/n:.3f}, Avg PSNR: {total_psnr/n:.2f} dB")

if __name__ == "__main__":
    main()