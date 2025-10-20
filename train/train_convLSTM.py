
'''
data_root = Path("data_imgs/light_cloud")  # or light_cloud if that’s ready
loader = get_dataloader(data_root, batch_size=2, seq_len=3)

x, y = next(iter(loader))
print("x:", x.shape, "y:", y.shape)

# Check model output
model = ConvLSTM(input_dim=3, hidden_dim=32, kernel_size=(3,3), n_layers=2)
output, states = model(x)
print("Model output:", output[0].shape)
'''

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from train.data_loader import get_dataloader
from models.convlstm import ConvLSTM, ConvLSTM_Predictor
from tqdm import tqdm

# --- metrics ---
import torchmetrics
ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0)
psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1.0)


def train():
    # --- configuration ---
    DATA_DIR = Path("data_imgs/light_cloud")
    SEQ_LEN = 3
    BATCH_SIZE = 1
    EPOCHS = 5
    LR = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    loader = get_dataloader(DATA_DIR, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, num_workers=0)

    # model + loss + optimizer
    model = ConvLSTM_Predictor(input_dim=3, hidden_dim=32, kernel_size=(3,3), n_layers=2).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        running_loss, running_ssim, running_psnr = 0.0, 0.0, 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            # compute metrics (detach to avoid gradients)
            with torch.no_grad():
                ssim_val = ssim_metric(y_pred, y)
                psnr_val = psnr_metric(y_pred, y)

            running_loss += loss.item()
            running_ssim += ssim_val.item()
            running_psnr += psnr_val.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", ssim=f"{ssim_val.item():.3f}", psnr=f"{psnr_val.item():.2f}")

        #  epoch summary 
        n = len(loader)
        avg_loss = running_loss / n
        avg_ssim = running_ssim / n
        avg_psnr = running_psnr / n
        print(f" Epoch {epoch+1} | Loss: {avg_loss:.4f} | SSIM: {avg_ssim:.3f} | PSNR: {avg_psnr:.2f}")

        # --- save model ---
        torch.save(model.state_dict(), f"checkpoints/convlstm_epoch{epoch+1}.pth")


if __name__ == "__main__":
    train()
