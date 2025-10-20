
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
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
from pathlib import Path
from train.data_loader import SatelliteSequenceDataset
from models.convlstm import ConvLSTM, ConvLSTM_Predictor
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio


def train():
    # configuration
    DATA_DIR = Path("data_imgs/hab_month_images")
    SEQ_LEN = 3
    BATCH_SIZE = 1
    EPOCHS = 5
    LR = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset split
    full_dataset = SatelliteSequenceDataset(DATA_DIR, seq_len=SEQ_LEN)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    # model + loss + optimizer
    model = ConvLSTM_Predictor(input_dim=3, hidden_dim=32, kernel_size=(3,3), n_layers=2).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # metrics 
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)

    for epoch in range(EPOCHS):
        # train
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{EPOCHS}")
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_train_loss = running_loss / len(train_loader)

        # validation
        model.eval()
        val_loss, val_ssim, val_psnr = 0.0, 0.0, 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Val Epoch {epoch+1}"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss += loss.item()

                ssim_val = ssim_metric(y_pred.detach().cpu(), y.detach().cpu())
                psnr_val = psnr_metric(y_pred.detach().cpu(), y.detach().cpu())

                val_ssim += ssim_val.item()
                val_psnr += psnr_val.item()

        n_val = len(val_loader)
        avg_val_loss = val_loss / n_val
        avg_val_ssim = val_ssim / n_val
        avg_val_psnr = val_psnr / n_val

        print(f" Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | SSIM: {avg_val_ssim:.3f} | PSNR: {avg_val_psnr:.2f}")

        # save checkpoint
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/convlstm_epoch{epoch+1}.pth")


if __name__ == "__main__":
    train()
