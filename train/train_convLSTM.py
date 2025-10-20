import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
from pathlib import Path
from train.data_loader import SatelliteSequenceDataset
from models.convlstm import ConvLSTM, ConvLSTM_Predictor
from tqdm import tqdm
from config import *
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio


def train():
    # configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset split
    full_dataset = SatelliteSequenceDataset(DATA_DIR, seq_len=SEQ_LEN)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    # model + loss + optimizer
    model = ConvLSTM_Predictor(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, kernel_size=KERNEL_SIZE, n_layers=N_LAYERS).to(DEVICE)
    criterion = lambda pred, target: 0.8 * nn.L1Loss()(pred, target) + 0.2 * (1 - ssim_metric(pred, target))

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # metrics 
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=255.0)
    psnr_metric = PeakSignalNoiseRatio(data_range=255.0)

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
        best_val_loss = float('inf')
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Val Epoch {epoch+1}"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss += loss.item()

                ssim_val = ssim_metric(y_pred, y)
                psnr_val = psnr_metric(y_pred, y)

                val_ssim += ssim_val.item()
                val_psnr += psnr_val.item()

        n_val = len(val_loader)
        avg_val_loss = val_loss / n_val
        avg_val_ssim = val_ssim / n_val
        avg_val_psnr = val_psnr / n_val

        print(f" Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | SSIM: {avg_val_ssim:.3f} | PSNR: {avg_val_psnr:.2f}")

        # save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CHECKPOINT_DIR / "convlstm_best.pth")
            print(f" Saved new best model (Val Loss: {best_val_loss:.4f})")


if __name__ == "__main__":
    train()
