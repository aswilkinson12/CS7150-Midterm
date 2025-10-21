"""
train.py - Training script for HAB prediction
"""

import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
from tqdm import tqdm
import config as cfg
from loss_v2 import IntensityWeightedMSELoss

from models.convlstm_v2 import HABPredictionModelV2 as HABInpaintModel
from utils.data_loader import create_dataloaders
from losses import WaterMaskedMSELoss


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()

    running_loss = 0.0
    running_mse = 0.0
    running_mae = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Train]")

    for x, y, water_mask in pbar:
        # Move to device
        x = x.to(device)  # [B, T=5, C=6, H, W]
        y = y.to(device)  # [B, 1, H, W]
        water_mask = water_mask.to(device)  # [B, 1, H, W]

        # Forward pass
        pred = model(x)  # [B, 1, H, W]

        # Compute loss
        loss, loss_dict = criterion(pred, y, water_mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)

        optimizer.step()

        # Track metrics
        running_loss += loss.item()
        running_mse += loss_dict.get('mse', loss.item())
        running_mae += loss_dict.get('mae', 0.0)
        n_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mae': f"{loss_dict.get('mae', 0.0):.4f}",
        })

    # Average metrics
    avg_loss = running_loss / n_batches
    avg_mse = running_mse / n_batches
    avg_mae = running_mae / n_batches

    return {
        'loss': avg_loss,
        'mse': avg_mse,
        'mae': avg_mae,
    }


def validate(model, dataloader, criterion, device, epoch):
    """Validate model."""
    model.eval()

    running_loss = 0.0
    running_mse = 0.0
    running_mae = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Val]")

    with torch.no_grad():
        for x, y, water_mask in pbar:
            x = x.to(device)
            y = y.to(device)
            water_mask = water_mask.to(device)

            # Forward pass
            pred = model(x)

            # Compute loss
            loss, loss_dict = criterion(pred, y, water_mask)

            # Track metrics
            running_loss += loss.item()
            running_mse += loss_dict.get('mse', loss.item())
            running_mae += loss_dict.get('mae', 0.0)
            n_batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}"
            })

    # Average metrics
    avg_loss = running_loss / n_batches
    avg_mse = running_mse / n_batches
    avg_mae = running_mae / n_batches

    return {
        'loss': avg_loss,
        'mse': avg_mse,
        'mae': avg_mae,
    }


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, path)
    print(f"Saved checkpoint: {path}")


def main():
    """Main training function."""

    # Create output directories
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Set device
    device = cfg.DEVICE
    print(f"Using device: {device}")

    # Create dataloaders
    print("\nLoading data...")
    loaders = create_dataloaders(
        index_csv=cfg.INDEX_CSV,
        batch_size=cfg.BATCH_SIZE,
        num_workers=4
    )

    print(f"Train batches: {len(loaders['train'])}")
    print(f"Val batches: {len(loaders['val'])}")

    # Create model
    print("\nInitializing model...")
    model = HABInpaintModel(
        input_channels=cfg.INPUT_CHANNELS,
        hidden_dims=cfg.HIDDEN_DIMS,
        kernel_size=cfg.KERNEL_SIZE
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Create loss function
    criterion = IntensityWeightedMSELoss(weight_power=0.8)

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY
    )

    # Create learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    # Training loop
    print("\nStarting training...")
    print("=" * 70)

    best_loss = float('inf')
    patience = 10
    patience_counter = 0

    log_data = []

    for epoch in range(cfg.EPOCHS):
        # Train
        train_metrics = train_epoch(
            model, loaders['train'], criterion, optimizer, device, epoch
        )

        # Validate
        val_metrics = validate(
            model, loaders['val'], criterion, device, epoch
        )

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics
        log_entry = {
            'epoch': epoch + 1,
            'lr': current_lr,
            'train_loss': train_metrics['loss'],
            'train_mse': train_metrics['mse'],
            'train_mae': train_metrics['mae'],
            'val_loss': val_metrics['loss'],
            'val_mse': val_metrics['mse'],
            'val_mae': val_metrics['mae']
        }
        log_data.append(log_entry)

        # Print summary
        print(f"\nEpoch {epoch + 1}/{cfg.EPOCHS}")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, f"model_epoch_{epoch + 1:03d}.pt")
        save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_path)

        # Early stopping
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            patience_counter = 0

            # Save best model
            best_path = os.path.join(cfg.CHECKPOINT_DIR, "model_best.pt")
            save_checkpoint(model, optimizer, epoch, val_metrics, best_path)
            print(f"  ★ New best loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")

        # Check early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

        print("=" * 70)

    # Save training log
    log_df = pd.DataFrame(log_data)
    log_path = os.path.join(cfg.OUTPUT_DIR, "train_log.csv")
    log_df.to_csv(log_path, index=False)
    print(f"\nTraining log saved to: {log_path}")

    print(f"\nTraining complete! Best val loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()