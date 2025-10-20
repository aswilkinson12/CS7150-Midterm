import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
from tqdm import tqdm
import argparse
import config as cfg

from models.convlstm import HABInpaintModel
from utils.data_loader import create_dataloaders
from losses import SimpleMaskedL1Loss


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()

    running_loss = 0.0
    running_l1 = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Train]")

    for x, y, mask, dist_weights in pbar:
        # Move to device
        x = x.to(device)  # [B, T=6, C=8, H, W]
        y = y.to(device)  # [B, 1, H, W]
        mask = mask.to(device)  # [B, 1, H, W]
        dist_weights = dist_weights.to(device)  # [B, 1, H, W]

        # Forward pass
        pred = model(x)  # [B, 1, H, W]

        # Compute loss
        loss, loss_dict = criterion(pred, y, mask, dist_weights)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)

        optimizer.step()

        # Track metrics
        running_loss += loss.item()
        running_l1 += loss_dict['l1']
        n_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'l1': f"{loss_dict['l1']:.4f}",
        })

    # Average metrics
    avg_loss = running_loss / n_batches
    avg_l1 = running_l1 / n_batches

    return {
        'loss': avg_loss,
        'l1': avg_l1,
    }


def validate(model, dataloader, criterion, device, epoch):
    """Validate model."""
    model.eval()

    running_loss = 0.0
    running_l1 = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Val]")

    with torch.no_grad():
        for x, y, mask, dist_weights in pbar:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            dist_weights = dist_weights.to(device)

            # Forward pass
            pred = model(x)

            # Compute loss
            loss, loss_dict = criterion(pred, y, mask, dist_weights)


            # Track metrics
            running_loss += loss.item()
            running_l1 += loss_dict['l1']
            n_batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}"
            })

    # Average metrics
    avg_loss = running_loss / n_batches
    avg_l1 = running_l1 / n_batches

    return {
        'loss': avg_loss,
        'l1': avg_l1,
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


def main(args):
    """Main training function."""

    # Create output directories
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataloaders
    print("\nLoading data...")
    loaders = create_dataloaders(
        index_csv=INDEX_CSV,
        batch_size=BATCH_SIZE,
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
    criterion = SimpleMaskedL1Loss()

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=cfg.WEIGHT_DECAY
    )

    # Create learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training loop
    print("\nStarting training...")
    print("=" * 70)

    best_l1 = float('inf')
    patience = 10
    patience_counter = 0

    log_data = []

    for epoch in range(EPOCHS):
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
            'train_l1': train_metrics['l1'],
            'val_loss': val_metrics['loss'],
            'val_l1': val_metrics['l1']
        }
        log_data.append(log_entry)

        # Print summary
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, L1: {train_metrics['l1']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, L1: {val_metrics['l1']:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, f"model_epoch_{epoch + 1:03d}.pt")
        save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_path)

        # Early stopping based on loss score
        if val_metrics['l1'] < best_l1:
            best_l1 = val_metrics['l1']
            patience_counter = 0

            # Save best model
            best_path = os.path.join(cfg.CHECKPOINT_DIR, "model_best.pt")
            save_checkpoint(model, optimizer, epoch, val_metrics, best_path)
            print(f"  ★ New best L1: {best_l1:.4f}")
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

    print(f"\n Training complete! Best val L1: {best_l1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ConvLSTM inpainting model")
    parser.add_argument('--index_csv', type=str, default=cfg.INDEX_CSV, help='Path to index CSV')
    parser.add_argument('--data_root', type=str, default=cfg.DATA_ROOT, help='Data root directory')
    parser.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE, help='Batch size')
    parser.add_argument('--epochs', type=int, default=cfg.EPOCHS, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=cfg.LR, help='Learning rate')

    args = parser.parse_args()

    # Update globals from args
    INDEX_CSV = args.index_csv
    DATA_ROOT = args.data_root
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr

    main(args)