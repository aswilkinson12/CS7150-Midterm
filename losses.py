"""
losses.py - Loss functions for HAB prediction
"""

import torch
from torch import nn


class WaterMaskedMSELoss(nn.Module):
    """MSE loss only on water regions (ignore land)."""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, water_mask):
        """
        pred: [B, 1, H, W] - predicted frame
        target: [B, 1, H, W] - actual frame
        water_mask: [B, 1, H, W] - binary mask (1=water, 0=land)
        """
        mask_bool = water_mask > 0.5

        if not mask_bool.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True), {
                'mse': 0.0, 'mae': 0.0
            }

        # MSE on water only
        diff = (pred - target) ** 2
        mse_loss = diff[mask_bool].mean()

        # Also track MAE
        mae = torch.abs(pred - target)[mask_bool].mean()

        return mse_loss, {
            'mse': mse_loss.item(),
            'mae': mae.item()
        }


class WaterMaskedMAELoss(nn.Module):
    """MAE (L1) loss only on water regions."""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, water_mask):
        """
        pred: [B, 1, H, W]
        target: [B, 1, H, W]
        water_mask: [B, 1, H, W]
        """
        mask_bool = water_mask > 0.5

        if not mask_bool.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True), {
                'mae': 0.0
            }

        # MAE on water only
        diff = torch.abs(pred - target)
        mae_loss = diff[mask_bool].mean()

        return mae_loss, {'mae': mae_loss.item()}