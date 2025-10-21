import torch
from torch import nn


class IntensityWeightedMSELoss(nn.Module):
    """MSE with higher weight on high-intensity errors."""

    def __init__(self, weight_power=1.5):
        super().__init__()
        self.weight_power = weight_power

    def forward(self, pred, target, water_mask):
        mask_bool = water_mask > 0.5

        if not mask_bool.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True), {
                'mse': 0.0, 'mae': 0.0
            }

        # Weights increase with target intensity
        # target in [0,1], weights in [1, 2^power]
        weights = 1.0 + target ** self.weight_power

        # Weighted MSE
        diff_sq = (pred - target) ** 2
        mse_loss = (diff_sq * weights)[mask_bool].mean()

        # Track MAE too
        mae = torch.abs(pred - target)[mask_bool].mean()

        return mse_loss, {'mse': mse_loss.item(), 'mae': mae.item()}