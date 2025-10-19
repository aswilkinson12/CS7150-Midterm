"""
Loss functions for ConvLSTM inpainting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_l1(pred, target, mask):
    """L1 loss over masked region only."""
    mask_bool = mask > 0.5
    if not mask_bool.any():
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    diff = torch.abs(pred - target)
    masked_diff = diff[mask_bool]
    return masked_diff.mean()


def masked_ssim(pred, target, mask, window_size=7):
    """
    Masked SSIM loss (returns 1-SSIM, so lower is better).
    Simplified implementation - computes SSIM over masked region.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mask_bool = mask > 0.5
    if not mask_bool.any():
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    # Apply mask
    pred_masked = pred * mask
    target_masked = target * mask

    # Compute local statistics with convolution
    kernel = torch.ones(1, 1, window_size, window_size, device=pred.device) / (window_size ** 2)

    mu1 = F.conv2d(pred_masked, kernel, padding=window_size//2)
    mu2 = F.conv2d(target_masked, kernel, padding=window_size//2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred_masked * pred_masked, kernel, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(target_masked * target_masked, kernel, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(pred_masked * target_masked, kernel, padding=window_size//2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # Average over masked region
    ssim_masked = ssim_map[mask_bool]
    return 1.0 - ssim_masked.mean()


class DistanceWeightedComboLoss(nn.Module):
    """
    Combined L1 + SSIM loss with distance weighting.
    Loss is computed only in masked region, with interior pixels weighted more.
    """

    def __init__(self, alpha=0.8, ssim_window=7, use_dist_weights=True):
        super().__init__()
        self.alpha = alpha
        self.ssim_window = ssim_window
        self.use_dist_weights = use_dist_weights

    def forward(self, pred, target, mask, dist_weights):
        """
        Args:
            pred: [B, 1, H, W] predictions
            target: [B, 1, H, W] ground truth
            mask: [B, 1, H, W] binary mask (1 where cloudy)
            dist_weights: [B, 1, H, W] distance transform weights
        """
        # Compute base losses
        l1_loss = masked_l1(pred, target, mask)
        ssim_loss = masked_ssim(pred, target, mask, window_size=self.ssim_window)

        # Apply distance weighting if enabled
        if self.use_dist_weights:
            mask_bool = mask > 0.5
            if mask_bool.any():
                diff = torch.abs(pred - target)
                weighted_diff = diff * dist_weights * mask
                dist_weighted_l1 = weighted_diff[mask_bool].mean()

                # Recompute combo with weighted L1
                combo_loss = self.alpha * dist_weighted_l1 + (1.0 - self.alpha) * ssim_loss
            else:
                combo_loss = self.alpha * l1_loss + (1.0 - self.alpha) * ssim_loss
        else:
            combo_loss = self.alpha * l1_loss + (1.0 - self.alpha) * ssim_loss

        # Return tensors with gradients, not Python floats
        return combo_loss, {
            'l1': l1_loss.item(),  # .item() for logging only
            'ssim': ssim_loss.item()
        }