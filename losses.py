import torch
from torch import nn


class SimpleMaskedL1Loss(nn.Module):
    """Plain L1 loss on masked region only."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target, mask, dist_weights=None):
        mask_bool = mask > 0.5
        
        if not mask_bool.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True), {'l1': 0.0}
        
        # L1 loss
        diff = torch.abs(pred - target)
        masked_diff = diff[mask_bool]
        l1_loss = masked_diff.mean()
        
        return l1_loss, {'l1': l1_loss.item()} 