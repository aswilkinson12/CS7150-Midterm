import torch
import torch.nn as nn
from models.convlstm import ConvLSTM


class HABPredictionModelV2(nn.Module):
    """
    Improved HAB prediction with:
    - Deeper decoder
    - Batch normalization
    """

    def __init__(self, input_channels=2, hidden_dims=[64, 64, 64], kernel_size=3):
        super().__init__()

        self.convlstm = ConvLSTM(input_channels, hidden_dims, kernel_size)

        self.hidden_dims = hidden_dims

        # Decoder path - gradually reduce channels to output
        self.decoder = nn.Sequential(
            # Block 1: 64 -> 64
            nn.Conv2d(hidden_dims[-1], 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 2: 64 -> 32
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 3: 32 -> 16
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Output: 16 -> 1
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        Returns: [B, 1, H, W]
        """
        # Get ConvLSTM features
        hidden = self.convlstm(x)  # [B, 64, H, W]

        # Decode to output
        out = self.decoder(hidden)  # [B, 1, H, W]

        return out


class HABPredictionModelV3(nn.Module):

    def __init__(self, input_channels=2, hidden_dims=[64, 64, 64], kernel_size=3):
        super().__init__()

        # Encoder (ConvLSTM)
        self.convlstm = ConvLSTM(input_channels, hidden_dims, kernel_size)

        # Decoder with upsampling
        self.up1 = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Final output
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        Returns: [B, 1, H, W]
        """
        # Encoder
        hidden = self.convlstm(x)  # [B, 64, H, W]

        # Decoder
        x = self.up1(hidden)  # [B, 64, H, W]
        x = self.up2(x)  # [B, 32, H, W]
        x = self.up3(x)  # [B, 16, H, W]
        out = self.output(x)  # [B, 1, H, W]

        return out