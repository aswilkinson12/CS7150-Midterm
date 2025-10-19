"""
ConvLSTM model for HAB cloud inpainting.
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding
        )

    def forward(self, x, hidden):
        h, c = hidden
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=3):
        super().__init__()
        self.num_layers = len(hidden_dims)
        self.hidden_dims = hidden_dims

        cells = []
        for i in range(self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            cells.append(ConvLSTMCell(cur_input_dim, hidden_dims[i], kernel_size))

        self.cells = nn.ModuleList(cells)

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        Returns: [B, hidden_dim, H, W] (last timestep, last layer)
        """
        B, T, C, H, W = x.size()

        # Init hidden states
        hidden_states = []
        for hidden_dim in self.hidden_dims:
            h = torch.zeros(B, hidden_dim, H, W, device=x.device)
            c = torch.zeros(B, hidden_dim, H, W, device=x.device)
            hidden_states.append((h, c))

        # Process sequence
        for t in range(T):
            x_t = x[:, t]

            for layer_idx in range(self.num_layers):
                h, c = hidden_states[layer_idx]
                h, c = self.cells[layer_idx](x_t, (h, c))
                hidden_states[layer_idx] = (h, c)
                x_t = h  # Input to next layer

        # Return last hidden state of last layer
        return hidden_states[-1][0]


class HABInpaintModel(nn.Module):
    """ConvLSTM for HAB inpainting."""

    def __init__(self, input_channels=8, hidden_dims=[32, 32, 32], kernel_size=3):
        super().__init__()

        self.convlstm = ConvLSTM(input_channels, hidden_dims, kernel_size)

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], hidden_dims[-1], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims[-1], 1, 1)
        )

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        Returns: [B, 1, H, W]
        """
        hidden = self.convlstm(x)
        out = self.decoder(hidden)
        return out