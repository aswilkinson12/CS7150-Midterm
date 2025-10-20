import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, batch_size, spatial_size, device):
        H, W = spatial_size
        h = torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, n_layers, bias=True, batch_first=True):
        super().__init__()
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.cells = nn.ModuleList([
            ConvLSTMCell(
                input_dim=input_dim if i == 0 else hidden_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                bias=bias
            ) for i in range(n_layers)
        ])

    def forward(self, x):
        # x: (B, T, C, H, W)
        if not self.batch_first:
            x = x.permute(1, 0, 2, 3, 4)

        B, T, C, H, W = x.size()
        device = x.device
        layer_input = x

        for layer in self.cells:
            h, c = layer.init_state(B, (H, W), device)
            outputs = []
            for t in range(T):
                h, c = layer(layer_input[:, t], (h, c))
                outputs.append(h)
            layer_input = torch.stack(outputs, dim=1)  # (B, T, hidden, H, W)

        return layer_input  # final layer outputs across all timesteps


class ConvLSTM_Predictor(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, kernel_size=(3, 3),
                 n_layers=2, predict_all=False):
        super().__init__()
        self.encoder = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            n_layers=n_layers,
            batch_first=True
        )
        self.predict_all = predict_all

        # stronger decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 3, 1),
            nn.Sigmoid()  # output normalized to [0,1]
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        outputs = self.encoder(x)  # (B, T, hidden, H, W)
        if self.predict_all:
            preds = [self.decoder(outputs[:, t]) for t in range(outputs.size(1))]
            return torch.stack(preds, dim=1)  # (B, T, 3, H, W)
        else:
            last_frame = outputs[:, -1]  # (B, hidden, H, W)
            return self.decoder(last_frame)
