'''
    This file builds up the ConvLSTM.
'''

import torch
import torch.nn as nn

# define the ConvLSTM cell
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2 # how much padding to add on height, width
        self.bias = bias # bool

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
    
    # update the cell state and hidden state using four gates
    def forward(self, input, state_t):
        hidden_t, cell_t = state_t

        xh = torch.cat([input, hidden_t], dim=1) # concatenate the input and the hidden state along channel

        gates = self.conv(xh) # jointly compute all gate pre activation

        i_gate, f_gate, o_gate, g_gate = torch.split(gates, self.hidden_dim, dim=1) # split to four gates (B, hidden_dim, H, W)

        # apply nonlinearities
        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        o = torch.sigmoid(o_gate)
        g = torch.sigmoid(g_gate)

        cell_t_ = f * cell_t + i * g # update cell state
        hidden_t_ = o * torch.tanh(cell_t_) # update hidden state

        return cell_t_, hidden_t_
    
    # initialize the states
    def initialize(self, batch_size, image_size):
        height, width = image_size[0], image_size[1]
        
        hidden_0 = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
        cell_0 = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)

        return hidden_0, cell_0

# define the multi layer LSTM
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, n_layers, batch_first=False, bias=True, return_all_layers=False):
        super().__init__()

        # extend the hidden_dim and kernel size to multiple layers
        if not isinstance(hidden_dim, list):
            hidden_dim = [hidden_dim] * n_layers
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * n_layers

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # track all the cell state and their parameters
        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim_t = self.input_dim
            else:
                input_dim_t = self.hidden_dim[i-1]

            cells.append(ConvLSTMCell(
                input_dim= input_dim_t,
                hidden_dim= self.hidden_dim[i],
                kernel_size=self.kernel_size[i],
                bias=self.bias
            ))

        self.cells = nn.ModuleList(cells)
        
    def forward(self, input, hidden_state=None):
        if not self.batch_first:
            input = input.permute(1, 0, 2, 3, 4) # (t, b, c, h, w) -> (b, t, c, h, w)
        b, t, c, h, w = input.size()
        hidden_state = self.initialize(batch_size=b, image_size=(h,w)) # initialize the hidden states
        layer_outputs, last_states = [], [] # sequence of hidden output / final hidden and cell state after processing
        cur_layer_input = input

        # loop through each layer
        for layer in range(self.n_layers):
            hidden, cell = hidden_state[layer]
            hidden_output = [] # store hidden output at each time step

            # loop over time
            for i in range(t):
                hidden, cell = self.cells[layer](
                    input = cur_layer_input[:, i, : ,:, :],
                    cur_state = [hidden, cell]
                )
                hidden_output.append(hidden)

            layer_output = torch.stack(hidden_output, dim=1) # stack hidden output along time
            cur_layer_input = layer_output # pass output to next layer

            layer_outputs.append(layer_outputs)
            last_states.append([hidden, cell])

            # return only final layer if specified
            if not self.return_all_layers:
                layer_outputs = layer_outputs[-1:]
                last_states = last_states[-1:]


        return layer_outputs, last_states

    def initialize(self, batch_size, image_size):
        states = []
        for i in range(self.n_layers):
            states.append(self.cells[i].init_hidden(batch_size, image_size))
        return states

