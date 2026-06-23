import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)


class Model(nn.Module):
    # Non-parametric: audio-only input, no knob channel.
    # x: (batch, time, 1)
    # returns (output, h) so eval can carry hidden state across chunks
    def __init__(self, gru_hidden=128, cell='gru'):
        super().__init__()
        self.conv = CausalConv1d(1, 16, 31)
        if cell == 'lstm':
            self.rnn = nn.LSTM(16, gru_hidden, batch_first=True)
        else:
            self.rnn = nn.GRU(16, gru_hidden, batch_first=True)
        self.dense = nn.Linear(gru_hidden, 1)

    def forward(self, x, h=None):
        conv_out = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        out, h   = self.rnn(conv_out, h)
        return self.dense(out) + x, h
