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
    # x: (batch, time, 2) — audio + knob (log-normalized 0-1)
    # returns (output, h) so eval scripts can carry hidden state across chunks
    def __init__(self, gru_hidden=128):
        super().__init__()
        self.conv  = CausalConv1d(1, 16, 31)
        self.gru   = nn.GRU(17, gru_hidden, batch_first=True)  # 16 conv channels + 1 knob
        self.dense = nn.Linear(gru_hidden, 1)

    def forward(self, x, h=None):
        audio    = x[:, :, :1]
        knob     = x[:, :, 1:]
        conv_out = self.conv(audio.permute(0, 2, 1)).permute(0, 2, 1)
        out, h   = self.gru(torch.cat([conv_out, knob], dim=-1), h)
        return self.dense(out) + audio, h
