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
    #
    # Knob is NOT fed into the GRU. Instead, post-GRU FiLM applies a per-sample
    # scale and shift derived from the knob, so conditioning is instantaneous
    # rather than requiring the GRU to integrate the knob value over time.
    def __init__(self, gru_hidden=64):
        super().__init__()
        self.conv  = CausalConv1d(1, 16, 31)
        self.gru   = nn.GRU(16, gru_hidden, batch_first=True)
        self.film  = nn.Linear(1, 2 * gru_hidden)  # knob → [gamma, beta]
        self.dense = nn.Linear(gru_hidden, 1)
        # identity init: gamma=1, beta=0 so FiLM starts as a pass-through
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.film.bias.data[:gru_hidden] = 1.0

    def forward(self, x, h=None):
        audio    = x[:, :, :1]
        knob     = x[:, :, 1:]
        conv_out = self.conv(audio.permute(0, 2, 1)).permute(0, 2, 1)
        out, h   = self.gru(conv_out, h)
        gamma, beta = self.film(knob).chunk(2, dim=-1)
        out = gamma * out + beta
        return self.dense(out) + audio, h
