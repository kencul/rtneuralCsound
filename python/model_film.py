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
    # film_pre=True: FiLM conditions conv features before the GRU so it can
    # learn different hidden state trajectories per cutoff (current approach).
    # film_pre=False: post-GRU placement — kept for loading run 21 (failed).
    def __init__(self, gru_hidden=64, film_pre=True):
        super().__init__()
        self._film_pre  = film_pre
        film_channels   = 16 if film_pre else gru_hidden
        self.conv  = CausalConv1d(1, 16, 31)
        self.film  = nn.Linear(1, 2 * film_channels)  # knob → [gamma, beta]
        self.gru   = nn.GRU(16, gru_hidden, batch_first=True)
        self.dense = nn.Linear(gru_hidden, 1)
        # identity init: gamma=1, beta=0 so FiLM starts as a pass-through
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.film.bias.data[:film_channels] = 1.0

    def forward(self, x, h=None):
        audio    = x[:, :, :1]
        knob     = x[:, :, 1:]
        conv_out = self.conv(audio.permute(0, 2, 1)).permute(0, 2, 1)
        if self._film_pre:
            gamma, beta = self.film(knob).chunk(2, dim=-1)
            out, h = self.gru(gamma * conv_out + beta, h)
        else:
            out, h = self.gru(conv_out, h)
            gamma, beta = self.film(knob).chunk(2, dim=-1)
            out = gamma * out + beta
        return self.dense(out) + audio, h
