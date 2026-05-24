import torch
import torch.nn as nn
import numpy as np
import librosa
import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <model.pt>")
    sys.exit(1)

FREQ_MIN = 20.0
FREQ_MAX = 20000.0
GRU_HIDDEN = 32

CUTOFF_FREQS = [20, 60, 100, 125, 250, 500, 800, 1000, 2000, 4000, 8000, 12000, 16000, 20000]

MODEL_PATH  = sys.argv[1]
DRY_PATH    = "audio/bench_mono.wav"
WET_DIR     = "audio/filteredOutput/bench"
WET_PATTERN = "bench_mono_{freq}hz.wav"


def normalize_knob(freq_hz):
    return (np.log(freq_hz) - np.log(FREQ_MIN)) / (np.log(FREQ_MAX) - np.log(FREQ_MIN))


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = CausalConv1d(in_channels=1, out_channels=16, kernel_size=31)
        self.norm = nn.LayerNorm(16)
        self.knob_to_h0 = nn.Sequential(nn.Linear(1, GRU_HIDDEN), nn.Tanh())
        self.gru = nn.GRU(17, GRU_HIDDEN, batch_first=True)
        self.dense = nn.Linear(GRU_HIDDEN, 1)

    def forward(self, x, h0=None):
        audio = x[:, :, :1]
        knob  = x[:, :, 1:]

        audio = audio.permute(0, 2, 1)
        audio = self.conv(audio)
        audio = audio.permute(0, 2, 1)
        audio = self.norm(audio)

        gru_input = torch.cat([audio, knob], dim=-1)

        if h0 is None:
            h0 = self.knob_to_h0(knob[:, 0, :]).unsqueeze(0)
        out, _ = self.gru(gru_input, h0)
        return self.dense(out)


window_size = 8192
warmup_size = 2048


def run_model_on_audio(model, dry, knob_val, device):
    """Process full audio through the model with stateful GRU inference."""
    n = len(dry)
    knob_channel = np.ones(n, dtype=np.float32) * knob_val
    x = np.stack([dry, knob_channel], axis=-1)  # (n, 2)

    output = np.zeros(n, dtype=np.float32)

    # Seed h0 once from the knob, then carry state across all windows
    knob_t = torch.tensor([[[knob_val]]], dtype=torch.float32).to(device)
    h = model.knob_to_h0(knob_t[:, 0, :]).unsqueeze(0)

    # First window: use warmup to let state settle, don't write output
    warmup_chunk = x[:warmup_size]
    xb = torch.tensor(warmup_chunk, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        audio = model.norm(model.conv(xb[:, :, :1].permute(0, 2, 1)).permute(0, 2, 1))
        gru_in = torch.cat([audio, xb[:, :, 1:]], dim=-1)
        _, h = model.gru(gru_in, h)

    # Process the rest in window_size blocks, carrying h forward
    i = warmup_size
    while i < n:
        chunk = x[i : i + window_size]
        if len(chunk) == 0:
            break
        xb = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            audio = xb[:, :, :1].permute(0, 2, 1)
            audio = model.conv(audio).permute(0, 2, 1)
            audio = model.norm(audio)
            knob_in = xb[:, :, 1:]
            gru_in = torch.cat([audio, knob_in], dim=-1)
            gru_out, h = model.gru(gru_in, h)
            pred = model.dense(gru_out)
        output[i : i + len(chunk)] = pred[0, :, 0].cpu().numpy()
        i += window_size

    return output


def esr(pred, target):
    error  = np.mean((pred - target) ** 2)
    energy = np.mean(target ** 2)
    if energy < 1e-8:
        return float('nan')
    return error / energy


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = Model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    dry, sr = librosa.load(DRY_PATH, sr=None, mono=True)

    print(f"\n{'Freq (Hz)':>10}  {'ESR':>8}  {'ESR (dB)':>10}  {'Status'}")
    print("-" * 45)

    for freq in CUTOFF_FREQS:
        wet_path = f"{WET_DIR}/{WET_PATTERN.format(freq=freq)}"
        wet, _ = librosa.load(wet_path, sr=sr, mono=True)

        n = min(len(dry), len(wet))
        dry_  = dry[:n]
        wet_  = wet[:n]

        knob = normalize_knob(freq)
        pred = run_model_on_audio(model, dry_, knob, device)

        esr_val = esr(pred[warmup_size:], wet_[warmup_size:])
        esr_db  = 10 * np.log10(esr_val) if esr_val > 0 else float('-inf')

        status = "good" if esr_db < -20 else ("ok" if esr_db < -10 else "poor")
        print(f"{freq:>10}  {esr_val:>8.4f}  {esr_db:>9.1f}dB  {status}")

    print()


if __name__ == "__main__":
    main()
