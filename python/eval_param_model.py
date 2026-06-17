import torch
import torch.nn as nn
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sys
import os

# Parse --save <dir> before handling positional args
HELP = """
Evaluate a trained model against static Moog reference files at a grid of cutoff frequencies.

Usage:
  eval_param_model.py <model.pt> [warmup] [freq_min] [gru_hidden] [--save <dir>] [--show]

Arguments:
  model.pt      Path to the trained model checkpoint (.pt)
  warmup        Warmup samples fed to the GRU before scoring (default: 2048)
  freq_min      Lowest cutoff frequency to evaluate in Hz (default: 20)
  gru_hidden    GRU hidden size matching the checkpoint (default: 128)

Flags:
  --save <dir>  Write evalOutput.txt and evalOutput.png to <dir>
  --force       Overwrite existing output files without prompting
  --show        Open the interactive plot window
  --help, -h    Show this message
""".strip()

SHOW_PLOT = '--show' in sys.argv
FORCE     = '--force' in sys.argv
SAVE_DIR = None
if '--save' in sys.argv:
    idx = sys.argv.index('--save')
    if idx + 1 >= len(sys.argv):
        print("Error: --save requires a directory argument")
        sys.exit(1)
    SAVE_DIR = sys.argv[idx + 1]
    _args = [a for i, a in enumerate(sys.argv) if i not in (idx, idx + 1)]
else:
    _args = list(sys.argv)
_args = [a for a in _args if a not in ('--show', '--force', '--help', '-h')]

if '--help' in sys.argv or '-h' in sys.argv:
    print(HELP)
    sys.exit(0)

if len(_args) < 2:
    print(HELP)
    sys.exit(1)

FREQ_MIN   = float(_args[3]) if len(_args) > 3 else 20.0
FREQ_MAX   = 20000.0
GRU_HIDDEN = int(_args[4])   if len(_args) > 4 else 128

ALL_CUTOFF_FREQS = [20, 60, 100, 125, 250, 500, 800, 1000, 2000, 4000, 8000, 12000, 16000, 20000]
CUTOFF_FREQS = [f for f in ALL_CUTOFF_FREQS if f >= FREQ_MIN]

MODEL_PATH  = _args[1]
DRY_PATH    = "audio/bench_mono.wav"
WET_DIR     = "audio/filteredOutput/bench"
WET_PATTERN = "bench_mono_{freq}hz.wav"

window_size = 8192
warmup_size = int(_args[2]) if len(_args) > 2 else 2048


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
        self.conv  = CausalConv1d(in_channels=1, out_channels=16, kernel_size=31)
        self.gru   = nn.GRU(17, GRU_HIDDEN, batch_first=True)
        self.dense = nn.Linear(GRU_HIDDEN, 1)

    def forward(self, x, h=None):
        audio = x[:, :, :1]
        knob  = x[:, :, 1:]
        conv_out = audio.permute(0, 2, 1)
        conv_out = self.conv(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        gru_input = torch.cat([conv_out, knob], dim=-1)
        out, h_out = self.gru(gru_input, h)
        return self.dense(out) + audio, h_out


def run_model_on_audio(model, dry, knob_val, device):
    n = len(dry)
    knob_channel = np.ones(n, dtype=np.float32) * knob_val
    x = np.stack([dry, knob_channel], axis=-1)

    output = np.zeros(n, dtype=np.float32)

    xb = torch.tensor(x[:warmup_size], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        _, h = model(xb)

    i = warmup_size
    while i < n:
        chunk = x[i : i + window_size]
        if len(chunk) == 0:
            break
        xb = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred, h = model(xb, h)
        output[i : i + len(chunk)] = pred[0, :, 0].cpu().numpy()
        i += window_size

    return output


def esr(pred, target):
    error  = np.mean((pred - target) ** 2)
    energy = np.mean(target ** 2)
    if energy < 1e-8:
        return float('nan')
    return error / energy


def confirm_overwrite(save_dir):
    existing = [f for f in ('evalOutput.txt', 'evalOutput.png')
                if os.path.exists(os.path.join(save_dir, f))]
    if not existing or FORCE:
        return True
    resp = input(f"Overwrite {', '.join(existing)} in {save_dir}? [y/N] ").strip().lower()
    return resp == 'y'


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = Model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    dry, sr = librosa.load(DRY_PATH, sr=None, mono=True)

    header = f"{'Freq (Hz)':>10}  {'ESR':>8}  {'ESR (dB)':>10}  {'Status'}"
    sep    = "-" * 45
    print(f"\n{header}\n{sep}")
    table_lines = [header, sep]

    results = []
    for freq in CUTOFF_FREQS:
        wet_path = f"{WET_DIR}/{WET_PATTERN.format(freq=freq)}"
        wet, _ = librosa.load(wet_path, sr=sr, mono=True)

        n = min(len(dry), len(wet))
        dry_ = dry[:n]
        wet_ = wet[:n]

        knob    = normalize_knob(freq)
        pred    = run_model_on_audio(model, dry_, knob, device)
        esr_val = esr(pred[warmup_size:], wet_[warmup_size:])
        esr_db  = 10 * np.log10(esr_val) if esr_val > 0 else float('-inf')
        results.append((freq, pred[:n], wet_[:n], esr_db))

        status = "good" if esr_db < -20 else ("ok" if esr_db < -10 else "poor")
        row = f"{freq:>10}  {esr_val:>8.4f}  {esr_db:>9.1f}dB  {status}"
        print(row)
        table_lines.append(row)

    print()

    # Spectrograms at the frequency closest to 1kHz
    plot_idx = min(range(len(results)), key=lambda i: abs(results[i][0] - 1000))
    plot_freq, plot_pred, plot_ref, _ = results[plot_idx]
    scored_pred = plot_pred[warmup_size:]
    scored_ref  = plot_ref[warmup_size:]

    n_fft, hop = 2048, 512
    db_ref  = librosa.amplitude_to_db(np.abs(librosa.stft(scored_ref,  n_fft=n_fft, hop_length=hop)), ref=np.max)
    db_pred = librosa.amplitude_to_db(np.abs(librosa.stft(scored_pred, n_fft=n_fft, hop_length=hop)), ref=np.max)
    db_diff = np.abs(db_ref - db_pred)

    freqs   = [r[0] for r in results]
    esr_dbs = [r[3] for r in results]

    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    kw = dict(sr=sr, hop_length=hop, x_axis='time', y_axis='log')

    im0 = librosa.display.specshow(db_ref,  ax=axes[0], **kw)
    axes[0].set_title(f'Reference (RK Moog) — {plot_freq} Hz')
    fig.colorbar(im0, ax=axes[0], format='%+2.0f dB')

    im1 = librosa.display.specshow(db_pred, ax=axes[1], **kw)
    axes[1].set_title(f'Model Output — {plot_freq} Hz')
    fig.colorbar(im1, ax=axes[1], format='%+2.0f dB')

    im2 = librosa.display.specshow(db_diff, ax=axes[2], **kw, cmap='magma')
    axes[2].set_title(f'Difference — {plot_freq} Hz')
    fig.colorbar(im2, ax=axes[2], format='%+2.0f dB')

    axes[3].semilogx(freqs, esr_dbs, 'o-', linewidth=1)
    axes[3].axhline(-20, color='green', linestyle='--', alpha=0.7, label='-20 dB (ok)')
    axes[3].axhline(-40, color='blue',  linestyle='--', alpha=0.7, label='-40 dB (good)')
    axes[3].set_xticks(freqs)
    axes[3].set_xticklabels(freqs, rotation=45, fontsize=8)
    axes[3].set_xlabel('Cutoff Frequency (Hz)')
    axes[3].set_ylabel('ESR (dB, lower = better)')
    axes[3].set_title('ESR by Cutoff Frequency')
    axes[3].legend()
    axes[3].invert_yaxis()

    model_stem = os.path.splitext(os.path.basename(MODEL_PATH))[0]
    plt.suptitle(f'Static Eval — {model_stem}', fontsize=11)
    plt.tight_layout()

    if SAVE_DIR and confirm_overwrite(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)
        txt_path = os.path.join(SAVE_DIR, 'evalOutput.txt')
        png_path = os.path.join(SAVE_DIR, 'evalOutput.png')
        with open(txt_path, 'w') as f:
            f.write('\n'.join(table_lines) + '\n')
        fig.savefig(png_path, dpi=150)
        print(f"Saved {txt_path}\nSaved {png_path}")

    if SHOW_PLOT:
        plt.show()


if __name__ == "__main__":
    main()
