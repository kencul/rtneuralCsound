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
Evaluate a trained model against a dynamic-cutoff reference produced by sweep_ref.

Usage:
  eval_dynamic.py <model.pt> <ref.wav> <ref.csv> [gru_hidden] [warmup] [--save <dir>] [--show]

Arguments:
  model.pt      Path to the trained model checkpoint (.pt)
  ref.wav       Reference WAV produced by sweep_ref
  ref.csv       Companion knob schedule CSV produced by sweep_ref (normalized 0-1 values)
  gru_hidden    GRU hidden size matching the checkpoint (default: 128)
  warmup        Warmup samples fed to the GRU before scoring (default: 256)

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

if len(_args) < 4:
    print(HELP)
    sys.exit(1)

MODEL_PATH = _args[1]
REF_WAV    = _args[2]
REF_CSV    = _args[3]
GRU_HIDDEN = int(_args[4]) if len(_args) > 4 else 128
WARMUP     = int(_args[5]) if len(_args) > 5 else 256
DRY_PATH   = "audio/bench_mono.wav"
WINDOW     = 8192


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
        self.conv  = CausalConv1d(1, 16, 31)
        self.gru   = nn.GRU(17, GRU_HIDDEN, batch_first=True)
        self.dense = nn.Linear(GRU_HIDDEN, 1)

    def forward(self, x, h=None):
        audio = x[:, :, :1]
        knob  = x[:, :, 1:]
        conv_out = audio.permute(0, 2, 1)
        conv_out = self.conv(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        gru_in   = torch.cat([conv_out, knob], dim=-1)
        out, h   = self.gru(gru_in, h)
        return self.dense(out) + audio, h


def load_knob_schedule(csv_path):
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    return data[:, 1].astype(np.float32)


def run_model(model, dry, knob, device):
    n   = min(len(dry), len(knob))
    x   = np.stack([dry[:n], knob[:n]], axis=-1)
    out = np.zeros(n, dtype=np.float32)

    # Warmup: settle GRU from zero state
    xb = torch.tensor(x[:WARMUP], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        _, h = model(xb)

    i = WARMUP
    while i < n:
        chunk = x[i : i + WINDOW]
        xb = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred, h = model(xb, h)
        out[i : i + len(chunk)] = pred[0, :, 0].cpu().numpy()
        i += WINDOW

    return out


def esr_db(pred, target):
    energy = np.mean(target ** 2)
    if energy < 1e-8:
        return float('nan')
    return 10 * np.log10(np.mean((pred - target) ** 2) / energy)


def windowed_esr(pred, target, sr, window_sec=0.5):
    ws = int(window_sec * sr)
    times, vals = [], []
    for i in range(0, len(pred) - ws + 1, ws):
        times.append((i + ws / 2) / sr)
        vals.append(esr_db(pred[i : i + ws], target[i : i + ws]))
    return np.array(times), np.array(vals)


def confirm_overwrite(save_dir, filenames):
    existing = [f for f in filenames if os.path.exists(os.path.join(save_dir, f))]
    if not existing or FORCE:
        return True
    resp = input(f"Overwrite {', '.join(existing)} in {save_dir}? [y/N] ").strip().lower()
    return resp == 'y'


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  GRU: {GRU_HIDDEN} units  |  Warmup: {WARMUP}")

    model = Model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    dry,  sr = librosa.load(DRY_PATH, sr=None, mono=True)
    ref,   _ = librosa.load(REF_WAV,  sr=sr,   mono=True)
    knob     = load_knob_schedule(REF_CSV)

    pred = run_model(model, dry, knob, device)

    n    = min(len(pred), len(ref))
    pred = pred[WARMUP:n]
    ref  = ref[WARMUP:n]

    overall = esr_db(pred, ref)
    print(f"Overall ESR: {overall:.1f} dB")

    times, esr_vals = windowed_esr(pred, ref, sr)

    n_fft, hop = 2048, 512
    db_ref  = librosa.amplitude_to_db(np.abs(librosa.stft(ref,  n_fft=n_fft, hop_length=hop)), ref=np.max)
    db_pred = librosa.amplitude_to_db(np.abs(librosa.stft(pred, n_fft=n_fft, hop_length=hop)), ref=np.max)
    db_diff = np.abs(db_ref - db_pred)

    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    kw = dict(sr=sr, hop_length=hop, x_axis='time', y_axis='log')

    im0 = librosa.display.specshow(db_ref,  ax=axes[0], **kw)
    axes[0].set_title('Reference (RK Moog)')
    fig.colorbar(im0, ax=axes[0], format='%+2.0f dB')

    im1 = librosa.display.specshow(db_pred, ax=axes[1], **kw)
    axes[1].set_title('Model Output')
    fig.colorbar(im1, ax=axes[1], format='%+2.0f dB')

    im = librosa.display.specshow(db_diff, ax=axes[2], **kw, cmap='magma')
    axes[2].set_title('Difference (absolute)')
    fig.colorbar(im, ax=axes[2], format='%+2.0f dB')

    axes[3].plot(times, esr_vals, linewidth=0.8)
    axes[3].axhline(-20, color='green', linestyle='--', alpha=0.7, label='-20 dB (ok)')
    axes[3].axhline(-40, color='blue',  linestyle='--', alpha=0.7, label='-40 dB (good)')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('ESR (dB, lower = better)')
    axes[3].set_title(f'Windowed ESR (0.5 s windows) — overall {overall:.1f} dB')
    axes[3].legend()
    axes[3].invert_yaxis()

    ref_stem   = os.path.splitext(os.path.basename(REF_WAV))[0]
    model_stem = os.path.splitext(os.path.basename(MODEL_PATH))[0]
    # Strip the common "bench_mono_" prefix so output names stay short
    sweep_name = ref_stem.replace('bench_mono_', '', 1)
    plt.suptitle(f'{ref_stem}  |  {model_stem}', fontsize=11)
    plt.tight_layout()

    if SAVE_DIR:
        txt_name = f'evalOutput_{sweep_name}.txt'
        png_name = f'evalOutput_{sweep_name}.png'
        if confirm_overwrite(SAVE_DIR, (txt_name, png_name)):
            os.makedirs(SAVE_DIR, exist_ok=True)
            txt_path = os.path.join(SAVE_DIR, txt_name)
            png_path = os.path.join(SAVE_DIR, png_name)
            with open(txt_path, 'w') as f:
                f.write(f"Ref: {os.path.basename(REF_WAV)}\n")
                f.write(f"Overall ESR: {overall:.1f} dB\n")
            fig.savefig(png_path, dpi=150)
            print(f"Saved {txt_path}\nSaved {png_path}")

    if SHOW_PLOT:
        plt.show()


if __name__ == "__main__":
    main()
