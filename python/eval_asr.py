"""
Measure Aliasing-to-Signal Ratio (ASR) of a distortion model.

Per Sato & Smith (DAFx 2025). Feeds clean sine inputs at multiple
frequencies through the model and measures the ratio of energy in
non-harmonic FFT bins (aliasing) to energy in the input frequency
and its harmonics (signal). Lower (more negative) ASR is better.

Usage:
  eval_asr.py <model.pt> [--save <dir>] [--show] [--force]

Flags:
  --save <dir>  Save spectrum plot and asr_summary.txt to <dir>
  --show        Open interactive plot window
  --force       Overwrite existing files without prompting
  --help, -h    Show this message
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.signal import welch

HELP = __doc__.strip()


class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
    def flush(self):
        for s in self._streams:
            s.flush()


SHOW  = '--show'  in sys.argv
FORCE = '--force' in sys.argv

if '--help' in sys.argv or '-h' in sys.argv:
    print(HELP)
    sys.exit(0)

SAVE_DIR = None
args = list(sys.argv[1:])
if '--save' in args:
    idx = args.index('--save')
    SAVE_DIR = args[idx + 1]
    del args[idx:idx + 2]

args = [a for a in args if a not in ('--show', '--force')]

if len(args) < 1:
    print(HELP)
    sys.exit(1)

MODEL_PATH  = args[0]

SR            = 48000
WINDOW_SIZE   = 8192
WARMUP_SIZE   = 256

TEST_FREQS    = [500, 1000, 2000, 4000, 8000]   # Hz
TEST_AMP      = 0.5
TEST_DURATION = 4.0      # seconds
DISCARD_HEAD  = 0.5      # seconds dropped for warmup + transient

FFT_SIZE      = 32768
HARM_TOL_BINS = 3        # half-width of harmonic window in FFT bins
LOW_CUT_HZ    = 30.0     # ignore DC and sub-bass below this


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if 'model_state' in ckpt:
        state  = ckpt['model_state']
        hidden = ckpt['gru_hidden']
        arch   = ckpt.get('arch', 'distortion_gru')
    else:
        state  = ckpt
        hidden = ckpt['rnn.weight_ih_l0'].shape[0] // 3
        arch   = 'distortion_gru'
    cell = 'lstm' if 'lstm' in arch else 'gru'
    state = {k.replace('gru.', 'rnn.').replace('lstm.', 'rnn.'): v for k, v in state.items()}
    from model_distortion import Model
    model = Model(gru_hidden=hidden, cell=cell).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, hidden, cell


def run_model(model, dry, device):
    x      = dry.reshape(-1, 1)
    n      = len(x)
    output = np.zeros(n, dtype=np.float32)
    conv_k = model.conv.conv.kernel_size[0] - 1

    warmup = x[:WARMUP_SIZE]
    xb = torch.tensor(warmup, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        _, h = model(xb)
    conv_buf = warmup[-conv_k:]

    i = WARMUP_SIZE
    while i < n:
        chunk = x[i : i + WINDOW_SIZE]
        if len(chunk) == 0:
            break
        padded = np.concatenate([conv_buf, chunk], axis=0)
        xb = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred, h = model(xb, h)
        out = pred[0, conv_k:, 0].cpu().numpy()
        output[i : i + len(out)] = out
        conv_buf = chunk[-conv_k:]
        i += WINDOW_SIZE

    return output


def generate_sine(f0, amp, duration, sr):
    t = np.arange(int(duration * sr)) / sr
    return (amp * np.sin(2 * np.pi * f0 * t)).astype(np.float32)


def compute_spectrum(y, sr, fft_size):
    freqs, psd = welch(y, fs=sr, nperseg=fft_size, noverlap=fft_size // 2,
                       window='hann', scaling='density')
    return freqs, psd


def asr_db(freqs, psd, f0, sr, harm_tol_bins, low_cut_hz):
    bin_hz = freqs[1] - freqs[0]
    nyq    = sr / 2
    harm_mask = np.zeros_like(freqs, dtype=bool)
    k = 1
    while k * f0 < nyq:
        center = int(round(k * f0 / bin_hz))
        if center >= len(freqs):
            break
        lo = max(0, center - harm_tol_bins)
        hi = min(len(freqs), center + harm_tol_bins + 1)
        harm_mask[lo:hi] = True
        k += 1
    low_mask   = freqs < low_cut_hz
    alias_mask = ~harm_mask & ~low_mask
    e_harm  = np.sum(psd[harm_mask])
    e_alias = np.sum(psd[alias_mask])
    return 10 * np.log10(e_alias / (e_harm + 1e-20))


def plot_spectra(specs, asrs, model_label, out_path):
    n = len(specs)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (f0, freqs, psd), asr in zip(axes, specs, asrs):
        psd_db = 10 * np.log10(psd + 1e-20)
        ax.plot(freqs, psd_db, color='steelblue', lw=0.6)
        k = 1
        nyq = SR / 2
        while k * f0 < nyq:
            ax.axvline(k * f0, color='orange', alpha=0.4, lw=0.8)
            k += 1
        ax.set_xlim(0, nyq)
        ax.set_ylim(np.max(psd_db) - 100, np.max(psd_db) + 5)
        ax.set_ylabel('PSD (dB)')
        ax.set_title(f'f0 = {f0} Hz   ASR = {asr:.1f} dB   (orange = harmonics)')
    axes[-1].set_xlabel('Frequency (Hz)')
    fig.suptitle(model_label)
    plt.tight_layout()
    if out_path and out_path != os.devnull:
        fig.savefig(out_path, dpi=150)
        print(f"  Saved {out_path}")
    if SHOW:
        plt.show()
    plt.close(fig)


def confirm_overwrite(path):
    if not os.path.exists(path) or FORCE:
        return True
    resp = input(f"Overwrite {path}? [y/N] ").strip().lower()
    return resp == 'y'


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_file = None
    if SAVE_DIR:
        os.makedirs(SAVE_DIR, exist_ok=True)
        log_path = os.path.join(SAVE_DIR, "asr_log.txt")
        log_file = open(log_path, 'w')
        sys.stdout = _Tee(sys.__stdout__, log_file)

    model, gru_hidden, cell = load_model(MODEL_PATH, device)
    print(f"Model: {MODEL_PATH}  ({cell.upper()} {gru_hidden}u)")
    print(f"Device: {device}\n")

    discard_n = int(DISCARD_HEAD * SR)
    specs   = []
    results = []

    for f0 in TEST_FREQS:
        print(f"[{f0} Hz]")
        x = generate_sine(f0, TEST_AMP, TEST_DURATION, SR)
        y = run_model(model, x, device)
        y_ana = y[discard_n:]
        freqs, psd = compute_spectrum(y_ana, SR, FFT_SIZE)
        asr = asr_db(freqs, psd, f0, SR, HARM_TOL_BINS, LOW_CUT_HZ)
        print(f"  ASR: {asr:.1f} dB")
        specs.append((f0, freqs, psd))
        results.append((f0, asr))

    mean_asr = float(np.mean([a for _, a in results]))
    print(f"\n{'Test freq (Hz)':<16} {'ASR (dB)':>10}")
    print("-" * 28)
    for f0, asr in results:
        print(f"{f0:<16} {asr:>10.1f}")
    print("-" * 28)
    print(f"{'mean':<16} {mean_asr:>10.1f}")

    model_label = f"{MODEL_PATH} ({cell.upper()} {gru_hidden}u)"

    if SAVE_DIR:
        png_path = os.path.join(SAVE_DIR, "asr_spectra.png")
        if confirm_overwrite(png_path):
            plot_spectra(specs, [a for _, a in results], model_label, png_path)
        summary_path = os.path.join(SAVE_DIR, "asr_summary.txt")
        if confirm_overwrite(summary_path):
            with open(summary_path, 'w') as f:
                f.write(f"model: {MODEL_PATH}\n")
                f.write(f"cell:  {cell.upper()} {gru_hidden}u\n")
                f.write(f"test:  sine amp={TEST_AMP} duration={TEST_DURATION}s sr={SR}\n")
                f.write(f"fft:   nperseg={FFT_SIZE} harm_tol={HARM_TOL_BINS} bins low_cut={LOW_CUT_HZ} Hz\n\n")
                f.write(f"{'Test freq (Hz)':<16} {'ASR (dB)':>10}\n")
                f.write("-" * 28 + "\n")
                for f0, asr in results:
                    f.write(f"{f0:<16} {asr:>10.1f}\n")
                f.write("-" * 28 + "\n")
                f.write(f"{'mean':<16} {mean_asr:>10.1f}\n")
            print(f"Saved {summary_path}")
    elif SHOW:
        plot_spectra(specs, [a for _, a in results], model_label, os.devnull)

    if log_file is not None:
        sys.stdout = sys.__stdout__
        log_file.close()


if __name__ == "__main__":
    main()
