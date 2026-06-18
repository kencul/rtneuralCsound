import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sys
import os

HELP = """
Evaluate a trained model against static Moog reference files at a grid of cutoff frequencies.

Usage:
  eval_param_model.py <model.pt> [warmup] [--dry <path>] [--save <dir>] [--show] [--force]

Arguments:
  model.pt    Path to trained model checkpoint (.pt)
  warmup      Warmup samples fed to GRU before scoring (default: 256)

Flags:
  --dry <path>  Dry audio file (default: audio/bench_mono.wav). Wet dir and pattern
                are derived from the stem (e.g. audio/ruin_mono.wav -> filteredOutput/ruin/)
  --save <dir>  Write evalOutput[_stem].txt and evalOutput[_stem].png to <dir>
  --force       Overwrite existing files without prompting
  --show        Open the interactive plot window
  --help, -h    Show this message
""".strip()

SHOW_PLOT = '--show' in sys.argv
FORCE     = '--force' in sys.argv
SAVE_DIR  = None
if '--save' in sys.argv:
    idx = sys.argv.index('--save')
    if idx + 1 >= len(sys.argv):
        print("Error: --save requires a directory argument")
        sys.exit(1)
    SAVE_DIR = sys.argv[idx + 1]
    _args = [a for i, a in enumerate(sys.argv) if i not in (idx, idx + 1)]
else:
    _args = list(sys.argv)

DRY_PATH = "audio/bench_mono.wav"
if '--dry' in _args:
    idx = _args.index('--dry')
    if idx + 1 >= len(_args):
        print("Error: --dry requires a path argument")
        sys.exit(1)
    DRY_PATH = _args[idx + 1]
    _args = [a for i, a in enumerate(_args) if i not in (idx, idx + 1)]

_args = [a for a in _args if a not in ('--show', '--force', '--help', '-h')]

if '--help' in sys.argv or '-h' in sys.argv:
    print(HELP)
    sys.exit(0)

if len(_args) < 2:
    print(HELP)
    sys.exit(1)

MODEL_PATH  = _args[1]
warmup_size = int(_args[2]) if len(_args) > 2 else 256

dry_stem    = os.path.splitext(os.path.basename(DRY_PATH))[0]  # e.g. "ruin_mono"
audio_dir   = os.path.dirname(DRY_PATH)
wet_subdir  = dry_stem.replace('_mono', '')                    # e.g. "ruin"
WET_DIR     = os.path.join(audio_dir, "filteredOutput", wet_subdir)
WET_PATTERN = dry_stem + "_{freq}hz.wav"
ALL_CUTOFF_FREQS = [20, 60, 100, 125, 250, 500, 800, 1000, 2000, 4000, 8000, 12000, 16000, 20000]
window_size      = 8192


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    if 'model_state' in ckpt:
        return (ckpt['model_state'], ckpt['arch'], ckpt['gru_hidden'],
                ckpt.get('freq_min', 100.0), ckpt.get('freq_max', 20000.0))
    # legacy checkpoint: infer arch and hidden size from GRU weight shape
    ih         = ckpt['gru.weight_ih_l0']  # shape (3*hidden, input_size)
    gru_hidden = ih.shape[0] // 3
    arch       = 'concat' if ih.shape[1] == 17 else 'film'
    return ckpt, arch, gru_hidden, 100.0, 20000.0


def normalize_knob(freq_hz, freq_min, freq_max):
    return (np.log(freq_hz) - np.log(freq_min)) / (np.log(freq_max) - np.log(freq_min))


def run_model_on_audio(model, dry, knob_val, device):
    n            = len(dry)
    knob_channel = np.ones(n, dtype=np.float32) * knob_val
    x            = np.stack([dry, knob_channel], axis=-1)
    output       = np.zeros(n, dtype=np.float32)

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

    state_dict, arch, gru_hidden, freq_min, freq_max = load_checkpoint(MODEL_PATH, device)
    print(f"arch={arch}  gru_hidden={gru_hidden}  freq_min={freq_min:.0f}  warmup={warmup_size}")

    if arch == 'film':
        from model_film import Model
        # film weight shape [2*16, 1] = pre-GRU; [2*gru_hidden, 1] = post-GRU
        film_pre = state_dict['film.weight'].shape[0] == 32
        model = Model(gru_hidden=gru_hidden, film_pre=film_pre).to(device)
    else:
        from model_concat import Model
        model = Model(gru_hidden=gru_hidden).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    cutoff_freqs = [f for f in ALL_CUTOFF_FREQS if f >= freq_min]
    dry, sr = librosa.load(DRY_PATH, sr=None, mono=True)

    header = f"{'Freq (Hz)':>10}  {'ESR':>8}  {'ESR (dB)':>10}  {'Status'}"
    sep    = "-" * 45
    print(f"\n{header}\n{sep}")
    table_lines = [header, sep]

    results = []
    for freq in cutoff_freqs:
        wet_path = f"{WET_DIR}/{WET_PATTERN.format(freq=freq)}"
        wet, _ = librosa.load(wet_path, sr=sr, mono=True)

        n       = min(len(dry), len(wet))
        knob    = normalize_knob(freq, freq_min, freq_max)
        pred    = run_model_on_audio(model, dry[:n], knob, device)
        esr_val = esr(pred[warmup_size:], wet[:n][warmup_size:])
        esr_db  = 10 * np.log10(esr_val) if esr_val > 0 else float('-inf')
        results.append((freq, pred[:n], wet[:n], esr_db))

        status = "good" if esr_db < -20 else ("ok" if esr_db < -10 else "poor")
        row = f"{freq:>10}  {esr_val:>8.4f}  {esr_db:>9.1f}dB  {status}"
        print(row)
        table_lines.append(row)

    print()

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
        suffix   = f'_{dry_stem}' if dry_stem != 'bench_mono' else ''
        txt_path = os.path.join(SAVE_DIR, f'evalOutput{suffix}.txt')
        png_path = os.path.join(SAVE_DIR, f'evalOutput{suffix}.png')
        with open(txt_path, 'w') as f:
            f.write('\n'.join(table_lines) + '\n')
        fig.savefig(png_path, dpi=150)
        print(f"Saved {txt_path}\nSaved {png_path}")

    if SHOW_PLOT:
        plt.show()


if __name__ == "__main__":
    main()
