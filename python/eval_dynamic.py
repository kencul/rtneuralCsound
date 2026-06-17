import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sys
import os

HELP = """
Evaluate a trained model against a dynamic-cutoff reference produced by sweep_ref.

Usage:
  eval_dynamic.py <model.pt> <ref.wav> <ref.csv> [warmup] [--save <dir>] [--show]

Arguments:
  model.pt      Path to the trained model checkpoint (.pt)
  ref.wav       Reference WAV produced by sweep_ref
  ref.csv       Companion knob schedule CSV produced by sweep_ref (normalized 0-1 values)
  warmup        Warmup samples fed to the GRU before scoring (default: 256)

Flags:
  --save <dir>  Write evalOutput.txt and evalOutput.png to <dir>
  --force       Overwrite existing output files without prompting
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
WARMUP     = int(_args[4]) if len(_args) > 4 else 256
DRY_PATH   = "audio/bench_mono.wav"
WINDOW     = 8192


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    if 'model_state' in ckpt:
        return ckpt['model_state'], ckpt['arch'], ckpt['gru_hidden']
    # legacy checkpoint: infer from GRU weight shape
    ih         = ckpt['gru.weight_ih_l0']  # (3*hidden, input_size)
    gru_hidden = ih.shape[0] // 3
    arch       = 'concat' if ih.shape[1] == 17 else 'film'
    return ckpt, arch, gru_hidden


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

    state_dict, arch, gru_hidden = load_checkpoint(MODEL_PATH, device)
    print(f"Device: {device}  |  arch={arch}  gru_hidden={gru_hidden}  warmup={WARMUP}")

    if arch == 'film':
        from model_film import Model
    else:
        from model_concat import Model

    model = Model(gru_hidden=gru_hidden).to(device)
    model.load_state_dict(state_dict)
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
