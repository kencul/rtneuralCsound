"""
Evaluate a distortion model against a standard set of dry/wet pairs.

Runs against held-out bench pairs and training giga pairs, prints an ESR
table, and optionally saves output WAVs and spectrogram plots.

Usage:
  eval_distortion.py <model.pt> [--save <dir>] [--show] [--force]

Flags:
  --save <dir>  Save output WAVs, per-pair spectrograms, and summary.txt to <dir>
  --show        Open interactive plot windows
  --force       Overwrite existing files without prompting
  --help, -h    Show this message
"""

import sys
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import torch

HELP = __doc__.strip()

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
SR          = 48000
WINDOW_SIZE = 8192
WARMUP_SIZE = 256

# (label, dry, wet, split)
EVAL_PAIRS = [
    ("bench",        "audio/updatedDistortion/benchDry.wav",        "audio/updatedDistortion/benchWet.wav",        "held-out"),
    ("bench-10dB",   "audio/updatedDistortion/bench-10dBDry.wav",   "audio/updatedDistortion/bench-10dBWet.wav",   "held-out"),
    ("training",     "audio/updatedDistortion/trainingDry.wav",     "audio/updatedDistortion/trainingWet.wav",     "train"),
    ("training-10dB","audio/updatedDistortion/training-10dBDry.wav","audio/updatedDistortion/training-10dBWet.wav","train"),
]


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
    # Remap legacy key names (gru.* / lstm.*) to current rnn.*
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
    conv_k = model.conv.conv.kernel_size[0] - 1  # samples needed for conv context

    # Warmup: initialise GRU hidden state and prime the conv context buffer.
    warmup = x[:WARMUP_SIZE]
    xb = torch.tensor(warmup, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        _, h = model(xb)
    conv_buf = warmup[-conv_k:]  # last conv_k samples carried into next chunk

    i = WARMUP_SIZE
    while i < n:
        chunk = x[i : i + WINDOW_SIZE]
        if len(chunk) == 0:
            break

        # Prepend conv context so the conv sees correct history at chunk start.
        padded = np.concatenate([conv_buf, chunk], axis=0)
        xb = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred, h = model(xb, h)

        # Discard the conv_k output samples that correspond to the context prefix.
        out = pred[0, conv_k:, 0].cpu().numpy()
        output[i : i + len(out)] = out
        conv_buf = chunk[-conv_k:]
        i += WINDOW_SIZE

    return output


def esr_db(pred, ref):
    pred, ref = pred[WARMUP_SIZE:], ref[WARMUP_SIZE:]
    energy = np.mean(ref ** 2)
    esr    = np.mean((pred - ref) ** 2) / max(energy, 1e-8)
    return 10 * np.log10(esr)


def save_spectrogram(label, ref, pred, sr, out_path):
    n_fft, hop = 2048, 512
    kw = dict(sr=sr, hop_length=hop, x_axis='time', y_axis='log')

    db_ref  = librosa.amplitude_to_db(np.abs(librosa.stft(ref,  n_fft=n_fft, hop_length=hop)), ref=np.max)
    db_pred = librosa.amplitude_to_db(np.abs(librosa.stft(pred, n_fft=n_fft, hop_length=hop)), ref=np.max)
    db_diff = np.abs(db_ref - db_pred)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    im0 = librosa.display.specshow(db_ref,  ax=axes[0], **kw)
    axes[0].set_title(f'{label} — reference')
    fig.colorbar(im0, ax=axes[0], format='%+2.0f dB')

    im1 = librosa.display.specshow(db_pred, ax=axes[1], **kw)
    axes[1].set_title(f'{label} — model output')
    fig.colorbar(im1, ax=axes[1], format='%+2.0f dB')

    im2 = librosa.display.specshow(db_diff, ax=axes[2], **kw, cmap='magma')
    axes[2].set_title(f'{label} — difference')
    fig.colorbar(im2, ax=axes[2], format='%+2.0f dB')

    plt.tight_layout()
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
    model, gru_hidden, cell = load_model(MODEL_PATH, device)
    model_stem = os.path.basename(os.path.dirname(MODEL_PATH))
    print(f"Model: {MODEL_PATH}  ({cell.upper()} {gru_hidden}u)")
    print(f"Device: {device}\n")

    if SAVE_DIR:
        os.makedirs(SAVE_DIR, exist_ok=True)

    results = []

    for label, dry_path, wet_path, split in EVAL_PAIRS:
        print(f"[{label}] ({split})")
        dry, _   = librosa.load(dry_path, sr=SR, mono=True)
        ref, _   = librosa.load(wet_path, sr=SR, mono=True)
        n        = min(len(dry), len(ref))
        dry, ref = dry[:n], ref[:n]

        pred  = run_model(model, dry, device)
        score = esr_db(pred, ref)
        print(f"  ESR: {score:.1f} dB")
        results.append((label, split, score))

        if SAVE_DIR:
            wav_path = os.path.join(SAVE_DIR, f"{label}_output.wav")
            png_path = os.path.join(SAVE_DIR, f"{label}_spectrogram.png")
            if confirm_overwrite(wav_path):
                sf.write(wav_path, pred, SR, subtype='FLOAT')
                print(f"  Saved {wav_path}")
            save_spectrogram(label, ref[WARMUP_SIZE:], pred[WARMUP_SIZE:], SR, png_path)
        elif SHOW:
            save_spectrogram(label, ref[WARMUP_SIZE:], pred[WARMUP_SIZE:], SR, os.devnull)

    print(f"\n{'Pair':<14} {'Split':<10} {'ESR (dB)':>10}")
    print("-" * 36)
    for label, split, score in results:
        print(f"{label:<14} {split:<10} {score:>10.1f}")

    if SAVE_DIR:
        summary_path = os.path.join(SAVE_DIR, "summary.txt")
        if confirm_overwrite(summary_path):
            with open(summary_path, 'w') as f:
                f.write(f"model: {MODEL_PATH}\n")
                f.write(f"cell:  {cell.upper()} {gru_hidden}u\n\n")
                f.write(f"{'Pair':<14} {'Split':<10} {'ESR (dB)':>10}\n")
                f.write("-" * 36 + "\n")
                for label, split, score in results:
                    f.write(f"{label:<14} {split:<10} {score:>10.1f}\n")
            print(f"\nSaved {summary_path}")


if __name__ == "__main__":
    main()
