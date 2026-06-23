"""
Measure and correct the latency offset between a dry and wet recording pair.

When audio is played through hardware and recorded back, the ADC/DAC round-trip
introduces a fixed sample delay. This script finds that delay via cross-correlation
and optionally writes a corrected wet file (dry is never modified).

Usage:
  python align_audio.py <dry> <wet> [--sr <rate>] [--save <path>] [--force]

Arguments:
  dry            Dry (unprocessed) audio file
  wet            Wet (hardware-processed) audio file

Flags:
  --sr <rate>    Resample both files to this rate before aligning (e.g. 48000)
  --save <path>  Write the aligned wet file to this path (overwrites if exists + --force)
  --force        Overwrite existing file without prompting
  --help, -h     Show this message
"""

import sys
import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import correlate

HELP = __doc__.strip()

FORCE = '--force' in sys.argv

if '--help' in sys.argv or '-h' in sys.argv:
    print(HELP)
    sys.exit(0)

SAVE_PATH = None
TARGET_SR = None
args = list(sys.argv[1:])

if '--save' in args:
    idx = args.index('--save')
    SAVE_PATH = args[idx + 1]
    del args[idx:idx + 2]

if '--sr' in args:
    idx = args.index('--sr')
    TARGET_SR = int(args[idx + 1])
    del args[idx:idx + 2]

args = [a for a in args if a not in ('--force',)]

if len(args) < 2:
    print(HELP)
    sys.exit(1)

DRY_PATH = args[0]
WET_PATH = args[1]

# Use first 5 seconds for correlation — fast even on large files.
MEASURE_SECS = 5.0


def measure_lag(dry, wet, sr):
    """Return lag in samples. Positive = wet is delayed. Negative = wet is early."""
    n = min(len(dry), len(wet), int(MEASURE_SECS * sr))
    corr = correlate(wet[:n], dry[:n], mode='full')
    peak = int(np.argmax(corr))
    return peak - (n - 1)


def confirm_overwrite(path):
    if not os.path.exists(path) or FORCE:
        return True
    resp = input(f"Overwrite {path}? [y/N] ").strip().lower()
    return resp == 'y'


def main():
    print(f"Dry: {DRY_PATH}")
    print(f"Wet: {WET_PATH}")

    load_sr = TARGET_SR  # None = native SR of dry
    dry, sr = librosa.load(DRY_PATH, sr=load_sr, mono=True)
    wet, _  = librosa.load(WET_PATH, sr=sr,      mono=True)

    if TARGET_SR:
        print(f"Resampled to {sr} Hz")

    lag = measure_lag(dry, wet, sr)
    lag_ms = lag / sr * 1000

    if lag > 0:
        print(f"Lag: {lag} samples ({lag_ms:.2f} ms) — wet is delayed, trim {lag} from wet start")
        wet_aligned = wet[lag:]
    elif lag < 0:
        print(f"Lag: {-lag} samples ({-lag_ms:.2f} ms) — wet is early, prepend {-lag} zeros to wet start")
        wet_aligned = np.concatenate([np.zeros(-lag, dtype=wet.dtype), wet])
    else:
        print(f"Lag: 0 samples — files are already aligned")
        wet_aligned = wet

    n = min(len(dry), len(wet_aligned))
    wet_aligned = wet_aligned[:n]

    print(f"Aligned length: {n} samples ({n / sr:.1f} s)")

    if SAVE_PATH:
        if confirm_overwrite(SAVE_PATH):
            os.makedirs(os.path.dirname(SAVE_PATH) or '.', exist_ok=True)
            sf.write(SAVE_PATH, wet_aligned, sr, subtype='FLOAT')
            print(f"Saved {SAVE_PATH}")


if __name__ == "__main__":
    main()
