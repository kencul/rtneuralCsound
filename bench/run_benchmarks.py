"""
Drives the full opcode benchmark matrix via offline Csound renders.
Outputs bench/results/opcode.csv in the same schema as dsp_5600x_kr.csv.

Usage:
    python bench/run_benchmarks.py [options]

Options:
    --dur N       Audio duration per trial in seconds (default 10)
    --trials N    Timed trials per config (default 5)
    --out PATH    Output CSV path (default bench/results/opcode.csv)
    --only LABEL  Run only configs whose label contains this string
    --no-256u     Skip 256u configs (if model not yet trained)
    --dry-run     Print commands without executing
"""

import argparse
import csv
import os
import subprocess
import sys
import time

SR         = 48000
BLOCK_SIZE = 64
CSD        = "bench/bench_opcode.csd"

MODELS = {
    "moognn_32u":  "models/run_24_32u/weights.json",
    "moognn_64u":  "models/16_moog_100-20k_64u_w256/weights.json",
    "moognn_128u": "models/19_moog_100-20k_128u_w256/weights.json",
    "moognn_256u": "models/run_25_256u/weights.json",
    "distnn_128u": "models/dist_07_gru128_mrstft/weights.json",
}

# Each moognn label maps to a size-specific opcode name registered in
# moognn.dll. The opcode's compile-time GRU shape must match the weights.
OPCODE_NAMES = {
    "moognn_32u":  "moognn32",
    "moognn_64u":  "moognn64",
    "moognn_128u": "moognn128",
    "moognn_256u": "moognn256",
}

FIELDNAMES = [
    "implementation", "cutoff_mode", "n_voices", "block_size", "sr",
    "total_samples", "trial", "wall_ms", "rtf", "cpu_pct", "voices_at_realtime",
]


def build_matrix(include_256u: bool) -> list:
    rows = []

    def moognn_row(label, voices):
        return dict(label=label, instr=1, voices=voices,
                    weights=MODELS[label], opcode=OPCODE_NAMES[label])

    # CPU frontier: single voice at each GRU size
    for label in ["moognn_32u", "moognn_64u", "moognn_128u"]:
        rows.append(moognn_row(label, 1))
    if include_256u:
        rows.append(moognn_row("moognn_256u", 1))

    # moognn 128u polyphony sweep (voice=1 already covered above)
    for v in [2, 4, 8, 16]:
        rows.append(moognn_row("moognn_128u", v))

    # distnn 128u polyphony sweep
    for v in [1, 2, 4, 8, 16]:
        rows.append(dict(label="distnn_128u", instr=2, voices=v,
                         weights=MODELS["distnn_128u"], opcode="distnn"))

    # moogladder DSP baseline polyphony sweep
    for v in [1, 2, 4, 8, 16, 32, 64]:
        rows.append(dict(label="moogladder", instr=3, voices=v,
                         weights="", opcode=""))

    return rows


def run_csound(instr: int, voices: int, dur: float, weights: str,
               opcode: str, trials: int, dry_run: bool) -> list:
    cmd = [
        "csound", CSD,
        f"--omacro:INSTR={instr}",
        f"--omacro:VOICES={voices}",
        f"--omacro:DUR={dur}",
        f"--omacro:WEIGHTS={weights}",
        "-n", "-d", "-m0",
    ]
    if opcode:
        cmd.insert(2, f"--omacro:OPCODE={opcode}")

    if dry_run:
        print("  CMD:", " ".join(cmd))
        return []

    times = []
    for t in range(trials):
        t0     = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True)
        t1     = time.perf_counter()

        if result.returncode != 0:
            print(f"  ERROR trial {t + 1}:")
            print((result.stderr or result.stdout)[-400:])
            continue

        times.append((t1 - t0) * 1000.0)
        print(f"    trial {t + 1}: {times[-1]:.0f} ms")

    return times


def summarise(times: list, audio_ms: float, n_voices: int) -> dict:
    s = sorted(times)
    if len(s) > 2:
        s = s[:-1]          # drop slowest outlier
    median_ms = s[len(s) // 2]
    rtf       = median_ms / audio_ms
    cpu_pct   = rtf * 100.0
    vrt       = int(n_voices / rtf) if rtf > 0 else 9999
    return dict(kept=s, median_ms=median_ms, rtf=rtf, cpu_pct=cpu_pct, vrt=vrt)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dur",     type=float, default=10.0,
                    help="audio duration per trial in seconds (default 10)")
    ap.add_argument("--trials",  type=int,   default=5,
                    help="timed trials per config (default 5)")
    ap.add_argument("--out",     default="bench/results/opcode.csv")
    ap.add_argument("--only",    default="",
                    help="run only configs whose label contains this string")
    ap.add_argument("--no-256u", action="store_true",
                    help="skip 256u configs")
    ap.add_argument("--dry-run", action="store_true",
                    help="print commands without executing")
    args = ap.parse_args()

    matrix = build_matrix(include_256u=not args.no_256u)

    if args.only:
        matrix = [c for c in matrix if args.only in c["label"]]

    # Skip configs whose weights file is missing
    available = []
    for cfg in matrix:
        if cfg["weights"] and not os.path.exists(cfg["weights"]):
            print(f"SKIP  {cfg['label']:20s} voices={cfg['voices']}: "
                  f"{cfg['weights']} not found")
        else:
            available.append(cfg)
    matrix = available

    if not matrix:
        print("Nothing to run.")
        sys.exit(0)

    audio_ms  = args.dur * 1000.0
    write_hdr = not os.path.exists(args.out)
    results   = []

    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_hdr:
            writer.writeheader()

        for cfg in matrix:
            label   = cfg["label"]
            voices  = cfg["voices"]
            instr   = cfg["instr"]
            weights = cfg["weights"]
            opcode  = cfg["opcode"]

            print(f"\n{label}  voices={voices}")

            times = run_csound(instr, voices, args.dur, weights, opcode,
                               args.trials, args.dry_run)
            if not times:
                continue

            stats = summarise(times, audio_ms, voices)
            results.append((label, voices, stats))

            print(f"  -> median={stats['median_ms']:.0f} ms  "
                  f"rtf={stats['rtf']:.4f}  "
                  f"cpu={stats['cpu_pct']:.1f}%  "
                  f"voices_at_realtime={stats['vrt']}")

            total_samples = (int(args.dur * SR) // BLOCK_SIZE) * BLOCK_SIZE
            for i, t in enumerate(stats["kept"], 1):
                wrtf = t / audio_ms
                writer.writerow({
                    "implementation":     label,
                    "cutoff_mode":        "kr",
                    "n_voices":           voices,
                    "block_size":         BLOCK_SIZE,
                    "sr":                 SR,
                    "total_samples":      total_samples,
                    "trial":              i,
                    "wall_ms":            f"{t:.4f}",
                    "rtf":                f"{wrtf:.6f}",
                    "cpu_pct":            f"{wrtf * 100:.4f}",
                    "voices_at_realtime": stats["vrt"],
                })
            f.flush()

    if args.dry_run:
        return

    print(f"\nResults written to {args.out}")
    print(f"\n{'Config':<22} {'Voices':>6} {'Median ms':>10} {'RTF':>8} {'CPU%':>7} {'VRT':>6}")
    print("-" * 62)
    for label, voices, stats in results:
        print(f"{label:<22} {voices:>6} {stats['median_ms']:>10.0f} "
              f"{stats['rtf']:>8.4f} {stats['cpu_pct']:>7.1f} {stats['vrt']:>6}")


if __name__ == "__main__":
    main()
