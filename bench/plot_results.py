"""
Reads benchmark CSVs and ESR eval outputs; generates figures and a headline
table for the Csound conference paper.

Outputs (written to --out-dir):
    cost_frontier.png   ESR vs CPU%/voice scatter for single-voice neural runs
    polyphony_plot.png  Total CPU% vs voice count for deployed models
    headline_table.md   Paper-ready markdown table

Usage:
    python bench/plot_results.py [options]

Options:
    --dsp-csv PATH    DSP benchmark CSV (default: bench/results/dsp_5600x_kr.csv)
    --opcode-csv PATH Opcode benchmark CSV (default: bench/results/opcode.csv)
    --out-dir DIR     Output directory (default: bench/results)
    --show            Open interactive matplotlib window after saving
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── Paths to static ESR eval outputs ──────────────────────────────────────────

ESR_FILES = {
    "moognn_32u":  "models/25_moog_100-20k_32u/eval/evalOutput.txt",
    "moognn_64u":  "models/16_moog_100-20k_64u_w256/eval/evalOutput.txt",
    "moognn_128u": "models/19_moog_100-20k_128u_w256/eval/evalOutput.txt",
    "moognn_256u": "models/26_moog_100-20k_256u/eval/evalOutput.txt",
}

# Display metadata per implementation label
META = {
    "moognn_32u":   dict(gru=32,  label="moognn 32u",               color="#4477AA", marker="o", ms=7),
    "moognn_64u":   dict(gru=64,  label="moognn 64u",               color="#EE6677", marker="o", ms=9),
    "moognn_128u":  dict(gru=128, label="moognn 128u (deployed)",   color="#228833", marker="o", ms=11),
    "moognn_256u":  dict(gru=256, label="moognn 256u",              color="#AA3377", marker="o", ms=13),
    "distnn_128u":  dict(gru=128, label="distnn 128u (deployed)",   color="#CCBB44", marker="s", ms=9),
    "moogladder":   dict(gru=None, label="moogladder (Csound DSP)", color="#66CCEE", marker="^", ms=9),
    "RKSimulation": dict(gru=None, label="RKSimulation (ODE ref.)", color="#AAAAAA", marker="D", ms=8),
}

NEURAL_ORDER    = ["moognn_32u", "moognn_64u", "moognn_128u", "moognn_256u"]
POLYPHONY_ORDER = ["moognn_128u", "distnn_128u", "moogladder", "RKSimulation"]


# ── ESR parsing ────────────────────────────────────────────────────────────────

def parse_esr_file(path):
    """Return mean ESR (dB) across all cutoffs, or None if file not found."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        dbs = [float(m.group(1))
               for line in f
               for m in [re.search(r"(-?\d+\.\d+)dB", line)] if m]
    return sum(dbs) / len(dbs) if dbs else None


def load_esr_table():
    return {label: parse_esr_file(path) for label, path in ESR_FILES.items()}


# ── Benchmark CSV loading + aggregation ───────────────────────────────────────

def load_csvs(dsp_csv, opcode_csv):
    frames = []
    for path in (dsp_csv, opcode_csv):
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
        else:
            print(f"  (not found, skipping: {path})")
    if not frames:
        raise FileNotFoundError("No benchmark CSVs found.")
    return pd.concat(frames, ignore_index=True)


def aggregate(df):
    """Median cpu_pct and vrt per (implementation, n_voices)."""
    return (
        df.groupby(["implementation", "n_voices"], as_index=False)
          .agg(cpu_pct=("cpu_pct", "median"),
               vrt=("voices_at_realtime", "median"))
          .sort_values(["implementation", "n_voices"])
    )


# ── Plot 1: cost / accuracy frontier ──────────────────────────────────────────

def plot_frontier(agg, esr, out_path, show):
    single = agg[agg["n_voices"] == 1]
    neural = [l for l in NEURAL_ORDER if esr.get(l) is not None
              and not single[single["implementation"] == l].empty]

    if not neural:
        print("No single-voice neural data — skipping frontier plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    # Scatter points: accuracy = -ESR dB (higher = better)
    pts = []
    for label in neural:
        row = single[single["implementation"] == label]
        m   = META[label]
        acc = -esr[label]
        cpu = row["cpu_pct"].iloc[0]
        pts.append((acc, cpu))
        ax.scatter(acc, cpu,
                   color=m["color"], marker=m["marker"], s=m["ms"] ** 2,
                   zorder=3, label=m["label"])

    # Pareto line connecting neural points (sorted by accuracy)
    pts_sorted = sorted(pts, key=lambda p: p[0])
    xs, ys = zip(*pts_sorted)
    ax.plot(xs, ys, color="#cccccc", linestyle="--", linewidth=0.9, zorder=2)

    # Horizontal reference lines for DSP baselines
    x_lo = min(xs) - 2.0
    x_hi = max(xs) + 4.5
    ax.set_xlim(x_lo, x_hi)

    for label in ("RKSimulation", "moogladder"):
        row = single[single["implementation"] == label]
        if row.empty:
            continue
        m   = META[label]
        cpu = row["cpu_pct"].iloc[0]
        ax.axhline(cpu, linestyle=":", color=m["color"], linewidth=1.2, zorder=1)
        ax.text(x_hi - 0.2, cpu + 0.25, m["label"],
                ha="right", va="bottom", fontsize=8, color=m["color"])

    # Annotate deployed model
    dep_row = single[single["implementation"] == "moognn_128u"]
    if not dep_row.empty and esr.get("moognn_128u"):
        acc = -esr["moognn_128u"]
        cpu = dep_row["cpu_pct"].iloc[0]
        ax.annotate("deployed",
                    xy=(acc, cpu), xytext=(acc - 1.8, cpu + 2.0),
                    fontsize=8, color=META["moognn_128u"]["color"],
                    arrowprops=dict(arrowstyle="->",
                                    color=META["moognn_128u"]["color"], lw=0.8))

    ax.set_xlabel("Mean accuracy  (−ESR dB, higher = better)")
    ax.set_ylabel("CPU % / voice  (single core, 48 kHz, ksmps=64)")
    ax.set_title("moognn: accuracy vs CPU cost per voice")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ── Plot 2: polyphony scaling ──────────────────────────────────────────────────

def plot_polyphony(agg, out_path, show):
    present = [l for l in POLYPHONY_ORDER if l in agg["implementation"].values]
    if not present:
        print("No polyphony data yet — skipping polyphony plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    for label in present:
        sub = agg[agg["implementation"] == label].sort_values("n_voices")
        m   = META[label]
        ax.plot(sub["n_voices"], sub["cpu_pct"],
                color=m["color"], marker=m["marker"], markersize=m["ms"],
                label=m["label"], linewidth=1.6)

    ax.axhline(100, linestyle="--", color="red", linewidth=1.0,
               label="100% CPU (realtime limit)")

    all_voices = sorted(agg["n_voices"].unique())
    voice_ticks = [v for v in [1, 2, 4, 8, 16, 32, 64] if v in all_voices]
    ax.set_xscale("log", base=2)
    ax.set_xticks(voice_ticks)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))

    ax.set_xlabel("Number of voices")
    ax.set_ylabel("Total CPU %  (single core, 48 kHz, ksmps=64)")
    ax.set_title("Polyphony scaling")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ── Headline table ─────────────────────────────────────────────────────────────

def write_headline_table(agg, esr, out_path):
    rows = [
        ("moognn_128u",  "moognn 128u (deployed)"),
        ("moognn_32u",   "moognn 32u"),
        ("moognn_64u",   "moognn 64u"),
        ("moognn_256u",  "moognn 256u"),
        ("distnn_128u",  "distnn 128u (deployed)"),
        ("moogladder",   "moogladder (Csound DSP)"),
        ("RKSimulation", "RKSimulation (ODE ref.)"),
    ]

    single = agg[agg["n_voices"] == 1]
    lines  = [
        "| Implementation | GRU units | CPU% / voice | Voices at realtime | Mean ESR (dB) |",
        "|---|---|---|---|---|",
    ]

    for impl, display in rows:
        row = single[single["implementation"] == impl]
        if row.empty:
            continue
        m       = META.get(impl, {})
        gru     = str(m.get("gru") or "—")
        cpu     = f"{row['cpu_pct'].iloc[0]:.1f}%"
        vrt     = str(int(row["vrt"].iloc[0]))
        esr_val = f"{esr[impl]:.1f}" if esr.get(impl) else "—"
        lines.append(f"| {display} | {gru} | {cpu} | {vrt} | {esr_val} |")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsp-csv",    default="bench/results/dsp_5600x_kr.csv")
    ap.add_argument("--opcode-csv", default="bench/results/opcode.csv")
    ap.add_argument("--out-dir",    default="bench/results")
    ap.add_argument("--show",       action="store_true")
    args = ap.parse_args()

    print("Loading ESR data...")
    esr = load_esr_table()
    for label, val in esr.items():
        print(f"  {label:<14}: {f'{val:.1f} dB' if val is not None else 'not found'}")

    print("\nLoading benchmark CSVs...")
    df  = load_csvs(args.dsp_csv, args.opcode_csv)
    agg = aggregate(df)

    print(f"\nAggregated configs ({len(agg)} rows):")
    print(agg.to_string(index=False))

    os.makedirs(args.out_dir, exist_ok=True)

    print("\nGenerating plots...")
    plot_frontier(agg, esr,
                  os.path.join(args.out_dir, "cost_frontier.png"), args.show)
    plot_polyphony(agg,
                   os.path.join(args.out_dir, "polyphony_plot.png"), args.show)
    write_headline_table(agg, esr,
                         os.path.join(args.out_dir, "headline_table.md"))


if __name__ == "__main__":
    main()
