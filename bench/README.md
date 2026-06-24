# Benchmark suite

Measures single-voice CPU cost and polyphony ceiling for the moognn, distnn,
and moogladder Csound opcodes, and the RKSimulationMoog DSP reference.
Produces the cost/accuracy frontier figure and headline table used in the
Csound conference paper.

## Test rig

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 5 5600X |
| OS | Windows 10 Enterprise 10.0.19045 |
| Compiler | MSVC 19.44.35214 (x64) |
| CMake | 4.0.3 |
| Build type | Release |
| RTNeural backend | XSIMD + AVX2 (set in CMakeLists.txt) |
| Csound | 7.0 double-samples (commit 6f3999e) |
| Sample rate | 48000 Hz |
| Block size | ksmps = 64 |
| Power plan | High Performance (CPU boost variability suppressed) |

All headline numbers in the paper were taken on this machine. M1 results are
not included; cross-architecture comparison is out of scope.

## Files

| File | Purpose |
|------|---------|
| `bench_dsp.cpp` | Standalone C++ harness for RKSimulationMoog |
| `bench_opcode.csd` | Parameterised Csound score for opcode timing |
| `run_benchmarks.py` | Driver: runs the full matrix, writes `results/opcode.csv` |
| `plot_results.py` | Reads CSVs + ESR files, writes figures and headline table |
| `results/dsp_5600x_kr.csv` | Committed DSP baseline results (RKSimulation, k-rate) |
| `results/opcode.csv` | Opcode benchmark results (generated, not committed) |
| `results/cost_frontier.png` | Figure: accuracy vs CPU%/voice |
| `results/polyphony_plot.png` | Figure: total CPU% vs voice count |
| `results/headline_table.md` | Paper-ready markdown table |

## Build

```bash
cmake -Bbuild
cmake --build build --config Release --target bench_dsp
cmake --build build --config Release --target moognn
cmake --build build --config Release --target distnn
```

Binaries output to `build/bin/Release/`.

## Running the benchmarks

### DSP baseline (RKSimulationMoog)

Already committed at `results/dsp_5600x_kr.csv`. To re-run:

```bash
build/bin/Release/bench_dsp.exe
# results written to bench/results/dsp.csv by default
# rename to dsp_<machine>_kr.csv to keep runs separate
```

Key flags:
```
--block N           ksmps block size (default 64)
--sr N              sample rate (default 48000)
--seconds N         audio duration per trial (default 10)
--trials N          timed trials per config (default 5)
--cutoff-mode kr|ar k-rate or a-rate cutoff updates (default kr)
--voices 1,2,4,...  voice counts to sweep
--out path          output CSV
```

### Opcode benchmarks (moognn, distnn, moogladder)

Run when the machine is idle and no training is in progress. The 256u model
must be trained first (`models/run_25_256u/weights.json` must exist) or pass
`--no-256u` to skip it.

```bash
# Full matrix (~15-20 min at --dur 10 --trials 5)
python bench/run_benchmarks.py

# Skip 256u until model is ready
python bench/run_benchmarks.py --no-256u

# Fill in 256u alone once training completes
python bench/run_benchmarks.py --only moognn_256u

# Quick smoke test (2s, 2 trials)
python bench/run_benchmarks.py --dur 2 --trials 2 --no-256u
```

Key flags:
```
--dur N       audio duration per trial in seconds (default 10)
--trials N    timed trials per config (default 5)
--out PATH    output CSV (default bench/results/opcode.csv)
--only LABEL  run only configs whose label contains this string
--no-256u     skip 256u configs
--dry-run     print commands without executing
```

Results append to `opcode.csv` so partial runs are safe. If re-running a
config, load the CSV into pandas and deduplicate on
`(implementation, n_voices, trial)` before plotting.

### Generating figures

```bash
python bench/plot_results.py
# or with interactive window:
python bench/plot_results.py --show
```

Works with partial data: if `opcode.csv` does not exist only the DSP
polyphony line appears. The frontier plot requires at least one neural model
entry in `opcode.csv`.

## CSV schema

Both `dsp_5600x_kr.csv` and `opcode.csv` share the same columns:

| Column | Description |
|--------|-------------|
| `implementation` | Model/opcode identifier (`moognn_128u`, `moogladder`, `RKSimulation`, …) |
| `cutoff_mode` | `kr` (one cutoff update per ksmps block) or `ar` (per sample) |
| `n_voices` | Number of simultaneous voices |
| `block_size` | ksmps |
| `sr` | Sample rate |
| `total_samples` | Samples processed per trial |
| `trial` | Trial index (1-based) |
| `wall_ms` | Wall-clock time for this trial in milliseconds |
| `rtf` | Real-time factor: `wall_ms / audio_duration_ms` |
| `cpu_pct` | `rtf × 100`; CPU % of one core for N voices |
| `voices_at_realtime` | `floor(n_voices / rtf)`; theoretical polyphony ceiling |

## Measurement notes

**Cutoff schedule.** Both harnesses run a log sweep from 100 Hz to 20 kHz
over the trial duration. This exercises the full operating range and matches
the training data distribution.

**Timing.** `bench_dsp` uses `std::chrono::steady_clock`. `run_benchmarks.py`
wraps each Csound subprocess with `time.perf_counter()`. Both include a
startup overhead component (~100 ms for Csound JSON parsing, negligible for
the C++ binary); at `--dur 10` this is under 1% of total wall time.

**Outlier removal.** The slowest trial is dropped before taking the median to
reduce OS scheduling noise. With `--trials 5` this leaves four trials.

**Linear scaling.** Confirmed on the DSP baseline: per-voice CPU cost is
constant across 1–64 voices (variance < 0.3%). The `voices_at_realtime`
figure is therefore reliable as a polyphony ceiling even for configs measured
at low voice counts.

**Voice state across trials (DSP).** `bench_dsp` constructs voices once per
voice-count and reuses them across warmup + timed trials. `RKSimulationMoog`
has no public `Reset`, so filter state at the end of trial N carries into
trial N+1. The Process loop is state-independent (no early-out branches), so
this does not bias timing for this model. Trial-to-trial variance in
`dsp_5600x_kr.csv` is <0.3%, confirming the assumption. If a future DSP
model has state-dependent cost, reconstruct voices between trials.

**k-rate vs a-rate.** The opcode reads cutoff at k-rate (once per ksmps
block). `bench_dsp` defaults to `--cutoff-mode kr` to match. The `ar` mode
(one `SetCutoff` per sample, matching how training targets were generated via
`sweep_ref`) is available for comparison but is not the headline figure.

## Caveats

- Results are single-core. Csound can run multiple cores with `--nchnls-i`
  and orchestra threading, but the opcodes themselves are single-threaded.
- Windows scheduler decisions introduce run-to-run variance of ~1-2% at low
  voice counts. High Performance power plan reduces but does not eliminate
  this.
- Csound 7 startup (~80-150 ms) is included in opcode wall times. At 10s
  duration this is <1.5% of total time.
- The DSP baseline measures `RKSimulationMoog` from `vendor/MoogLadders/`,
  which uses RK4 ODE integration. The Csound `moogladder` opcode (Huovilainen
  2004, Euler 2x) is cheaper; its cost appears in `opcode.csv` under
  `implementation = moogladder`.
