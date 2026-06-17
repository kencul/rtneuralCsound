# moogGen

C++ tools for generating Moog ladder filter reference data. Both use the same RK4 simulation of the Moog ladder circuit (`RKSimulationModel.h`) with 8x oversampling.

Built as part of the root CMake project:

```bash
cmake --build build --config Release --target moogGen
cmake --build build --config Release --target sweep_ref
```

---

## moogGen

Filters a WAV at a fixed grid of static cutoff frequencies, one output file per frequency. Used to generate training targets for `tensor_torch_param.py` and ground truth for `eval_param_model.py`.

```bash
build/bin/Release/moogGen.exe -f <input.wav> -o <output_dir>
```

Output files are named `{stem}_{freq}hz.wav`. For example:

```bash
build/bin/Release/moogGen.exe -f audio/bench_mono.wav -o audio/filteredOutput/bench
# writes: bench_mono_100hz.wav, bench_mono_1000hz.wav, ...
```

Cutoffs (Hz): `20, 60, 100, 125, 250, 500, 800, 1000, 2000, 4000, 8000, 12000, 16000, 20000, 24000`

Resonance is fixed at `0.5`. Input is mixed to mono before filtering.

---

## sweep_ref

Filters a WAV with a time-varying cutoff and writes two outputs: a float32 reference WAV and a per-sample CSV of normalized knob values (0-1, log-scaled 100Hz-20kHz, matching the model's input). The CSV is consumed by `eval_dynamic.py` to drive the neural model through the same cutoff trajectory for comparison.

```bash
# Exponential sweep from freq_start to freq_end over the full file
build/bin/Release/sweep_ref.exe <in.wav> <out.wav> <out.csv> log <freq_start> <freq_end> [resonance=1.0]

# Sinusoidal LFO oscillating between freq_low and freq_high
build/bin/Release/sweep_ref.exe <in.wav> <out.wav> <out.csv> lfo <freq_low> <freq_high> [resonance=1.0] [lfo_rate_hz=1.0]
```

The LFO oscillates in log-frequency space, so it moves equal musical distances up and down.

Example:

```bash
# Log sweep 100Hz to 20kHz
build/bin/Release/sweep_ref.exe audio/bench_mono.wav out.wav out.csv log 100 20000 0.5

# LFO between 100Hz and 10kHz at 2Hz
build/bin/Release/sweep_ref.exe audio/bench_mono.wav out.wav out.csv lfo 100 10000 0.5 2.0
```
