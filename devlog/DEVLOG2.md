# rtneuralCsound — condensed dev diary

This is a summary of [DEVLOG.md](DEVLOG.md). For the full experiment-by-experiment record, see that file.

---

## Goal

Train a neural network to model a 4-pole Moog ladder low-pass filter with a real-time controllable cutoff, then run it as a Csound opcode plugin.

---

## Phase 1: Baseline (TensorFlow → PyTorch)

Started with TensorFlow on CPU — too slow. Switched to PyTorch with CUDA, which reduced epoch time to under a second. Built the training pipeline around a GRU-based architecture with a Conv1d feature extractor.

Key training improvements made along the way:
- **Gradient clipping** to prevent GRU explosions during long runs
- **Save-best checkpoint** instead of taking the final epoch
- **Larger windows (8192 samples)** to give the GRU longer context for high-resonance behavior
- **Warmup period**: feed the GRU extra samples before the scored window so it can reach a realistic hidden state; loss is only computed on the actual window. Allows large batches while maintaining stateful behavior.
- **Pre-emphasis loss**: high-pass the prediction and target before computing ESR, forcing the model to get the filter rolloff right, not just the low frequencies.

Static (no-parameter) models reached ~-50dB ESR across the board — nearing inaudible error.

---

## Phase 2: Parameterized model (cutoff as a knob)

Added a normalized knob value (log-scaled 0–1 over 20Hz–20kHz) as a second input channel.

**Training data**: a C++ tool (`moogGen`) filters a benchmark WAV at a grid of static cutoff frequencies using a Runge-Kutta Moog ladder implementation with 8× oversampling for stability above 12kHz.

**Key problems solved:**

- **ESR math breaks with mixed cutoffs in a batch**: high-cutoff windows dominate the denominator, making low-cutoff windows nearly invisible to the loss. Fixed by computing ESR per-cutoff rather than across the batch.
- **Knob input placement**: passing the knob into Conv1d (a pattern recognizer) is a waste. Fixed by appending the knob after the conv stage, just before the GRU.
- **Bad training data at 16–20kHz**: the Moog implementation went unstable without oversampling, producing a piercing whistle in the training targets. Fixed with 8× oversampling.

After those fixes, the model reached ~-50dB ESR across 20Hz–20kHz — essentially inaudible error everywhere.

---

## Phase 3: Architecture ablation

Motivated by whether `knob_to_h0` and `LayerNorm` are load-bearing components, or whether the model could be simplified to all-RTNeural-native layers.

**`knob_to_h0`**: a small Linear+Tanh network that maps the knob value to the GRU's initial hidden state before each window. Gives the GRU prior knowledge of the target filter mode.

**`LayerNorm`**: normalizes the Conv1d output (16 arbitrary-scale channels) before the GRU sees it.

All four 32-unit combinations were trained. Key findings:

| Freq | k2h0 + LN | k2h0 only | LN only | neither |
|------|-----------|-----------|---------|---------|
| 20Hz | -27.5dB | -22.6dB | -19.4dB | -19.9dB |
| 500Hz | -48.6dB | -42.6dB | -37.8dB | -39.0dB |
| 1kHz | -48.9dB | -43.2dB | -40.0dB | -42.3dB |

- `knob_to_h0` alone: ~3dB improvement, most pronounced at low frequencies.
- `LayerNorm` alone: negligible improvement.
- **Both together: ~10dB improvement** — they are synergistic. `LayerNorm` normalizes the conv features so `knob_to_h0`'s h0 seed is more meaningful.

A 64-unit all-native model (run 11, no k2h0, no LN) was also trained as a deployment option if the custom inference path proved too slow. It reached ~-43dB at most frequencies, ~5dB behind the 32-unit k2h0+LN baseline at mid-low frequencies.

**Run 14** (64 units + k2h0 + LN) was trained as the architecture sweet spot: consistently 1–2dB better than run 11 across the board, reaching -48dB at high frequencies. This is the current deployed model.

---

## Phase 4: Csound opcode

Built `moognn.dll` as a Csound plugin using the CPOF (C++ Plugin Opcode Framework). Key build notes:

- The Csound vendor submodule is at a pre-7.0 snapshot; building against it puts `AppendOpcode` at the wrong offset and opcodes silently fail to register. Build against the **installed Csound 7 headers** (`C:/Program Files/Csound7/include/csound/`) instead.
- RTNeural and standard library headers must be included **before** Csound headers to avoid macro collisions with C++ reserved words.

The opcode signature is `moognn aout, ain, Spath, kcutoff`. The Conv1d and GRU stages are in separate RTNeural `ModelT` instances. The knob is normalized and appended as the 17th GRU input per sample. `LayerNorm` and `KnobToH0` are implemented as small manual structs (RTNeural has no native support for either).

The model path is loaded from JSON at i-time. To avoid a ~8ms parse blocking the first MIDI note, a `moognn_preload "path"` opcode pre-warms the JSON cache at score time 0 before any notes fire.

**Per-note click fix**: Csound reuses instrument memory blocks without calling C++ constructors, so `fade_counter` was not reset between notes. Explicitly resetting it to 0 at the top of `init()` fixed clicks on notes 3+. A 2048-sample linear fade-in at note-on covers the GRU's cold-start convergence window.

Two test scripts in `csound/`:
- `test_passthrough.csd` — processes `audio/bench_mono.wav` with a `linseg` cutoff sweep
- `test_midi_saw.csd` — live MIDI sawtooth with CC 110 mapped to cutoff on a log scale

---

## Phase 5: Warmup experiment series

The 2048-sample (42ms) fade-in is long enough to be noticeable without an envelope. Three runs compared shorter training warmup sizes:

| Run | Warmup | 20Hz | 500Hz | 1kHz | Notes |
|-----|--------|------|-------|------|-------|
| 11 | 2048 | -24.9dB | -42.8dB | -43.2dB | baseline |
| 12 | 512 | -19.1dB | -43.2dB | -46.5dB | sweet spot |
| 13 | 256 | -17.4dB | -37.4dB | -43.7dB | too short |

Run 12 (512-sample warmup) was the sweet spot: mid-high accuracy improved slightly due to faster learned convergence, and the fade can drop to ~10ms. Low-frequency accuracy degraded but remains usable.

However, in practice the shorter fade still clicked. At the time this was attributed to an architectural issue — without `knob_to_h0` seeding the GRU, it has to infer the filter mode from the knob in the audio stream over time — but the actual root cause turned out to be a bug in the opcode itself: `fade_counter` was never reset between notes. Csound reuses instrument memory blocks without calling C++ constructors, so on note 3+ `fade_counter` was already at 2048 and the fade was completely skipped. The fix was a single line: reset `fade_counter = 0` at the top of `init()`.

This raises an open question: now that the fade-in is actually working, do the earlier models (runs 11–13, no k2h0) perform acceptably? All the click testing before the bug fix was measuring a broken fade. It's possible those models are fine with a working fade-in, which would mean k2h0 may not be necessary for the click problem, only for accuracy.

**Run 15** (64 units + k2h0 + LN + 512-sample warmup) is the current training run, combining the architectural accuracy of run 14 with the shorter warmup of run 12.

---

## Phase 6: Re-testing runs 11 and 13 after fade-in bug fix

With the `fade_counter` reset bug fixed, runs 11 and 13 (both without `knob_to_h0`) were re-tested via live MIDI.

- **Run 11** (2048-sample training warmup, no k2h0): faint click audible under polyphonic load even at short fade durations, even at 256 samplees. The 2048-sample training warmup taught the GRU it would have a long convergence window, so it never learned to produce correct output quickly from a cold hidden state.
- **Run 13** (256-sample training warmup, no k2h0): click-free at a 256-sample fade-in. The shorter warmup forced the model to converge fast from h0=0.

**Conclusions:**

Training warmup length is the primary factor controlling cold-start convergence speed, not `knob_to_h0`. A shorter training warmup implicitly trains the GRU to produce correct output from nearly the first sample.

`knob_to_h0` was attempting to solve two problems at once: click suppression and accuracy. It didn't solve the click, but the click problem is better solved by shortening the training warmup. `knob_to_h0` and `LayerNorm` remain valuable for the ~10dB accuracy gain shown in the ablation (Phase 3), not for note-on behavior.

Run 15 (k2h0 + LN + 512-sample warmup) was the intended next step under this reasoning. Its results are analyzed in Phase 7.

---

## Run 15 evaluation and warmup accuracy ceiling

Run 15 trained cleanly: LR stepped down 4 times (1e-3 to 6.25e-5), early stopping at epoch 251, final val_loss 0.0001. Good convergence by training metrics. The eval results told a different story.

Full comparison across all runs at key frequencies:

| Freq | Run 11 (2048w) | Run 12 (512w) | Run 13 (256w) | Run 14 (2048w+k2h0) | Run 15 (512w+k2h0) |
|------|---------------|--------------|--------------|--------------------|--------------------|
| 20Hz | -24.9dB | -19.1dB | -17.4dB | -25.0dB | -14.8dB |
| 100Hz | -38.6dB | -35.4dB | -33.3dB | -41.0dB | -30.8dB |
| 500Hz | -42.8dB | -43.2dB | -37.4dB | -43.7dB | -38.9dB |
| 1kHz | -43.2dB | -46.5dB | -43.7dB | -43.4dB | -42.9dB |
| 12kHz | -44.9dB | -47.1dB | -42.2dB | -46.1dB | -40.9dB |
| 20kHz | -46.5dB | -46.3dB | -43.0dB | -48.3dB | -42.0dB |

**Three findings:**

**Low-frequency accuracy is purely a warmup-length problem.** Runs 11 and 14 have the same 2048-sample warmup and nearly identical 20Hz accuracy (-24.9 vs -25.0dB), despite k2h0+LN giving 10dB gains at mid frequencies. k2h0 contributes nothing at 20Hz. The reason is geometric: at 48kHz a 20Hz signal has a 2400-sample period. A 256-sample warmup shows the GRU 0.11 periods; a 2048-sample warmup shows 0.85 periods. The model cannot fit what it cannot see.

**k2h0 hurts at short warmup.** Run 12 (512w, no k2h0) reaches -46.5dB at 1kHz. Run 15 (512w, k2h0+LN) reaches only -42.9dB. The h0 seeding interferes with the convergence strategy the GRU would otherwise learn for a short warmup. k2h0 is only beneficial when the warmup is long enough to actually use the seeded state.

**Run 12 is surprisingly strong.** With no k2h0 and only 512-sample warmup it beats run 11 (2048 warmup, no k2h0) at 1kHz and 12kHz. Shorter warmup training appears to encourage the model to learn more responsive mid-high representations.

**The 256-sample constraint.** A 256-sample fade at 48kHz is ~5.3ms, which is imperceptible and does not affect note envelopes. This is the target fade-in. Run 13 achieves it and is click-free, but pays a significant accuracy penalty below 250Hz. The question is whether that penalty is audible in practice, and whether architecture changes can recover some of it.

**Options considered:**

- **Raise the lower frequency boundary to 100Hz**: the 20-60Hz training examples are unlearnable at 256 warmup and add gradient noise. Removing them would let the optimizer focus on frequencies it can fit, likely improving 100-250Hz accuracy as a side effect. Most musical material has fundamentals above 100Hz. Highest leverage, lowest effort.
- **More GRU units (64 to 128)**: adds capacity everywhere but does not change the period-visibility ceiling. Worth combining with the boundary change rather than using alone.
- **LSTM instead of GRU**: the separate cell state gives LSTM a slower-changing memory track that may accumulate warmup information more effectively. RTNeural supports it natively. Costs ~33% more inference compute at the same hidden size. Worth trying if the boundary change alone is insufficient.
- **k2h0 at 256w**: ruled out by run 15 data.

---

## Eval warmup investigation

re-running evals for runs 12 and 13 with their correct warmup sizes (512 and 256) produced essentially identical numbers to the original 2048-warmup evals. The GRU converges to its correct state within a few hundred samples regardless of how much settling time the eval gives it. The previous accuracy comparisons were already valid.

## Run 16(100Hz lower boundary, 256 warmup, no k2h0, 64 units)

Trained all 300 epochs without early stopping, suggesting the cleaner training set allowed continuous slow improvement rather than plateau. Strict improvement over run 13 at every frequency:

| Freq | Run 13 (20Hz floor) | Run 16 (100Hz floor) | Delta |
|------|---------------------|----------------------|-------|
| 100Hz | -33.2dB | -34.0dB | +0.8dB |
| 250Hz | -34.9dB | -37.3dB | +2.4dB |
| 500Hz | -37.3dB | -41.0dB | +3.7dB |
| 1kHz | -43.3dB | -42.8dB | -0.5dB |
| 4kHz | -43.9dB | -45.3dB | +1.4dB |
| 16kHz | -40.0dB | -48.2dB | +8.2dB |
| 20kHz | -42.8dB | -46.8dB | +4.0dB |

The 100-500Hz range gained 1-4dB, directly from removing the unlearnable 20/60Hz gradient noise. High-frequency gains (up to 8dB at 16kHz) suggest the model was previously spending capacity trying to fit the impossible low-frequency examples. No tradeoffs observed.


## Run 17 (128-sample warmup)

Run 17 tested whether the warmup could be halved again: 128 samples, 100Hz floor, 64 units, no k2h0. Result: a clear step backwards from run 16, across the entire frequency range.

| Freq | Run 16 (256w) | Run 17 (128w) | Delta |
|------|--------------|--------------|-------|
| 100Hz | -34.0dB | -26.4dB | -7.6dB |
| 250Hz | -37.3dB | -34.9dB | -2.4dB |
| 500Hz | -41.0dB | -38.0dB | -3.0dB |
| 1kHz | -42.8dB | -39.1dB | -3.7dB |
| 4kHz | -45.3dB | -40.3dB | -5.0dB |
| 20kHz | -46.8dB | -41.9dB | -4.9dB |

Unlike previous warmup reductions where damage was concentrated at low frequencies, run 17 degraded everywhere. The training confirmed the problem: best val_loss was 0.0003 vs 0.0001 for run 16, and the LR scheduler stepped down 6 times (to 1.56e-05) without reaching convergence. At 128 samples the GRU sees only 0.27 periods of a 100Hz signal, below the threshold where it can represent the filter state at all. This is a geometric constraint, not an optimization one. k2h0 and LSTM are unlikely to help here for the same reason.

**256-sample warmup is the floor for the 100Hz lower boundary.** Run 16 is the best short-fade model. Next steps: try more units or LSTM at 256 warmup.

---

## Run 18 (LSTM)

Swapped GRU for LSTM, same config as run 16 (64 units, 256 warmup, 100Hz floor). LSTM was worse at every frequency, by a lot.

| Freq | Run 16 (GRU) | Run 18 (LSTM) | Delta |
|------|-------------|--------------|-------|
| 100Hz | -34.0dB | -26.8dB | -7.2dB |
| 500Hz | -41.0dB | -34.3dB | -6.7dB |
| 1kHz | -42.8dB | -34.8dB | -8.0dB |
| 4kHz | -45.3dB | -38.4dB | -6.9dB |
| 20kHz | -46.8dB | -45.2dB | -1.6dB |

Training reached val_loss 0.0001 and ran all 300 epochs without early stopping, so it was still slowly improving at the end. LSTM has ~33% more parameters than GRU at the same hidden size, and those extra parameters didn't have enough epochs to converge. GRU fits the data more efficiently within the 300-epoch budget. LSTM is not ruled out permanently but GRU is the right call for now.

---

## Run 19 (128 GRU units)

Doubled the GRU hidden size to 128, same config as run 16 (256 warmup, 100Hz floor, no k2h0). Clear winner.

| Freq | Run 16 (64u) | Run 19 (128u) | Delta |
|------|-------------|--------------|-------|
| 100Hz | -34.0dB | -34.7dB | +0.7dB |
| 250Hz | -37.3dB | -41.9dB | +4.6dB |
| 500Hz | -41.0dB | -45.8dB | +4.8dB |
| 1kHz | -42.8dB | -46.7dB | +3.9dB |
| 12kHz | -45.7dB | -50.2dB | +4.5dB |
| 16kHz | -48.2dB | -53.7dB | +5.5dB |
| 20kHz | -46.8dB | -52.6dB | +5.8dB |

2-6dB gains across the whole range. High frequencies now exceed -50dB, approaching the inaudible threshold from Phase 1. 100Hz barely moved (+0.7dB) as expected since that is still a warmup period-visibility constraint, not a capacity one.

Run 19 also beats run 14 (the long-warmup k2h0 model) from 500Hz upward, despite the shorter warmup and higher frequency floor. Run 14 only wins below 100Hz where run 19 has no coverage.

Training converged cleanly: early stopping at epoch 279, val_loss 0.0001. Epoch time ~12s vs ~8s for 64-unit runs, total 53 minutes.

---

## Run 20 (256 GRU units)

Doubled units again to 256. Training took 10.4 hours (150s/epoch), early stopping at epoch 294. Significant gains across the board.

| Freq | Run 19 (128u) | Run 20 (256u) | Delta |
|------|--------------|--------------|-------|
| 100Hz | -34.7dB | -41.3dB | +6.6dB |
| 250Hz | -41.9dB | -46.7dB | +4.8dB |
| 500Hz | -45.8dB | -48.9dB | +3.1dB |
| 1kHz | -46.7dB | -48.6dB | +1.9dB |
| 4kHz | -46.6dB | -51.7dB | +5.1dB |
| 12kHz | -50.2dB | -56.0dB | +5.8dB |
| 16kHz | -53.7dB | -57.7dB | +4.0dB |
| 20kHz | -52.6dB | -55.7dB | +3.1dB |

The 100Hz improvement (+6.6dB) was unexpected given the warmup period-visibility argument. More capacity helps even at the constrained low end. High frequencies now reach -56 to -58dB, well into inaudible territory.

The open question is real-time viability. 256 units is 4x the GRU compute of 128 units. At 32 samples/block (667µs budget) this may not run in real time. The opcode needs to be updated and rebuilt to get timing stats from `deinit()`. If it cannot hit real time, run 19 remains the practical deployment choice.

Running this model in the opcode, 256 units is too much computationally for polyphony. Its fine with one note, but playing two notes makes the audio crackle horribly.

The 128 unit model starts crackling at 7 notes, unstable at 6.

---

## RTNeural optimization audit

Audited the build configuration and opcode implementation for inference efficiency. Findings:

- The build cache already had `RTNEURAL_XSIMD=ON` and `RTNEURAL_USE_AVX=ON` from a prior configure, so xsimd + AVX2 (256-bit SIMD, 32-byte alignment) were already active. These were not pinned in CMakeLists.txt, meaning a fresh configure would silently drop them. Both are now explicitly set with `FORCE` so they persist.
- `ModelT` (static, compile-time shapes) is already used throughout — correct for real-time audio; eliminates virtual dispatch and enables full compiler unrolling.
- A dead `#define EIGEN_STACK_ALLOCATION_LIMIT 0` was removed from moognn.cpp (Eigen-specific, irrelevant with xsimd backend).
- No further RTNeural-level optimizations are available. The inference is fundamentally sample-by-sample sequential due to the GRU hidden state dependency.

---

## Polyphony ceiling analysis

The polyphony limits are now measured:

| Model | Voices before crackle |
|-------|-----------------------|
| Run 20 (256u) | 2 |
| Run 19 (128u) | 6 (crackles at 7) |

Increasing buffer sizes (`-b`, `-B`) does not help. The GRU runs 48,000 times per second per voice regardless of block size — buffer size only changes how often the OS callback fires, not total compute.

A literature review of the field (Wright 2019, RTNeural paper, Steinmetz 2022, etc.) confirms that all published real-time neural audio effect models are designed for monophonic always-on effects — guitar amp sims, compressors, distortion pedals. None address polyphonic per-note instantiation. The community norm for single-voice real-time is 16–64 GRU/LSTM units. 128 units at 6-voice polyphony is already above what the literature targets.

---

## Polyphony improvement options

Three approaches were evaluated for reducing compute to increase polyphony:

**Linear Recurrent Units (LRUs):** The Esqueda & Murai DAFx25 paper proposes LRUs as a more efficient recurrent architecture. The diagonal state matrix (N scalars instead of N×N weights) makes the state update fundamentally cheaper than GRU. Their results show models with <1% CPU at 128-sample buffer. However: (1) inference is still sample-by-sample sequential — the parallel scan is training-only; (2) the benchmarks are on Apple M2 Pro, though the architectural savings transfer to x86 AVX2; (3) FiLM conditioning failed for LRU in their experiments, increasing hidden size requirements, and the conditioning problem for a parametric filter is unsolved for LRUs; (4) LRUs are not in RTNeural's native layer set, requiring a custom C++ inference path. Not pursued for now.

**GRU depth (stacking layers):** The LRU paper found depth more efficient than width for LRUs. This does not transfer to GRUs — stacking GRU layers doubles sequential compute per sample and the literature has not found it beneficial for single-layer audio effects. Not pursued.

**FiLM conditioning:** Feature-wise Linear Modulation is the most actionable path. The current architecture concatenates the knob as GRU input channel 17 — the SMC 2024 paper (Simionato & Fasciani) found this is the worst conditioning method. FiLM instead applies a learned scale and shift to the GRU output features based on the knob:

```
FiLM(x, knob) = γ(knob) ⊙ x + β(knob)
[γ, β] = Linear(knob)   # 1 → 2×hidden_size
```

Post-placement (after the GRU, before Dense) consistently outperformed pre-placement in that paper. The GRU input would drop from 17 to 16 channels. FiLM computation cost is negligible: γ and β are computed once per k-cycle (knob is k-rate), and the per-sample application is 128 multiply-adds vs the GRU's ~56,000. The hypothesis is that a 64-unit GRU with FiLM might match run 19 (128-unit, concatenation), since the GRU no longer spends capacity inferring the filter mode from the knob stream.

**Knowledge distillation:** Complementary to any smaller model training. Run 19 acts as teacher (frozen inference), the smaller student model trains against both the target signal and the teacher's outputs:
```
loss = ESR(student, target) + λ * MSE(student, teacher_output)
```
Approximately 10 lines added to the training loop. Can be combined with FiLM. Note: the Carson DAFx25 paper referenced elsewhere uses a teacher-student method for anti-aliasing (same-size model fine-tuning), which is a different use of the term — knowledge distillation for compression is a distinct technique.

---

## Repo cleanup

Moved `moogGen/` into `src/moogGen/` to keep all C++ tools under `src/`. Renamed `ref/` to `models/` since that's what it actually contains. Updated all hardcoded paths in the .csd files, training script, README, and CMakeLists.

---

## sweep_ref tool

Added `src/moogGen/sweep_ref.cpp`, a standalone C++ tool that runs the RK Moog with a time-varying cutoff and writes two outputs: a float32 reference WAV and a per-sample CSV of normalized knob values (same 0-1 log scaling the model uses). The CSV is what `eval_dynamic.py` will read to drive the neural model through the identical cutoff trajectory.

Two sweep modes:

- `log`: exponential ramp from freq_start to freq_end over the full file
- `lfo`: sinusoidal oscillation in log-frequency space between freq_low and freq_high at a given rate

The normalization happens in C++ so the CSV values are already in model space and Python doesn't need to know FREQ_MIN/FREQ_MAX.

Generated three reference files in `audio/filteredOutput/bench/`:

- `bench_mono_sweep_log_100-20000hz` -- full log sweep, 40 seconds
- `bench_mono_lfo_slow_100-10000hz` -- LFO at 0.25Hz (4s period)
- `bench_mono_lfo_fast_100-10000hz` -- LFO at 5Hz (0.2s period)

All use resonance=0.5 to match the training data.

---

## eval_dynamic.py

Written to evaluate the model against the sweep_ref reference files. Takes a model checkpoint, a reference WAV, and the companion knob CSV. Runs the model with the per-sample knob schedule from the CSV (the values are already normalized to [0,1] by sweep_ref, so no frequency math needed in Python), carries GRU hidden state across 8192-sample chunks, and scores the output against the reference WAV.

Output: overall ESR in dB printed to stdout, plus a 4-panel plot (reference spectrogram, model output spectrogram, difference spectrogram, windowed ESR over time at 0.5s resolution).

`eval_param_model.py` was also updated to match: it now produces the same 4-panel layout, with the spectrograms shown at 1kHz and the fourth panel showing ESR by frequency on a log scale. Its hardcoded `GRU_HIDDEN = 256` was replaced with a `[gru_hidden=128]` positional argument, defaulting to 128 to match run 19.

Both scripts share the same flags:

- `--save <dir>` -- write output files to the given directory, prompting before overwriting. For `eval_param_model` the files are always `evalOutput.txt` and `evalOutput.png`. For `eval_dynamic` the filenames include the sweep name derived from the ref WAV (e.g. `evalOutput_lfo_fast_100-10khz.txt`) so multiple sweeps can be saved to the same model directory without collision.
- `--force` -- skip the overwrite prompt; used for batch runs
- `--show` -- open the interactive plot window; without it the script runs headlessly
- `--help` / `-h` -- print usage; also printed on bad or missing arguments

```
python eval_param_model.py <model.pt> [warmup] [freq_min] [gru_hidden] [--save <dir>] [--force] [--show]
python eval_dynamic.py     <model.pt> <ref.wav> <ref.csv> [gru_hidden] [warmup] [--save <dir>] [--force] [--show]
```

---

## Cross-model dynamic eval (runs 16, 17, 19, 20)

Static and dynamic evals run across all four 100Hz-floor models.

**Static ESR (dB) at key frequencies:**

| Freq | Run 16 (64u, w256) | Run 17 (64u, w128) | Run 19 (128u, w256) | Run 20 (256u, w256) |
|------|-------------------:|-------------------:|--------------------:|--------------------:|
| 100 Hz | -34.0 | -26.4 | -34.7 | -41.3 |
| 250 Hz | -37.3 | -34.9 | -41.9 | -46.7 |
| 1 kHz | -42.8 | -39.1 | -46.7 | -48.6 |
| 4 kHz | -45.3 | -40.3 | -46.6 | -51.7 |
| 16 kHz | -48.2 | -45.1 | -53.7 | -57.7 |

**Dynamic ESR (dB) overall:**

| Model | sweep_log | lfo_slow (0.25 Hz) | lfo_fast (5 Hz) | fast LFO gap vs static 1 kHz |
|-------|----------:|-------------------:|----------------:|-----------------------------:|
| Run 16 (64u, w256) | -41.2 | -40.3 | -39.5 | 3.3 dB |
| Run 17 (64u, w128) | -38.0 | -35.6 | -26.8 | 12.3 dB |
| Run 19 (128u, w256) | -40.8 | -42.0 | -29.4 | 17.3 dB |
| Run 20 (256u, w256) | -41.6 | -47.7 | -37.3 | 11.3 dB |

**Warmup effect (runs 16 vs 17):** halving warmup from 256 to 128 samples costs 4-8 dB at every static frequency, with the largest hit at 100 Hz (7.6 dB). On the fast LFO the penalty is even larger: run 17 drops to -26.8 dB, the worst result across all conditions.

**Capacity effect (runs 16, 19, 20):** doubling units at fixed warmup gives consistent static gains. 16 to 19 adds 3-7 dB, mostly at high frequencies; the 100 Hz floor barely moves (+0.7 dB), confirming the period-visibility constraint still dominates there. 19 to 20 adds another 4-7 dB, and notably +6.6 dB at 100 Hz -- at 256 units, capacity starts to partially overcome the warmup constraint.

**Log sweep is a poor differentiator:** all four models cluster between -38 and -42 dB. The 40-second range pulls the average down at the 100 Hz end where all models are weak, and the slow rate does not test tracking speed. The slow LFO (-40 to -48 dB) is a better proxy for static performance.

**Larger models degrade more under fast modulation.** Run 16 loses only 3.3 dB from its static 1 kHz result to the fast LFO. Run 19, which is 4 dB better statically, collapses by 17.3 dB on the fast LFO and ends up worse than run 16 (-29.4 vs -39.5). Run 20 fares better than run 19 but still loses 11 dB. The accuracy gains from additional capacity appear primarily in static or slowly-varying conditions. Under 5 Hz modulation those gains largely disappear.

**Interpretation:** the model infers the current filter mode by integrating the knob value over time through the GRU hidden state. A static cutoff gives the GRU unlimited time to settle; a 5 Hz LFO forces it to re-converge 10 times per second. Larger models have learned more precise but slower-adapting representations. This is an architectural problem. The conditioning mechanism, not capacity, is the bottleneck. FiLM conditioning addresses this directly by applying scale and shift to the GRU output from the knob in a single step, rather than requiring the GRU to integrate it over time. Variable-parameter training data (LFO sweeps as targets) is also likely necessary.

## FiLM Conditioning Implementation Plan
### Context

Dynamic eval (DEVLOG2) showed the current knob-concatenation approach causes large degradation under fast modulation — run 19 drops 17 dB on the 5 Hz LFO. The GRU must integrate the knob value over time through hidden state to infer the current filter mode; fast sweeps don't give it enough settling time. FiLM applies scale and shift to GRU outputs in a single step, making conditioning speed-independent.

### Design Decisions (backed by SMC 2024 — Simionato & Fasciani)

https://smcnetwork.org/smc2024/papers/SMC2024_paper_id83.pdf

- Post-GRU placement The paper tested pre, post, and pre-post placements across all conditioning methods. Post (after the recurrent layer, before Dense) was the consistent winner for compressor-like effects. The paper's reasoning: "it is more beneficial for the networks to use the information given by the control parameters to project the output of the [state] layer... rather than influencing the inference of the recurrent layer." A Moog filter is closer to a compressor than an overdrive, so post applies.

- Naked FiLM (no GLU gate) The paper's FiLM-GLU adds a softsign gate after the affine transform, which lets the network learn to suppress conditioning when irrelevant. For a Moog with a single always-active cutoff knob, there is no "ignore the knob" case — the gate has nothing to do. The simplest form (output = γ ⊙ x + β) is sufficient and adds fewer parameters.

- Linear FiLM (not cubic/quintic) The paper tested odd-order nonlinear transforms inside FiLM and found they help for overdrive — where the parameter-to-response relationship is strongly nonlinear — but not for compressors. The Moog cutoff knob is already log-normalized so its relationship to the sonic response is approximately linear in model input space. No nonlinear transform.

- 64 GRU units Hypothesis from DEVLOG2: the GRU currently spends capacity integrating the knob through time. With FiLM handling conditioning in a single step, a 64-unit GRU should match larger concatenation-based models dynamically. This makes run 16 (64u, concatenation) the direct static comparison and run 19 (128u, concatenation) the dynamic comparison target.

### Plan Outline

1. tensor_torch_param.py
   - GRU_HIDDEN: 256 → 64
   - Model: GRU input 17 → 16 (remove knob channel)
   - Model: add self.film = nn.Linear(1, 2 * GRU_HIDDEN)
   - forward(): run GRU on conv_out only, apply FiLM post-GRU, then Dense + skip

2. eval_param_model.py
   - Same Model change (also returns h for stateful chunk inference)

3. eval_dynamic.py
   - Same Model change

4. moognn.cpp
   - Add constexpr GRU_H = 64
   - Add FiLM struct: weight[2*H], bias[2*H], compute(knob, gamma, beta)
   - Split RecurrentStage (GRU+Dense) → GRUStage + DenseStage (FiLM goes between)
   - init(): load film.weight (shape [2H,1] → flatten) and film.bias from JSON
   - aperf(): compute gamma/beta once per block (k-rate), apply per-sample between GRU and Dense


---

## Python tooling refactor

The `Model` class was duplicated across `tensor_torch_param.py`, `eval_param_model.py`, and `eval_dynamic.py`. Any architecture change required touching all three files in sync — a practical problem when adding FiLM conditioning, which changes the model signature enough that a shape mismatch would silently corrupt eval results if one file was missed.

**New structure:** each architecture lives in its own file:

- `model_concat.py` — knob-concatenation architecture (all runs 11–20)
- `model_film.py` — FiLM conditioning architecture (pre- and post-GRU placements, experiments complete)

Both define `Model(gru_hidden)` and return `(output, h)` from `forward()` so eval scripts can carry GRU hidden state across chunks. The training and eval scripts import from these files directly.

**Checkpoint format updated.** New checkpoints embed arch metadata:

```python
{'model_state': ..., 'arch': 'concat', 'gru_hidden': 128, 'freq_min': 100.0, 'freq_max': 20000.0}
```

Eval scripts auto-detect architecture and hidden size from the checkpoint. Legacy checkpoints (runs 11–20, raw state dicts) are handled by inferring arch from GRU input width (17 = concat, 16 = film) and hidden size from `weight_ih_l0` shape.

**Updated eval usage.** The `gru_hidden` and `freq_min` positional args are removed from both eval scripts — they now come from the checkpoint. `warmup` default updated from 2048 → 256 in `eval_param_model.py` to match current short-warmup runs.

```bash
python eval_param_model.py <model.pt> [warmup] [--save <dir>] [--show]
python eval_dynamic.py     <model.pt> <ref.wav> <ref.csv> [warmup] [--save <dir>] [--show]
```

Verified on runs 16 and 19: ESR numbers match prior results exactly.

With the model files in place, adding a new architecture for training is two lines: import the right model file and set `GRU_HIDDEN`. `tensor_torch_film.py` is the FiLM training script, identical to `tensor_torch_param.py` except it imports from `model_film` and sets `GRU_HIDDEN = 64`. The checkpoint it saves is tagged `'arch': 'film'` so eval scripts auto-detect it.

```bash
env/Scripts/python.exe python/tensor_torch_film.py models/21_moog_film_64u_w256
```

---

## Run 21 (FiLM, 64 units, 256 warmup)

### FiLM initialization bug

The first training attempt plateaued immediately at val_loss ~0.21 and never improved. The cause: `nn.Linear` default initialization sets the FiLM weights and biases to random values in roughly [-1, 1]. At epoch 1 the FiLM layer is randomly scaling and inverting GRU output before Dense sees it, putting the optimizer in a hole it cannot escape from.

Fix: zero the FiLM weights and set the gamma bias to 1, beta bias to 0. This makes FiLM a pass-through at initialization -- gamma=1, beta=0 for any knob value -- so the model learns the base GRU behavior first and gradually acquires the modulation.

```python
nn.init.zeros_(self.film.weight)
nn.init.zeros_(self.film.bias)
self.film.bias.data[:gru_hidden] = 1.0
```

### Training

Training ran 163 epochs before early stopping (3 LR steps: 1e-3 to 1.25e-4). Best val_loss: **0.2100**.

For reference, run 16 (64 units, concat) converged to val_loss 0.0001. The FiLM model is 2000x worse in linear ESR. Despite the identity initialization fix, the model fundamentally failed to learn.

### Eval results

**Static ESR:**

| Freq | Run 21 (FiLM, 64u) | Run 16 (concat, 64u) |
|------|-------------------:|--------------------:|
| 100 Hz | -5.4 dB | -34.0 dB |
| 500 Hz | -4.9 dB | -41.0 dB |
| 1 kHz | -5.2 dB | -42.8 dB |
| 4 kHz | -8.1 dB | -45.3 dB |
| 16 kHz | -4.4 dB | -48.2 dB |

**Dynamic (fast LFO, 5 Hz):** -6.4 dB. Run 16 was -39.5 dB.

### Analysis

Post-GRU FiLM does not work for parametric filter emulation.

In the concat model the GRU sees the knob at every time step and learns different hidden state trajectories for different cutoffs. The GRU at 100Hz behaves differently from the GRU at 20kHz because the knob is part of its input at every step.

In the FiLM model the GRU receives only conv audio features -- no knob information at all. It processes all cutoff frequencies identically and produces the same hidden state regardless of the target filter mode. FiLM can only scale and shift the result after the fact; it cannot change what the GRU computed. A 100Hz low-pass and a 20kHz near-passthrough are not related by a scale and shift of the same hidden state.

The SMC 2024 paper validated post-GRU FiLM for a compressor, where the core computation (detect transient, apply gain) is consistent across parameter values and conditioning only modulates magnitude. A Moog filter at different cutoffs requires different recurrent behavior, not just different output scaling.

### Next steps

Two paths forward:

**Variable training data.** The fast LFO degradation has two causes: slow conditioning (the knob integration problem) and distributional mismatch (the model was trained only on static cutoffs). Variable training data -- LFO-swept targets mixed into training -- addresses the second cause directly and requires no architecture change.

**Pre-GRU FiLM.** Applying FiLM to the conv features before the GRU gives the GRU conditioned input, so it can route itself differently per cutoff. This preserves the architectural motivation (single-step conditioning) while fixing the information bottleneck. Worth one experiment after variable training data is established.

## Script updates for FiLM architecture variants

Run 21 (post-GRU) and run 22 (pre-GRU) both carry `arch='film'` in their checkpoints but have incompatible FiLM weight shapes. `model_film.py` was updated to support both placements via a `film_pre` constructor argument (default `True`). The eval scripts auto-detect placement from the FiLM weight shape: `[32, 1]` (2 x 16 conv channels) is pre-GRU; `[128, 1]` (2 x 64 GRU units) is post-GRU.

---

## Run 22 (pre-GRU FiLM, 64 units, 256 warmup)

### Training

Val_loss 0.0001, ran all 300 epochs through 5 LR steps (1e-3 to 3.13e-5). No early stopping. 34.7 minutes total.

The training curve has a notable phase transition. Epochs 1-22 descend slowly from 0.29 to 0.22 -- FiLM is near identity, GRU is learning audio processing. Between epochs 22-28 val_loss drops suddenly from 0.246 to 0.004. That is the moment FiLM breaks out of identity and the GRU and FiLM find a cooperative solution. Post-GRU run 21 never reached this transition because the GRU had no useful representation for FiLM to modulate.

### Eval results

**Static ESR (dB):**

| Freq | Run 16 (concat, 64u) | Run 19 (concat, 128u) | Run 22 (FiLM pre, 64u) |
|------|--------------------:|---------------------:|----------------------:|
| 100 Hz | -34.0 | -34.7 | -33.4 |
| 500 Hz | -41.0 | -45.8 | -39.2 |
| 1 kHz | -42.8 | -46.7 | -39.6 |
| 4 kHz | -45.3 | -46.6 | -41.5 |
| 16 kHz | -48.2 | -53.7 | -47.9 |
| 20 kHz | -46.8 | -52.6 | -49.1 |

**Dynamic (fast LFO, 5 Hz):** run 22: -24.2 dB. Run 16: -39.5 dB. Run 19: -29.4 dB.

### Analysis

Pre-GRU FiLM works as an architecture. Static accuracy is within 2-3 dB of run 16 at mid frequencies and matches or beats it above 12 kHz. Pre-GRU FiLM at 64 units is roughly equivalent to concat at 64 units, not an improvement.

The fast LFO result is the key finding. Run 22 at -24.2 dB is the worst dynamic result of any viable model, worse than both run 16 and run 19. The static-to-LFO gap is 15.4 dB at 1 kHz, similar to run 19 (17.3 dB) and far above run 16 (3.3 dB).

Pre-GRU FiLM does not improve dynamic tracking. FiLM conditions the GRU inputs in a single step, but the GRU hidden state still integrates those inputs over time to settle into the correct filter mode. When the cutoff sweeps at 5 Hz the hidden state is pulled in a new direction 10 times per second. The integration problem is in the recurrent state, not the conditioning pathway. FiLM placement does not affect this.

Run 16 (concat, 64 units) remains the most dynamically stable model. The training data distribution is the bottleneck, not the architecture. The FiLM conditioning experiments are concluded.

---

## Run 23 (128 units, 256 warmup, variable training data)

Same architecture as run 19 (128 GRU units, concat conditioning, 256 warmup, 100Hz floor). Training data extended with a 5Hz LFO sweep (100Hz-10kHz) of `testSound_mono.wav` generated by `sweep_ref`. Validation extended with the same LFO sweep of `bench_mono.wav`. Static bench cutoffs remain the primary validation set.

`tensor_torch_variable.py` adds `load_variable_windows`, which reads per-sample knob values from the sweep_ref CSV rather than tiling a scalar. The variable windows are concatenated onto the static training set before shuffling.

**Static ESR (dB):**

| Freq | Run 19 (static) | Run 23 (variable) | Delta |
|------|----------------:|------------------:|------:|
| 100 Hz | -34.7 | -34.6 | 0 |
| 250 Hz | -41.9 | -39.7 | -2.2 |
| 500 Hz | -45.8 | -42.5 | -3.3 |
| 1 kHz | -46.7 | -45.0 | -1.7 |
| 16 kHz | -53.7 | -51.1 | -2.6 |
| 20 kHz | -52.6 | -52.2 | 0 |

**Dynamic ESR (dB) overall:**

| Condition | Run 19 | Run 23 | Delta |
|-----------|-------:|-------:|------:|
| sweep_log | -40.8 | -43.7 | +2.9 |
| lfo_slow | -42.0 | -42.3 | +0.3 |
| lfo_fast | -29.4 | -39.5 | **+10.1** |

The fast LFO result is the headline finding. Run 23 gains 10.1 dB on the 5Hz LFO, matching run 16 (-39.5 dB) despite having twice the GRU units. Static accuracy drops 1-3 dB at mid frequencies, a reasonable trade. This confirms that distributional mismatch was the primary cause of fast LFO degradation, not architecture or capacity.

---

## Current state

- Run 23 (128 units, variable training) is now the best overall model: matches run 19 statically at extreme frequencies, +10 dB on fast LFO, 6-voice polyphony
- Run 19 (128 units, static training) remains the static accuracy reference
- Run 16 (64 units, static training) is the most dynamically stable static-trained model
- Run 20 (256 units) is better statically but limited to single voice
- The RTNeural build is fully optimized (xsimd + AVX2, pinned in CMakeLists.txt)
- FiLM conditioning experiments (runs 21-22) are complete. Architecture is not the bottleneck for fast LFO degradation -- training data distribution is.

---

## Eval methodology improvements

### Motivation

Two problems with the original eval setup:

1. **Val set = eval set.** Bench data was used both to guide early stopping and LR scheduling during training and as the final reported metric. The best checkpoint was selected to minimize bench loss, so bench eval numbers are slightly optimistic. A fully held-out test set is needed for unbiased results.

2. **Eval on training-distribution data only.** Static evals used the exact cutoff frequencies in the training grid. Dynamic evals only had the 5Hz LFO the model was trained on (for run 23). Neither tests generalization beyond the training distribution.

### Changes

**Held-out eval set (`ruin_mono.wav`):** 60 seconds trimmed from `ruin.wav` (0:46-1:46), converted to mono at 48kHz. Never used in training or validation. Static filtered versions generated by `moogGen` at all cutoff frequencies. LFO versions generated by `sweep_ref` at 1, 2, 5, 10, and 20 Hz (100Hz-10kHz, resonance=0.5).

**Additional LFO rates for bench and testSound:** sweep_ref was run at 1, 2, 10, and 20 Hz for both bench and testSound, giving a full modulation rate grid for future training and validation use.

**`--dry` flag added to both eval scripts:** `eval_param_model.py` now accepts `--dry <path>` and derives the wet directory and filename pattern from the stem (e.g. `audio/ruin_mono.wav` -> `filteredOutput/ruin/ruin_mono_{freq}hz.wav`). `eval_dynamic.py` accepts `--dry` to override the hardcoded bench dry path. Output filenames include the dry stem when non-default.

### Results

Runs 16, 17, 19, 20, and 23 were evaluated on ruin_mono. Static and dynamic (1, 2, 5, 10, 20 Hz LFO) results:

**Static ESR (dB):**

| Freq | Run 16 | Run 17 | Run 19 | Run 20 | Run 23 |
|------|-------:|-------:|-------:|-------:|-------:|
| 100 Hz | -28.8 | -24.8 | -30.1 | -36.7 | -29.7 |
| 250 Hz | -34.1 | -32.8 | -37.2 | -42.8 | -36.9 |
| 500 Hz | -40.0 | -37.7 | -43.6 | -46.5 | -41.9 |
| 1 kHz | -43.4 | -40.4 | -46.2 | -47.7 | -45.4 |
| 16 kHz | -48.3 | -45.6 | -54.2 | -56.7 | -51.6 |
| 20 kHz | -47.4 | -42.4 | -52.1 | -58.1 | -52.9 |

**Dynamic ESR (dB):**

| Rate | Run 16 | Run 17 | Run 19 | Run 20 | Run 23 |
|------|-------:|-------:|-------:|-------:|-------:|
| 1 Hz | -39.1 | -36.0 | -40.1 | -45.0 | -40.8 |
| 2 Hz | -39.2 | -34.7 | -37.9 | -42.6 | -40.4 |
| 5 Hz | -38.6 | -30.7 | -32.7 | -36.7 | -38.9 |
| 10 Hz | -37.3 | -25.8 | -27.4 | -31.2 | -36.1 |
| 20 Hz | -33.6 | -20.3 | -22.0 | -25.5 | -31.8 |

### Analysis

Results are highly consistent with bench evals. 100Hz static is 4-7 dB worse on ruin across all models, likely due to richer low-frequency content in that audio. Everything from 500Hz up is within 1-2 dB of bench. Dynamic rankings are identical. The bench eval numbers were not artificially inflated by the val/eval overlap -- the models generalize cleanly to new audio.

Run 23 remains the best all-around model: matches run 19 statically from 500Hz up, and holds within 7 dB of its 5Hz performance all the way to 20Hz where runs 19 and 20 have collapsed to -22 and -25 dB.

---

## Run 24 (LFO-only training, 128 units, 256 warmup)

### Motivation

Run 23 showed variable training data fixes fast-LFO degradation, but the training set still included static cutoffs. The question is whether static examples are necessary at all, or whether LFO sweeps alone are sufficient. An LFO-only model would have no gradient signal pushing it toward stable static-cutoff behavior, which is the hypothesis for why larger static-trained models degrade under fast modulation.

### Setup

`tensor_torch_lfo_only.py`: no static training data. Training set is five LFO rates (1, 2, 5, 10, 20 Hz) from `testSound_mono.wav`, all sweeping 100Hz to 10kHz at resonance=0.5. Validation is bench static + bench 5Hz LFO, same as run 23. Architecture and hyperparameters unchanged (128 GRU units, 256 warmup).

### Results

**Static ESR (dB) -- bench / ruin:**

| Freq | Run 23 bench | Run 24 bench | Run 23 ruin | Run 24 ruin |
|------|-------------:|-------------:|------------:|------------:|
| 100 Hz | -34.6 | -34.0 | -29.7 | -30.7 |
| 500 Hz | -42.5 | -37.6 | -41.9 | -36.7 |
| 1 kHz | -45.0 | -38.6 | -45.4 | -38.6 |
| 8 kHz | -46.3 | -42.1 | -47.6 | -42.9 |
| 12 kHz | -48.0 | -31.0 | -48.8 | -30.0 |
| 16 kHz | -51.1 | -20.0 | -51.6 | -18.8 |
| 20 kHz | -52.2 | -16.4 | -52.9 | -15.2 |

**Dynamic ESR (dB) -- ruin:**

| Rate | Run 23 | Run 24 |
|------|-------:|-------:|
| 1 Hz | -40.8 | -38.4 |
| 2 Hz | -40.4 | -38.5 |
| 5 Hz | -38.9 | -38.4 |
| 10 Hz | -36.1 | -38.2 |
| 20 Hz | -31.8 | -37.6 |

### Analysis

**Static collapse above 10kHz.** The LFO training data only sweeps 100Hz to 10kHz. The model has no examples of any static or dynamic behavior above that. 12kHz accuracy drops to -31 dB, 16kHz to -20 dB, 20kHz to -16 dB -- well into audible error territory. This is a hard frequency ceiling imposed by the training data range.

**Rate-invariant dynamic behavior.** Run 24 scores between -37.6 and -38.7 dB across all LFO rates (1 to 20 Hz) on both bench and ruin -- a spread of about 1 dB. Run 23 spans 9 dB over the same range (-40.4 at 2Hz, -31.8 at 20Hz). The LFO-only model has essentially no rate sensitivity; it tracks fast and slow modulation equally well because it was trained on nothing else.

**Below 10kHz, static accuracy is intact but reduced.** At 100Hz run 24 matches run 23 (-34 vs -34.6 dB). From 500Hz to 8kHz it is 3-6 dB worse than run 23. The model can learn static cutoff behavior from LFO data alone, but with reduced precision since no window ever holds a constant cutoff.

**The tradeoff.** LFO-only training eliminates rate sensitivity entirely but sacrifices high-frequency accuracy due to the 10kHz LFO ceiling. Two practical fixes: (1) extend the LFO sweep range to 20kHz; (2) mix in a small number of high-frequency static windows to anchor the model above the LFO ceiling without re-introducing rate-sensitivity bias.

---

## Current state

- Run 23 (128 units, static + 5Hz LFO training) is the best all-around model: strong static accuracy, +10 dB on fast LFO vs static-only training, 6-voice polyphony
- Run 24 (128 units, LFO-only training, 100-10kHz) is fully rate-invariant dynamically but collapses above 10kHz due to the LFO frequency ceiling
- Run 19 (128 units, static-only training) remains the static accuracy reference
- Run 16 (64 units, static-only training) is the most dynamically stable static-trained model
- Run 20 (256 units) is better statically but limited to single voice
- Held-out eval set (ruin_mono.wav) confirms bench eval numbers are not inflated -- results are consistent across both datasets

---

## Next steps

- **Extend LFO range to 20kHz** and retrain LFO-only model to eliminate the frequency ceiling. Worth combining with a static/LFO mix to anchor high-frequency behavior.
- **Paper**: write up for the Csound conference -- architecture decisions, ablation results, warmup analysis, opcode implementation, dynamic eval, FiLM experiment results, and variable training data findings.
