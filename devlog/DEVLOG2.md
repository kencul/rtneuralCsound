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

## Current state

- Run 19 (128 GRU units, 256 warmup, 100Hz floor) is the deployed model: 6-voice polyphony, best accuracy for real-time use
- Run 20 (256 units) achieves better accuracy but is limited to single voice
- The RTNeural build is fully optimized (xsimd + AVX2, pinned in CMakeLists.txt)
- Knob conditioning is the main architectural weakness: input concatenation is the worst method per literature
- Dynamic reference data is ready; eval script (`eval_dynamic.py`) is the remaining piece before dynamic cutoff behavior can be measured

---

## Next steps

- **Dynamic cutoff eval**: write `eval_dynamic.py` to run the neural model with the sweep/LFO knob schedules from the CSV and compare against the reference WAVs. Time-windowed ESR + spectrogram plot.
- **FiLM experiment**: modify `tensor_torch_param.py` to replace knob concatenation with post-GRU FiLM conditioning. Train at 64 units. Compare ESR against run 19. If accuracy is comparable, rebuild opcode with 64-unit GRU + FiLM layer and test polyphony.
- **Knowledge distillation**: if FiLM at 64 units falls short of run 19, add run 19 as teacher to the loss function. Low effort add-on to any new training run.
- **Variable-parameter training data**: LFO-modulated cutoff sweeps as training targets to improve dynamic tracking. Depends on dynamic cutoff test results to know if this is needed.
- **Paper**: write up for the Csound conference -- architecture decisions, ablation results, warmup analysis, and opcode implementation. Dynamic cutoff results and FiLM experiment would strengthen it but current material is already sufficient for submission.
