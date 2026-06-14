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

## Phase 7: Run 15 evaluation and warmup accuracy ceiling

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

---

## Current state

- Opcode is functional and click-free with run 16 weights at a 256-sample fade-in
- Run 16 is the best short-fade model: 100Hz lower boundary, strict improvement over run 13
- Run 14 remains the highest-accuracy model overall but requires a 2048-sample fade

---

## Next steps

- **More GRU units (run 17)**: 128 units + 100Hz floor + 256 warmup, no k2h0. One change at a time.
- **Dynamic cutoff testing**: the model has only been evaluated on static cutoffs. Write a script that sweeps the cutoff mid-file and compares against the real Moog filter.
- **Variable-parameter training data**: LFO-modulated cutoff sweeps as training targets to improve dynamic tracking.
- **Paper**: write up for the Csound conference — architecture decisions, ablation results, and opcode implementation.
