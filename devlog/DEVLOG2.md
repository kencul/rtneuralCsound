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

## Current state

- Opcode is functional and click-free using run 14 weights with a 2048-sample fade-in
- `moognn_preload` eliminates the first-note JSON latency dropout
- The opcode has been tested in both file-playback and live MIDI contexts
- Run 15 in progress, targeting a shorter fade-in while keeping k2h0 accuracy

---

## Next steps

- **Re-test earlier models**: now that the fade-in bug is fixed, verify whether runs 11–13 (no k2h0) still click. If not, the architectural motivation for run 15 changes.
- **Evaluate run 15**: does k2h0 + shorter warmup give a noticeably shorter fade-in compared to run 14?
- **Dynamic cutoff testing**: the model has only been evaluated on static cutoffs. Write a script that sweeps the cutoff mid-file and compares against the real Moog filter.
- **Variable-parameter training data**: LFO-modulated cutoff sweeps as training targets to improve dynamic tracking.
- **Paper**: write up for the Csound conference — architecture decisions, ablation results, and opcode implementation.
