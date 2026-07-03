# rtneuralCsound

Research into neural network audio effect modeling using RTNeural, working toward a Csound opcode implementation. Trains a Conv1d+GRU+Dense network to model a 4-pole Moog ladder low-pass filter with a real-time controllable cutoff, then runs it as a Csound plugin opcode.

## Repo layout

```
├── audio/
│   ├── bench_mono.wav                  # Moog: validation audio
│   ├── testSound_mono.wav              # Moog: training audio
│   ├── ruin_mono.wav                   # Moog: held-out eval audio
│   ├── filteredOutput/
│   │   ├── bench/                      # Moog-filtered outputs for validation
│   │   ├── testSound/                  # Moog-filtered outputs for training
│   │   └── ruin/                       # Moog-filtered outputs for held-out eval
│   └── updatedDistortion/              # Distortion training/eval pairs (breadboard loopback recordings)
│       ├── trainingDry.wav / trainingWet.wav
│       ├── training-10dBDry.wav / training-10dBWet.wav
│       ├── benchDry.wav / benchWet.wav         # Held-out eval pair
│       └── bench-10dBDry.wav / bench-10dBWet.wav
├── models/                             # Trained model checkpoints by run number
├── python/
│   ├── model_concat.py                 # Knob-concatenation model architecture (Moog)
│   ├── model_film.py                   # FiLM conditioning model architecture (pre- and post-GRU)
│   ├── model_distortion.py             # Non-parametric distortion model architecture
│   ├── tensor_torch_param.py           # Training script (concat architecture, static data)
│   ├── tensor_torch_film.py            # Training script (FiLM architecture)
│   ├── tensor_torch_variable.py        # Training script (concat architecture, static + LFO data)
│   ├── tensor_torch_distortion.py      # Training script (distortion model)
│   ├── eval_param_model.py             # Static cutoff evaluation
│   ├── eval_dynamic.py                 # Dynamic cutoff evaluation (sweep/LFO)
│   ├── eval_distortion.py              # Distortion model evaluation
│   ├── eval_asr.py                     # Aliasing-to-Signal Ratio measurement
│   ├── align_audio.py                  # Dry/wet latency alignment tool
│   └── compareSpectrum.py              # Spectrogram comparison tool
├── src/
│   ├── moogGen/
│   │   ├── main.cpp                    # moogGen: static training data generator
│   │   └── sweep_ref.cpp               # sweep_ref: dynamic cutoff reference generator
│   └── csound_opcode/
│       ├── moognn.cpp                  # Csound plugin opcode (moognn.dll)
│       ├── distnn.cpp                  # Csound plugin opcode (distnn.dll)
│       └── rkmoog.cpp                  # Csound plugin opcode (rkmoog.dll) - RK4 reference
├── csound/
│   ├── test_passthrough.csd            # File playback through moognn with cutoff sweep
│   ├── test_midi_saw.csd               # Live MIDI with CC-controlled cutoff
│   ├── test_distnn.csd                 # File playback through distnn with wet/dry sweep
│   └── test_midi_distnn.csd            # Live MIDI through distnn
├── vendor/
│   ├── RTNeural/                       # Git submodule: real-time neural inference
│   ├── MoogLadders/                    # Moog ladder filter reference implementations
│   ├── csound/                         # Git submodule: Csound headers for opcode build
│   └── dr_wav.h                        # Single-header WAV I/O
├── devlog/                             # Full experiment diary
├── research/                           # Paper notes and references
├── CMakeLists.txt
└── requirements.txt
```

## Dependencies

- **RTNeural** (`vendor/RTNeural/`): real-time neural network inference. [Github](https://github.com/jatinchowdhury18/RTNeural)
- **MoogLadders** (`vendor/MoogLadders/`): Moog ladder filter implementations. `moogGen` and `sweep_ref` use `RKSimulationModel`. [Github](https://github.com/ddiakopoulos/MoogLadders)
- **dr_wav** (`vendor/dr_wav.h`): single-header WAV reader/writer. [Github](https://github.com/mackron/dr_libs/blob/master/dr_wav.h)
- **csound** (`vendor/csound/`): headers only, for building the opcode DLL. Large repo (~500MB). [Github](https://github.com/csound/csound)

## Build

```bash
git submodule update --init
cmake -Bbuild
cmake --build build --config Release
```

Binaries output to `build/bin/Release/`.

To build individual targets:

```bash
cmake --build build --config Release --target moogGen
cmake --build build --config Release --target sweep_ref
cmake --build build --config Release --target moognn
```

The opcode DLL is loaded by Csound via `--opcode-lib=build/bin/Release/moognn.dll`. To install permanently, copy it to `C:/Program Files/Csound7/plugins64/`.

## Generating training and reference data

### moogGen -- static cutoff training data

Filters a WAV through the RK Moog ladder at a fixed grid of cutoff frequencies. Output files go in `audio/filteredOutput/bench/` and are used as training targets by `tensor_torch_param.py` and as ground truth by `eval_param_model.py`.

```bash
build/bin/Release/moogGen.exe -f audio/bench_mono.wav -o audio/filteredOutput/bench
```

Generates one WAV per cutoff: `bench_mono_100hz.wav`, `bench_mono_1000hz.wav`, etc. Run against each audio file (`bench_mono.wav`, `testSound_mono.wav`, `ruin_mono.wav`) and direct output to the matching `filteredOutput/` subdirectory.

### sweep_ref -- dynamic cutoff reference

Filters a WAV through the same RK Moog with a time-varying cutoff. Writes a float32 reference WAV and a companion CSV of per-sample normalized knob values (0-1, same log scaling the model uses). The CSV is read by `eval_dynamic.py` to drive the neural model through the exact same cutoff trajectory for comparison.

```bash
# Exponential sweep from 100Hz to 20kHz
build/bin/Release/sweep_ref.exe audio/bench_mono.wav out.wav out.csv log 100 20000 [resonance=1.0]

# Sinusoidal LFO between 100Hz and 10kHz at 2Hz
build/bin/Release/sweep_ref.exe audio/bench_mono.wav out.wav out.csv lfo 100 10000 [resonance=1.0] [lfo_rate_hz=1.0]
```

Pre-generated reference files exist for bench, testSound, and ruin at LFO rates of 1, 2, 5, 10, and 20 Hz (100Hz-10kHz, resonance=0.5). bench also has a 0.25Hz slow LFO and a 40-second log sweep.

## Training

Set up a Python venv:

```bash
python -m venv env
source env/Scripts/activate    # Windows
pip install -r requirements.txt
# Install PyTorch separately: https://pytorch.org/get-started/locally/
```

Train:

```bash
# Static training data only (runs 16-20)
python python/tensor_torch_param.py models/my_run

# Static + LFO variable training data (run 23+)
python python/tensor_torch_variable.py models/my_run

# FiLM architecture (runs 21-22)
python python/tensor_torch_film.py models/my_run
```

Architecture and hyperparameters (`GRU_HIDDEN`, `warmup_size`, `CUTOFF_FREQS`) are set at the top of each training script. Checkpoints embed `arch`, `gru_hidden`, `freq_min`, and `freq_max` so eval scripts auto-detect the model configuration.

`tensor_torch_variable.py` trains on static cutoffs from `testSound` plus LFO-swept targets from `testSound` and validates against static bench cutoffs plus the bench 5Hz LFO. Additional LFO rates can be added to `VARIABLE_TRAIN_FILES` or `VARIABLE_VAL_FILES`.

## Evaluation

Both eval scripts produce a 4-panel plot (reference spectrogram, model output spectrogram, difference spectrogram, ESR metric) and print results to stdout. Pass `--help` to either script for full usage.

### Static cutoff eval

Runs the model at a grid of fixed cutoff frequencies and reports ESR for each. The fourth panel shows ESR vs frequency on a log scale.

```bash
python python/eval_param_model.py <model.pt> [warmup] [--dry <path>] [--save <dir>] [--show]
```

```bash
# Default: evaluates against bench
python python/eval_param_model.py models/my_run/best_model.pt

# Held-out eval set
python python/eval_param_model.py models/my_run/best_model.pt --dry audio/ruin_mono.wav
```

The `--dry` flag sets the dry audio file. The wet directory and filename pattern are derived from the stem automatically (e.g. `audio/ruin_mono.wav` resolves to `audio/filteredOutput/ruin/ruin_mono_{freq}hz.wav`). Output files include the stem suffix when non-default (`evalOutput_ruin_mono.txt`).

### Dynamic cutoff eval

Runs the model with a time-varying knob schedule from a sweep_ref CSV and compares against the companion reference WAV. The fourth panel shows windowed ESR over time (0.5s windows).

```bash
python python/eval_dynamic.py <model.pt> <ref.wav> <ref.csv> [warmup] [--dry <path>] [--save <dir>] [--show]
```

```bash
# Default: uses bench_mono.wav as dry signal
python python/eval_dynamic.py models/my_run/best_model.pt \
    audio/filteredOutput/bench/bench_mono_lfo_fast_100-10khz.wav \
    audio/filteredOutput/bench/bench_mono_lfo_fast_100-10khz.csv

# Held-out eval set
python python/eval_dynamic.py models/my_run/best_model.pt \
    audio/filteredOutput/ruin/ruin_mono_lfo_5hz_100-10khz.wav \
    audio/filteredOutput/ruin/ruin_mono_lfo_5hz_100-10khz.csv \
    --dry audio/ruin_mono.wav
```

### Flags (both scripts)

| Flag | Description |
|------|-------------|
| `--dry <path>` | Dry audio file. Defaults to `audio/bench_mono.wav`. |
| `--save <dir>` | Write eval output to `<dir>`. Static eval writes `evalOutput[_stem].txt/.png`. Dynamic eval uses sweep-specific names so multiple sweeps can share a model directory. |
| `--force` | Overwrite existing output files without prompting. Useful for batch runs. |
| `--show` | Open the interactive plot window. |
| `--help`, `-h` | Print usage. |

## Model architecture

Two architectures are implemented in `python/model_concat.py` and `python/model_film.py`.

**Concatenation** (`model_concat.py`) — runs 11-20, current deployed architecture:
```
Input audio ──┐
   |          |
Conv1d (16ch) | skip connection
   |          |
concat(knob)  |
   |          |
GRU           |
   |          |
Dense ────────+──> Output
```

**FiLM** (`model_film.py`) — runs 21-22, experiments complete:
```
Input audio ──┐
   |          |
Conv1d (16ch) | skip connection
   |          |
FiLM(knob)    |  (scale + shift applied to conv features, pre-GRU)
   |          |
GRU           |
   |          |
Dense ────────+──> Output
```

Pre-GRU FiLM (run 22) matched concat at the same unit count statically but did not improve dynamic tracking under fast modulation. Post-GRU FiLM (run 21) failed to learn. Concat remains the architecture of record.

The knob is log-normalized to [0, 1] over 100Hz–20kHz. The deployed model is run 23 (`model_concat`, 128 GRU units, 256-sample warmup, variable training data). See [devlog/DEVLOG2.md](devlog/DEVLOG2.md) for the full experiment history.

## Csound opcode

Opcode signature: `aout moognn{N} ain, Spath, kcutoff`

`moognn.dll` registers four size variants. The opcode name must match the GRU hidden size of the model in `Spath`; loading a mismatched JSON segfaults inside RTNeural's loader.

| Opcode | GRU units | Example model |
|--------|-----------|---------------|
| `moognn32`  | 32  | `models/25_moog_100-20k_32u/weights.json` |
| `moognn64`  | 64  | `models/16_moog_100-20k_64u_w256/weights.json` |
| `moognn128` | 128 | `models/23_moog_variable_128u_w256/weights.json` (deployed) |
| `moognn256` | 256 | `models/26_moog_100-20k_256u/weights.json` |

Arguments:
- `ain`: audio input
- `Spath`: path to the model JSON weights file
- `kcutoff`: cutoff frequency in Hz (k-rate)

`moognn_preload Spath` is shared across all sizes (one JSON cache). Call at score time 0 to pre-cache the weights before the first note fires.

Benchmark (Ryzen 5600X, 48kHz, ksmps=64):

| Opcode | CPU% / voice | Voices at realtime |
|--------|-------------|-------------------|
| `moognn32` | 5.9% | 16 |
| `moognn64` | 8.6% | 11 |
| `moognn128` | 16.9% | 6 |
| `moognn256` | 60.0% | 1 |

See `csound/test_passthrough.csd` and `csound/test_midi_saw.csd` for working examples (both use `moognn128`).

## RK reference opcode

Opcode signature: `aout rkmoog ain, kcutoff, kresonance`

Wraps `RKSimulationMoog` (RK4 integration, 8x oversampling — the same algorithm used to generate moognn training data) as a Csound opcode. Intended for apples-to-apples CPU benchmarking against moognn: identical Csound frame dispatch, k-rate cutoff updates, and MYFLT conversion overhead. No weights file; state resets to zero on note-on.

- `ain`: audio input
- `kcutoff`: cutoff frequency in Hz (k-rate)
- `kresonance`: resonance 0–10 (k-rate); 0.5 matches training data

Benchmark (Ryzen 5600X, 48kHz, ksmps=64): 2.6% CPU/voice, 39 voices at realtime. For comparison, `moogladder` (Csound built-in, Huovilainen DAFx 2004) costs 0.7% CPU/voice and sustains 141 voices.

## Distortion opcode

Opcode signature: `aout distnn ain, Spath, kmix`

- `ain`: audio input
- `Spath`: path to the model JSON weights file
- `kmix`: wet/dry blend, k-rate (0 = dry, 1 = fully wet)

Deployed model: `models/dist_12_gru128_l1x10/weights.json` (GRU 128 units, breadboard loopback recordings, L1 + MR-STFT loss with L1_WEIGHT=10). Held-out bench ESR: -29.2 dB. ASR mean: -42 dB (aliasing not a significant factor).

Use `distnn_preload Spath` at score time 0 to pre-cache the JSON before the first note fires. See `csound/test_distnn.csd` and `csound/test_midi_distnn.csd` for working examples.

Benchmark (Ryzen 5600X, 48kHz, ksmps=64): 16.5% CPU/voice, 6 voices at realtime — identical cost to `moognn128`. A 256u model (`dist_14_gru256_l1x10`, -30.2 dB held-out) is available for single-voice use at roughly 60% CPU.

## Distortion modeling

Second modeling target: a hardware breadboard diode distortion effect. Unlike the Moog filter, distortion is nonlinear and content-dependent. The architecture is the same Conv1d+GRU+Dense+skip structure but without the knob input channel (`model_distortion.py`).

### Preparing training data

Hardware recordings must be latency-corrected before training. When audio is played through the pedal and recorded back in, the ADC/DAC round-trip of the interface introduces a fixed sample offset between the dry and wet files. Use `align_audio.py` to measure and correct this:

```bash
# Measure the lag (no --save: report only)
python python/align_audio.py <dry> <wet>

# Write corrected wet file (overwrites target path)
python python/align_audio.py <dry> <wet> --save <output_wet_path>

# Resample both to 48kHz before aligning (for recordings captured at a different rate)
python python/align_audio.py <dry> <wet> --sr 48000 --save <output_wet_path>
```

The script uses cross-correlation on the first 5 seconds to find the lag. If wet is delayed (positive lag), it trims the start of wet. If wet is early (negative lag), it prepends silence. Dry is never modified. The offset is fixed for a given interface configuration, so measure once per session and apply to all pairs from that session.

### Training

```bash
python python/tensor_torch_distortion.py models/dist_my_run [gru_hidden] [l1_weight]
```

`l1_weight` scales the time-domain L1 term relative to the MR-STFT term (default 1.0). Without this weighting, large models exploit the phase-agnostic nature of MR-STFT and converge to spectrally plausible but waveform-incoherent solutions. A value of 10 was found to fix this for 128u and 256u models; 64u models do not need it (limited capacity provides implicit regularization). Setting `l1_weight > 1.0` automatically lowers LR, clip norm, and scheduler patience to compensate for the increased gradient scale.

Dry/wet paths are set near the top of the script. Training on diverse audio sources is important: a diode clipper's output depends on the instantaneous waveform, so a model trained on one audio source does not generalize to audio with different frequency content or dynamics. See DEVLOG2.md for the full experiment history.

### Evaluation

```bash
python python/eval_distortion.py <model.pt> [warmup] [--dry <path>] [--wet <path>] [--save <dir>] [--show]
```

## Next steps

- **Paper**: Csound conference writeup
- Moog: extend LFO training range to 20kHz and retrain LFO-only model (run 24 collapses above 10kHz due to the sweep ceiling); consider mixing with static high-frequency windows to anchor the model above the LFO range
- ~~Update `distnn` opcode to deploy `dist_12_gru128_l1x10` weights~~ -- done
- ~~Run full benchmark matrix (`bench/run_benchmarks.py`) and generate paper figures~~ -- done; see `bench/results/`
- ~~Variable-parameter training data~~ -- complete (run 23); +10 dB on fast LFO
- ~~FiLM conditioning experiment~~ -- complete (runs 21-22); pre-GRU FiLM matched concat statically but did not improve dynamic tracking
- ~~Csound opcode implementation~~ -- done, see `src/csound_opcode/`
- ~~Architecture sweet spot -- 64 GRU units with `knob_to_h0`~~ -- trained (run 14), ablation complete
- ~~Dynamic cutoff eval~~ -- done, see `eval_dynamic.py`

## References

- [RTNeural](https://github.com/jatinchowdhury18/RTNeural)
- [devlog/DEVLOG2.md](devlog/DEVLOG2.md)
https://medium.com/data-science/mini-neural-nets-for-guitar-effects-with-microcontrollers-ea9cdad2a29c