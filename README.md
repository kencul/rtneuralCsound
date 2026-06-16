# rtneuralCsound

Research into neural network audio effect modeling using RTNeural, working toward a Csound opcode implementation. Trains a Conv1d+GRU+Dense network to model a 4-pole Moog ladder low-pass filter with a real-time controllable cutoff, then runs it as a Csound plugin opcode.

## Repo layout

```
├── audio/
│   ├── bench_mono.wav                  # Primary training/eval input
│   └── filteredOutput/bench/           # Moog-filtered outputs (static and dynamic)
├── models/                             # Trained model checkpoints by run number
├── python/
│   ├── tensor_torch_param.py           # Training script (parameterized model)
│   ├── eval_param_model.py             # Static cutoff evaluation
│   └── compareSpectrum.py              # Spectrogram comparison tool
├── src/
│   ├── moogGen/
│   │   ├── main.cpp                    # moogGen: static training data generator
│   │   └── sweep_ref.cpp               # sweep_ref: dynamic cutoff reference generator
│   └── csound_opcode/
│       └── moognn.cpp                  # Csound plugin opcode (moognn.dll)
├── csound/
│   ├── test_passthrough.csd            # File playback through opcode with cutoff sweep
│   └── test_midi_saw.csd               # Live MIDI with CC-controlled cutoff
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

Generates one WAV per cutoff: `bench_mono_100hz.wav`, `bench_mono_1000hz.wav`, etc.

### sweep_ref -- dynamic cutoff reference

Filters a WAV through the same RK Moog with a time-varying cutoff. Writes a float32 reference WAV and a companion CSV of per-sample normalized knob values (0-1, same log scaling the model uses). The CSV is read by `eval_dynamic.py` to drive the neural model through the exact same cutoff trajectory for comparison.

```bash
# Exponential sweep from 100Hz to 20kHz
build/bin/Release/sweep_ref.exe audio/bench_mono.wav out.wav out.csv log 100 20000 [resonance=1.0]

# Sinusoidal LFO between 100Hz and 10kHz at 2Hz
build/bin/Release/sweep_ref.exe audio/bench_mono.wav out.wav out.csv lfo 100 10000 [resonance=1.0] [lfo_rate_hz=1.0]
```

Pre-generated reference files in `audio/filteredOutput/bench/`:

| File | Type | Range | Rate |
|------|------|-------|------|
| `bench_mono_sweep_log_100-20000hz` | log sweep | 100Hz to 20kHz | once over 40s |
| `bench_mono_lfo_slow_100-10000hz` | LFO | 100Hz to 10kHz | 0.25Hz (4s period) |
| `bench_mono_lfo_fast_100-10000hz` | LFO | 100Hz to 10kHz | 5Hz (0.2s period) |

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
python python/tensor_torch_param.py models/my_run
```

Evaluate at static cutoffs:

```bash
python python/eval_param_model.py models/my_run/best_model.pt [warmup_samples] [freq_min]
```

## Model architecture

```
Input audio (1 sample) ──┐
   |                     |
Conv1d (16ch, k=31)      |
   |                     | skip connection
LayerNorm                |
   |                     |
concat(knob 0-1)         |
   |                     |
GRU (128 units)          |
   |                     |
Dense (1 output) ────────+──> Output
```

The knob input is a log-normalized value in [0, 1] mapping 100Hz to 20kHz. The deployed model is run 19: 128 GRU units, 256-sample training warmup, 100Hz frequency floor, no knob_to_h0. See [devlog/DEVLOG2.md](devlog/DEVLOG2.md) for the full experiment history and ablation results.

## Csound opcode

Opcode signature: `aout moognn ain, Spath, kcutoff`

- `ain`: audio input
- `Spath`: path to the model JSON weights file
- `kcutoff`: cutoff frequency in Hz (k-rate)

Use `moognn_preload Spath` at score time 0 to pre-cache the JSON before the first note fires.

See `csound/test_passthrough.csd` and `csound/test_midi_saw.csd` for working examples.

## Next steps

- Write `eval_dynamic.py` to compare neural model output against sweep_ref on the dynamic cutoff files
- FiLM conditioning experiment: replace knob concatenation with post-GRU FiLM, train at 64 units
- Paper: Csound conference writeup
- Dynamic cutoff testing -- the model has only been evaluated on static cutoffs so far
- ~~Csound opcode implementation (must-have for the conference paper)~~ -- done, see `src/csound_opcode/`
- ~~Architecture sweet spot -- 64 GRU units with `knob_to_h0`~~ -- trained (run 14), ablation complete
- Variable-parameter training data with sweeps and modulation
- Fix per-note click on cold start without an always-on send effect workaround

## References

- [RTNeural](https://github.com/jatinchowdhury18/RTNeural)
- [devlog/DEVLOG2.md](devlog/DEVLOG2.md)
https://medium.com/data-science/mini-neural-nets-for-guitar-effects-with-microcontrollers-ea9cdad2a29c