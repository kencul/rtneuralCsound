# rtneuralCsound

Research into neural network audio effect modeling using RTNeural, working toward a Csound opcode implementation. Trains models that can simulate analog filters (particularly Moog ladder filters) with real-time controllable parameters: cutoff frequency, resonance, etc.

Currently the project can train a Conv1d→GRU→Dense neural network (with skip connection and optional `knob_to_h0` seeding) to model a 4-pole Moog low-pass filter. The trained model runs as a C++ inference tool on WAV files, and as a real-time Csound opcode plugin.

## Repo layout

```
├── audio/              # WAV files for training and evaluation
├── moogGen/            # C++ tool for generating Moog filter training data
├── python/             # Training scripts, evaluation, spectrum analysis
│   ├── tensor_torch.py          # PyTorch training (static model)
│   ├── tensor_torch_param.py    # PyTorch training (parameterized model)
│   ├── eval_param_model.py      # Evaluate a trained model across cutoff frequencies
│   └── compareSpectrum.py       # Spectrogram comparison tool
├── ref/                # Archived trained models by version
├── research/           # Notes and paper references
├── src/                # C++ source
│   ├── csound_opcode/            # Csound plugin opcode (moognn.dll)
│   ├── process_wav/              # RTNeural JSON model (legacy)
│   ├── process_wav_torch/        # PyTorch model, stereo
│   └── process_wav_torch_param/  # PyTorch model with cutoff parameter
├── csound/             # Csound test scripts
│   ├── test_passthrough.csd  # File playback through opcode
│   └── test_midi_saw.csd     # Live MIDI with CC-controlled cutoff
├── vendor/             # Dependencies
│   ├── MoogLadders/    # Moog ladder filter implementations (third-party)
│   │   ├── src/        # Filter model headers
│   │   └── example/    # Example files from the MoogLadders repo
│   ├── RTNeural/       # Git submodule -- real-time neural network inference
│   ├── csound/         # Git submodule -- Csound headers for opcode build
│   └── dr_wav.h        # Single-header WAV reader/writer
├── CMakeLists.txt
└── requirements.txt
```

## Dependencies

C++ build uses vendored libraries in `vendor/`:

- **RTNeural** (`vendor/RTNeural/`): real-time neural network inference. Git submodule. [RTNeural Github](https://github.com/jatinchowdhury18/RTNeural)
- **csound** (`vendor/csound/`): Csound source, used only for headers when building the opcode plugin. Git submodule. Note: this is a large repo (~500MB). [Csound Github](https://github.com/csound/csound)
- **dr_wav** (`vendor/dr_wav.h`): single-header WAV reader/writer. [dr_wav.h in dr_libs repo](https://github.com/mackron/dr_libs/blob/master/dr_wav.h )
- **MoogLadders** (`vendor/MoogLadders/`): collection of Moog ladder filter implementations. `moogGen` uses `RKSimulationModel` from `src/` for generating training data. [MoogLadders Github](https://github.com/ddiakopoulos/MoogLadders)

## Build

```bash
git submodule update --init
cmake -Bbuild
cmake --build build --config Release
```

Binaries output to `build/bin/Release/`.

To build only the Csound opcode DLL:

```bash
cmake --build build --config Release --target moognn
# output: build/bin/Release/moognn.dll
```

The opcode is loaded at runtime by Csound via `--opcode-lib=build/bin/Release/moognn.dll`. To install permanently, copy `moognn.dll` to `C:/Program Files/Csound7/plugins64/`.

## Usage

### C++ inference tools

```bash
# RTNeural JSON model (legacy, static)
build/bin/Release/process_wav <model.json> <input.wav> <output.wav>

# PyTorch model, stereo (static)
build/bin/Release/process_wav_torch <model.json> <input.wav> <output.wav>

# PyTorch model with cutoff parameter
build/bin/Release/process_wav_torch_param <model.json> <input.wav> <output.wav> <cutoff_hz>
```

### Training

Set up a Python venv:

```bash
python -m venv env
source env/Scripts/activate    # Windows
# source env/bin/activate       # Mac/Linux
pip install -r requirements.txt
# Install PyTorch from https://pytorch.org/get-started/locally/
```

Train a parameterized Moog filter model:

```bash
python python/tensor_torch_param.py
```

Evaluate a trained model across cutoff frequencies:

```bash
python python/eval_param_model.py best_model_param.pt
```

### Generating training data

The `moogGen/` tool filters a WAV file at multiple static cutoff frequencies using a 4-pole Moog ladder implementation with configurable oversampling:

```bash
cd moogGen
cmake -Bbuild
cmake --build build --config Release
build/Release/moogGen.exe <input.wav>
```

## Model architecture

The architecture is still being evaluated through ablation studies. The diagram below shows the base structure; unit count, LayerNorm, and knob_to_h0 are all variables under active experimentation. See [devlog/DEVLOG.md](devlog/DEVLOG.md) for the full ablation results.

```
Input audio (1 sample) ──┐
   │                     │
Conv1d (16ch, k=31)      │ skip connection
   │                     │
concat(knob value)       │
   │                     │
GRU (32 hidden units)    │
   │                     │
Dense (1 output)  ───────┼── + ──→ Output
```

All layers (Conv1d, GRU, Dense) are natively supported by RTNeural, meaning the model can be deserialized directly from JSON without custom inference code, and can use RTNeural's templated static graph mode for optimal performance.

An optional `knob_to_h0` layer seeds the GRU initial hidden state from the cutoff value, which helps especially at low cutoff frequencies.

## DEVLOG

The full development diary — every experiment, failure, breakthrough, and lesson learned — lives in [devlog/DEVLOG.md](devlog/DEVLOG.md). It starts with the TensorFlow baseline, covers the switch to PyTorch, GPU training, gradient explosions, windowed training with warmup, the Moog filter data pipeline, the parameterized model saga (including the infamous bad 16kHz training data), ablation studies, and the ongoing work toward the Csound opcode and dynamic parameter handling.

## Next steps

- Dynamic cutoff testing -- the model has only been evaluated on static cutoffs so far
- ~~Csound opcode implementation (must-have for the conference paper)~~ -- done, see `src/csound_opcode/`
- ~~Architecture sweet spot -- 64 GRU units with `knob_to_h0`~~ -- trained (run 14), ablation complete
- Variable-parameter training data with sweeps and modulation
- Fix per-note click on cold start without an always-on send effect workaround


## References

https://medium.com/data-science/mini-neural-nets-for-guitar-effects-with-microcontrollers-ea9cdad2a29c

https://github.com/jatinchowdhury18/RTNeural
