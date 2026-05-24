# moogGen

A command-line tool that runs a WAV file through a Moog ladder filter simulation at a bunch of preset cutoff frequencies, spitting out one file per frequency. Useful for generating training data pairs (dry input → filtered output) for neural network models.

## Building

```bash
g++ -std=c++17 -O2 main.cpp src/RKSimulationModel.h -o moogGen
```

## Usage

```bash
./moogGen -f <input.wav> [-o <output_dir>]
```

- `-f` / `--file` — the input WAV file (required)
- `-o` / `--output` — folder to write the output files into (optional, defaults to current directory)

The output directory is created automatically if it doesn't exist.

## Example

```bash
./moogGen -f ../bench_mono.wav -o output/bench
```

This produces 15 filtered WAV files, one for each cutoff frequency:

```
output/bench/bench_mono_20hz.wav
output/bench/bench_mono_60hz.wav
output/bench/bench_mono_100hz.wav
...
output/bench/bench_mono_24000hz.wav
```

## Cutoff Frequencies

The filter runs at these fixed cutoffs (Hz):

`20, 60, 100, 125, 250, 500, 800, 1000, 2000, 4000, 8000, 12000, 16000, 20000, 24000`

Resonance is fixed at `0.5`.

## Notes

- The filter model is an RK4 numerical simulation of the Moog ladder circuit — it's not a biquad approximation, so it sounds pretty authentic.
- Input files should be mono. Stereo files will work but the filter processes the interleaved samples as a flat buffer, which will sound wrong.
- Sample rate is read from the WAV file header automatically.
