# rtneuralCsound — dev diary

Research into neural network audio effect modeling using RTNeural, working toward a Csound opcode implementation.

## Dependencies

C++ build uses two vendored libraries in `vendor/`:

- **RTNeural** (`vendor/RTNeural/`): real-time neural network inference. Git submodule, initialize with `git submodule update --init`.
- **dr_wav** (`vendor/dr_wav.h`): single-header WAV reader/writer.
- **moog_ladders** (`moogGen/src` and `moogGen/example`): collection of moog ladder implementations and helper code.

## Build

```bash
git submodule update --init
cmake -Bbuild
cmake --build build --config Release
```

Binaries output to `build/bin/Release/`. Usage:

```bash
# RTNeural JSON model (legacy)
build/bin/Release/process_wav <model.json> <input.wav> <output.wav>

# PyTorch model, stereo
build/bin/Release/process_wav_torch <model.json> <input.wav> <output.wav>

# PyTorch model with cutoff parameter
build/bin/Release/process_wav_torch_param <model.json> <input.wav> <output.wav> <cutoff_hz>
```

## Python environment

```bash
python -m venv env
source env/Scripts/activate  # Windows; use `source env/bin/activate` on Mac/Linux
pip install -r requirements.txt
# download pytorch here: https://pytorch.org/get-started/locally/
```

PyTorch is installed separately due to the CUDA index URL. See [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) for other CUDA versions.

---

## TensorFlow baseline

Made an input test file. Generated pink noise, sine sweeps, velocity burst, and white noise burst using python scripts mostly with scipy.

Compiled these audio files into a single wav file with Ableton Live.

Create venv to prepare to train model in tensorflow.

```bash
python -m venv env
```

Creates env dir.

Activate venv:

```bash
source env/Scripts/activate
```

Install dependencies:

```bash
pip install tensorflow librosa numpy
```

Processed test file with auto filter in Ableton with a lowpass with high res.

Trained a model with a vibe coded tensorflow script. 50 epocs on the CPU, super slow.

```bash
python python/tensor.py
```

Processed wav files with a cpp script, also vibe coded.

```bash
cmake -Bbuild -DBUILD_EXAMPLES=ON
cmake --build build --config Release
./build/examples_out/process_wav rtneural_model_weights.json python/testSound.wav output.wav
```

> Note: these paths are from the original RTNeural clone. See current build instructions above.

Results were convincing, though not completely accurate in the case of the test file, comparing to the true file.

One major flaw is the validation split function may bite me in the butt, considering all parts of the test file train different things. Missing 10% will mean the validation is on test material it never got to train on.

Furthermore, i need to figure out an alternative to be able to use my GPU. I need to see if its faster.

Options are: downgrade tensorflow, use tensorflow through WSL, or use pytorch.

## Switch to PyTorch, GPU training, architecture tuning

Upgraded CUDA toolkit, installed nvidia driver and installed latest pytorch for CUDA 13.2.

Created a python script to train using Pytorch. Had to learn the specifics of ML a bit to understand, as pytorch exposes the inner workings unlike tensorflow, which mostly does the dirty work for you.

Training with GPU makes a huge difference. Each epoch doesnt take even a second.

Created new process_wav_torch.cpp file that applies the model to a wav file that is compatible with pytorch models.

Quick test with the pytorch version of the process shows it works fundamentally.

Added a bench.wav and benchLPF.wav to use as the validation file, so the entirety of the test file can be used for training. The bench file is just a short clip of one of my songs.

I then tried with 300 epochs, as the epochs are faster. This caused a problem where the loss jumps randomly, and most epochs are wasted recovering from these explosions.

Supposedly, these are "gradient explosions" in the GRU. The solution is to add gradient clipping to prevent explosions.

I can also make the script save the best model, not just the final one. This is nice as the final model was actually worse than one in the middle.

This process saved a model with a val_loss of 0.0222.

To improve the training more, i made the window size bigger to 8192. This means the gru sees larger chunks of audio in one pass, giving more context to observe the impulse response of the high resonance.

Second, i increased the batch size to 64, as sub second epochs are too fast and the GPU sits idle between the short batches.

Finally, added a LR scheduler. This reduces the learning rate (how much the model changes per step) when the val_loss plateaus. This means the model is able to fine tune more accurately.

These changes combined improved the val_loss greatly down to 0.0079! This is mostly attributed to the increased window size.

I restructured `process_wav_torch.cpp` to process stereo audio files. A separate model is applied to each channel.

Listening through the processed audio file, its almost good, but there seems to be weird resonant explosions that aren't present in the reference file.

For more changes to the training, I added pre-emphasis. Because the LPF attenuates highs, the model focuses all its learning on the low frequencies where the energy and changes are. By adding pre-emphasis, the model is forced to get the filter rolloff curve right as well, not just the low frequencies. This is implemented by adding a simple high pass filter on the prediction and target before computing the loss, skewing the loss calculation to higher frequencies.

Although with pre-emphasis, the val_loss isn't directly comparable to previous values, this model got a val_loss of 0.0316.

I analyzed the spectrogram of the model output with `python\compareSpectrum.py`. It includes the benchmark and model output audio files, as well as the spectrum analysis png.

I noticed that there are bright spots under 32hz, which im not too concerned with.

Up until 512hz, there is a constant +8db error. Everything under 512hz is very dark blue, and there a very distinct line at 512hz, where its clearly darker above it.

Finally there's some sprinkles of error between 1024 and 2048 hz. There are some especially bright moments that i can notice audibly as sounding like really high resonance spikes.

4 changes to tackle the resonance issues:

- Remove the tanh activation. As a LPF is a linear effect, the tanh forces saturation that makes the GRU confused. tanh is useful when simulating a non linear effect.
- Increase the kernel size from 3 to something like 31. Resonance is created from feedback, so the model will need more past sample data to correctly simulate it.
- Add more GRU hidden units. More capacity should mean it can approximate more accurately.
- Fix the continuous GRU state between windows. Before, the GRU state was reset between every window. This is unrealistic, as when the model is run, it keeps the hidden state as it processes audio. To simulate this in the training, disable shuffling the windows. This has a big downside of reducing the batch size to 1, meaning the training is magnitudes slower.

To speed up this now slow training, i implemented a warm up system. By feeding the GRU some extra samples before the window (for instance 2048 samples before the window), the GRU can get warmed up to the correct state. The loss is calculated only on the actual 8192 sample window, not the 2048 sample warmup. This lets me crank up the batch size back up to 64 to train without starving the GPU. This is a strategy used for production audio ML and is a sound engineering tradeoff.

This model resulted in a val_loss of 0.0036, a massive improvement from the previous model, though this is kind of obvious with the doubling of units.

As the model dimensions changed, i need to adjust process_wav_torch.cpp to update the hardcoded architecture.

Analyzing the spectrum again, there were slight improvements, but not as much as i'd hoped.

Tried throwing more computation at it by increasing the GRU hidden size to 64. Of course update the C++ to match.

The resulting model had a val_loss of 0.0011, not as large of a difference.

Looking at the spectrum as well, not much improvement. Although there are infinite things to try to improve the model, ill move on to creating a model with parameter control for the cutoff for now.

## Moog filter data generation, first parametric training attempt

To do this, i have two options: use c++ dsp, or use a vst.

Using c++ DSP just means creating a C++ script that processes the wav files to create training data, adjusting the cutoff at different frequencies.

If i were to use a vst, i can use [pedalboard by spotify](https://github.com/spotify/pedalboard) to run and control the vst within python. Then i can create the training data through python.

I found a big problem — the benchmark wav i was using of my song clip was in 44.1khz, which wont work with the model to be accurate.

I folded everything to mono, and standardized everything to 48khz.

Imported a moog ladder implementation in c++ from [moogladders github](https://github.com/ddiakopoulos/MoogLadders) and wrote a [C++ program](moogGen\README.md) to filter an input wav file at a collection of cutoffs. These can be used as training data.

Next, i need to redesign the training script (tensor_torch.py) to be able to handle an extra channel of data for the cutoff.

First, the cutoff needs to be scaled to a value between 0.0 and 1.0 to map to an actual knob. This is a log scale.

load_windows function converted into load_conditioned_windows, that stacks the knob value onto the audio. Frequency range and values are hardcoded.

After making sure the inputs of everything handle 2 channels, its good to go.

This crashes because it tries to allocate 140gb of vram. This is because the validation set is 15x larger and runs without batch validation how the training runs. Adding batch processing to the validation makes it work a lot better.

With my 5070ti, the 300 epochs of training would take over 50 min. From the first couple of runs, i deduced that i wont need the 24khz filter range. I limited the range from 20-20khz, avoiding using the 24khz filter data.

I also lowered the initial learning rate one magnitude to 1e-4, as the val loss was spiking badly, indicating it was overshooting.

The training went pretty poorly. After 52 minutes of training, the best val loss was 0.3812. The training loss never went below 0.95. The training loss being higher than the validation loss is an anomaly.

First, the scheduler wasted the last 130 epochs of the training, as it kept halving the learning rate, effectively setting it to zero. The behavior of the scheduler is adjusted so that it is more patient, and to set a minimum learning rate so it doesn't get too small.

The train loss ending at 0.952 means the model's error is at 95%. This means that the model likely doesn't have enough capacity to simulate the cutoffs. Increasing the number of units may help. Tried setting to 128.

Also added an early stop mechanic so that the training stops if the val_loss doesnt improve for 40 consecutive epochs.

While that new training runs, created a new version of process_wav_torch.cpp that takes a cutoff_hz value.

```bash
cmake -Bbuild -DBUILD_EXAMPLES=ON
cmake --build build --config Release
```

```bash
build/examples_out/process_wav_torch_param rtneural_model_param_weights.json input.wav output.wav 1000
```

> Note: these paths are from the original RTNeural clone. See current build instructions above.

Both parameter models produced disappointing outputs. When set to 1k cutoff, its a mush of low end. When set at 5k or 10k, it almost sounds like a high pass filter: no low end, but also just quiet. Something is very wrong with the process.

The changes to the scheduler and the increase in units didn't change the patterns in the loss values.

First, tried using the 1khz cutoff sample with the no-parameter training.

Using the moog filter training data, the training hits 0.0001 loss on training and validation 50 epochs in. The lack of resonance likely makes it easier to simulate, as well as the cleaning up of the training data.

Looking at the spectrogram shows the same story.

The question now is how do i scale this into a working version with parameters.

Also i need to stop hardcoding file names into my python scripts. It's getting annoying to manage.

## Fixing parameter model training

For one fix, passing the knob value into the Conv1d may be a mistake. It's doing a weighted sum of 31 identical knob values. The convolution layer's purpose is to extract features, such as transients or smooth curves. It's a pattern recognition machine. Providing the same number to it over and over does nothing. The fix is to pass only the audio to the convolution layer, tack on the knob value after the fact, then pass that to the GRU layer.

Second, calculating the ESR (loss calculation) per batch ruins the math when different cutoffs are mixed in it. As shuffling of windows is on, different cutoffs are included in a batch. The math of the ESR is to divide the total error by the total energy over the whole batch. The high cutoff windows let through more energy, so they're represented more in the denominator of the ESR, while low cutoff windows vanish from the calculation effectively. This means outputting quieter and mushy audio sort of half solves the ESR, though badly.

Fixed these two issues, and bumped the default LR back up a magnitude.

Still not great. Training loss is now lower than validation loss. 0.46 training loss, 0.89 validation loss.

Some fixes to apply:

First, I changed the cutoff range to start at 60hz. The low 20hz cutoff is just near silence that likely poisons gradients. Ill likely change this back, as a full cutoff is nice to have.

Second, added an extra layer before the GRU to stabilize the mixed scales. The audio features and the knob scalar are very different numbers. The GRU may struggle with making sense of them. Adding a layer norm helps stabilize the input, ensuring the mean is 0 and the variance is 1.

Third, increase the epsilon in the loss function. The current epsilon is too small. It's supposed to protect the loss function when the input is silence, but its too small to be effective.

Finally, seed the GRU h0 from the knob at the start of every window. The model starts completely clean at the start of every window. What is more fit is to initialize the state of the GRU even before the warm up period of the window to adapt its internal memory to the state of the cutoff input value.

This involves a restructuring of the model, where the network grabs the knob value at initialization, expands it into the 128 units with a tanh filter, then loads it before the window starts.

A normalization on the audio features is also applied, as mentioned above in the second fix, before the convolution and state initialization merges.

This results in essentially the math being solved. The training took 65 minutes, and settled at 0.1537 loss. There is definitely room for improvement, but this was a new limit in training. It was due to capacity instead of the training itself. It started at 0.355, kept going down, went through all LR decay steps, and made microscopic improvements. The architecture found everything it could learn and stopped. This suggests the model doesn't have the capacity for the task.

Trying out the model, it seems at 1000hz, the error is incredibly low. The spectrogram shows very little error, except the super low freqs as usual. Even at 60hz, the error is incredibly low, with a few spikes of marginal error.

From these two data points, it seems that the model is accurate. The next step is likely to make a test script that compares the accuracy of many frequencies for me.

Made a script that loads the .pt file, so i can do this in python with pytorch directly. It processes bench_mono.wav at each cutoff, then compares using the reference using ESR loss.

```bash
python python/eval_param_model.py
```

This was the result:

```bash
 Freq (Hz)       ESR    ESR (dB)  Status
---------------------------------------------
        60    0.0015      -28.3dB  good
       100    0.0026      -25.9dB  good
       125    0.0028      -25.5dB  good
       250    0.0033      -24.9dB  good
       500    0.0031      -25.1dB  good
       800    0.0031      -25.0dB  good
      1000    0.0030      -25.2dB  good
      2000    0.0024      -26.1dB  good
      4000    0.0023      -26.4dB  good
      8000    0.0023      -26.3dB  good
     12000    0.0024      -26.1dB  good
     16000    0.9792       -0.1dB  poor
     20000    0.9901       -0.0dB  poor
```

Everything under 12khz is passable, but 16khz and 20khz are horrifically wrong. Bounced out 20khz to see what went wrong.

It seems it was bad training data. The validation data for the 16khz data has a piercing whistle, and the 20khz one was just white noise. It's the same for the training data. Have to fix the dataset and try training without any changes.

The issue is that the moog filter gets too unstable at 16khz. With 4 poles, the eigen value goes to 8.4 at 16khz, when the limit is around 2.8. The solution is to increase oversampling. It was at 1, so set it to 8 for maximum stability and high accuracy of the filter.

Trained with 128 GRU hidden units on the fixed 8x-oversampled data, cutoff range 60–20kHz. Training ran the full 300 epochs at 13.2s/epoch for 66.0 minutes.

Val loss hit 0.0000 by epoch 38. The LR stepped five times: 1e-3 → 5e-4 → 2.5e-4 → 1.25e-4 → 6.25e-5 → 3.13e-5, with each step happening later and producing less improvement. This means the model was effectively solved by epoch 80 and the remainder of training was micro-refinement.

Best val_loss: 0.0000.

After updating `eval_param_model.py` to take the .pt file as an argument:

```bash
$ python eval_param_model.py best_model_param.pt
Using device: cuda

 Freq (Hz)       ESR    ESR (dB)  Status
---------------------------------------------
        60    0.0013      -28.7dB  good
       100    0.0025      -26.1dB  good
       125    0.0027      -25.6dB  good
       250    0.0033      -24.9dB  good
       500    0.0031      -25.1dB  good
       800    0.0031      -25.0dB  good
      1000    0.0030      -25.2dB  good
      2000    0.0024      -26.1dB  good
      4000    0.0023      -26.4dB  good
      8000    0.0023      -26.4dB  good
     12000    0.0023      -26.3dB  good
     16000    0.0023      -26.3dB  good
     20000    0.0023      -26.3dB  good
```

Not much improvement compared to the previous version, outside of the 16k and 20k results being on par with the other frequencies.

Running the previous model trained on the broken data:

```bash
$ python eval_param_model.py ref/moog_60-20k_v1/best_model_param.pt 
Using device: cuda

 Freq (Hz)       ESR    ESR (dB)  Status
---------------------------------------------
        60    0.0015      -28.3dB  good
       100    0.0026      -25.9dB  good
       125    0.0028      -25.5dB  good
       250    0.0033      -24.9dB  good
       500    0.0031      -25.1dB  good
       800    0.0031      -25.0dB  good
      1000    0.0030      -25.2dB  good
      2000    0.0024      -26.1dB  good
      4000    0.0023      -26.4dB  good
      8000    0.0024      -26.1dB  good
     12000    0.0156      -18.1dB  ok
     16000    0.1425       -8.5dB  poor
     20000    0.1369       -8.6dB  poor
```

Obviously, the 16k and 20k output will be bad, but there doesn't seem to be much improvement otherwise.

This disconnect between the test and the training result is mostly to do with how the eval script runs.

First, there was a bug: the model was given 2048 samples of warmup where it outputs 0s, but this warmup was included in the scoring.

This improves the score a bit:

```bash
$ python eval_param_model.py best_model_param.pt
Using device: cuda

 Freq (Hz)       ESR    ESR (dB)  Status
---------------------------------------------
        60    0.0008      -31.1dB  good
       100    0.0018      -27.5dB  good
       125    0.0021      -26.8dB  good
       250    0.0026      -25.9dB  good
       500    0.0023      -26.4dB  good
       800    0.0023      -26.3dB  good
      1000    0.0023      -26.5dB  good
      2000    0.0018      -27.4dB  good
      4000    0.0016      -27.8dB  good
      8000    0.0016      -27.9dB  good
     12000    0.0016      -28.0dB  good
     16000    0.0016      -28.0dB  good
     20000    0.0016      -28.0dB  good
```

One difference is that there isn't any warmup for the window. The training provides the h0 seeding at the start plus a 2048 sample warmup. The eval script doesn't have this, so the score is much worse.

This raises questions about the warmup. The purpose of the warmup in the training is that it allows the GRU to catch up to what a real filter state would be after being freshly seeded with h0.

The issue is that in an actual implementation of this effect, it may not get a warmup period. This means that although the model is accurate, the first 42ms of audio when the model starts processing may be inaccurate, as that is the length of the warmup in the training that wasn't calculated into the loss.

This isn't significant enough to warrant restructuring the whole training around it, but it is worth considering as a limit of the training for now. Not having the warmup likely will make the training harder, as it has to learn the filter behavior and the correct initial state at the same time. Right now, knob_to_h0 seems good enough.

A second difference is that there is no stateful inference. The model is reseeded with h0 at every start of a window. Instead, the internal state should be carried over between windows within a file.

Finally, as a byproduct of porting the training code into the eval, the final incomplete window isn't processed. This means any incomplete window outputs as all 0s. In the training, the incomplete window is just ignored, but in this case, it counts towards loss.

Running the eval script after these fixes:

```bash
$ python eval_param_model.py best_model_param.pt
Using device: cuda

 Freq (Hz)       ESR    ESR (dB)  Status
---------------------------------------------
        60    0.0000      -45.9dB  good
       100    0.0000      -52.5dB  good
       125    0.0000      -54.2dB  good
       250    0.0000      -54.2dB  good
       500    0.0000      -53.3dB  good
       800    0.0000      -52.2dB  good
      1000    0.0000      -51.2dB  good
      2000    0.0000      -50.7dB  good
      4000    0.0000      -49.3dB  good
      8000    0.0000      -48.1dB  good
     12000    0.0000      -47.8dB  good
     16000    0.0000      -47.9dB  good
     20000    0.0000      -48.6dB  good
```

-50dB across the board is fantastic, nearing inaudible differences.

The previous model trained on bugged data:

```bash
$ python eval_param_model.py ref/moog_60-20k_v1/best_model_param.pt 
Using device: cuda

 Freq (Hz)       ESR    ESR (dB)  Status
---------------------------------------------
        60    0.0002      -38.0dB  good
       100    0.0001      -39.4dB  good
       125    0.0001      -40.5dB  good
       250    0.0000      -46.8dB  good
       500    0.0000      -47.3dB  good
       800    0.0000      -46.7dB  good
      1000    0.0000      -46.2dB  good
      2000    0.0000      -47.6dB  good
      4000    0.0000      -45.0dB  good
      8000    0.0001      -38.9dB  good
     12000    0.0133      -18.8dB  ok
     16000    0.1208       -9.2dB  poor
     20000    0.1348       -8.7dB  poor
```

Comparing the spectrum of the 1000hz output is impressive as well. There is no visible error above 60hz, and the clumps of sub 60hz bright spots present in every previous model are dim spots.

I then trained a model that used the 20hz training as well, as the 60-20k model didn't handle 20hz at all because of the scaling of the cutoff knob. Training ran the full 300 epochs at 13.7s/epoch for 68.6 minutes total.

Val loss hit 0.0000 by epoch 39 and stayed there. The LR scheduler stepped far more aggressively than any previous run, halving six times: 1e-3 → 5e-4 → 2.5e-4 → 1.25e-4 → 6.25e-5 → 3.13e-5 → 1.56e-5, finishing at 1.56e-05. The model kept finding marginal improvements at each step down, suggesting 128 units has significantly more capacity than the task strictly requires.

Best val_loss: 0.0000.

The eval script showed this:

```bash
$ python eval_param_model.py best_model_param.pt 
Using device: cuda

 Freq (Hz)       ESR    ESR (dB)  Status
---------------------------------------------
        20    0.0002      -36.9dB  good
        60    0.0000      -50.2dB  good
       100    0.0000      -53.1dB  good
       125    0.0000      -53.8dB  good
       250    0.0000      -53.7dB  good
       500    0.0000      -52.3dB  good
       800    0.0000      -50.8dB  good
      1000    0.0000      -50.5dB  good
      2000    0.0000      -49.9dB  good
      4000    0.0000      -48.6dB  good
      8000    0.0000      -48.1dB  good
     12000    0.0000      -48.3dB  good
     16000    0.0000      -48.9dB  good
     20000    0.0000      -50.7dB  good
```

Because the 20hz sample is essentially silence, its harder on the ESR denominator. Regardless, -36.0dB is barely noticeable.

Moving on, 3 points of curiosity:

1. The model needs to be tested for variable parameters. The model is built to take different param values, but it has only been trained and evaluated on a static cutoff. Seeing how it handles a changing cutoff, and changing the training to improve its performance in this aspect is needed. Considering adding training data with changing knob values, as well as validation data that isn't at a cutoff that the model trains on, as well as moving cutoffs.

2. Implementing the Csound opcode. This will be particularly useful for experimenting with real time use cases and variable parameter control. This is also the part that MUST work for the paper for the Csound conference, so its simply high priority.

3. How much can i simplify the model. I kept increasing the number of units when it wasn't working well, but a large part of the error can be attributed to the bad training data. Perhaps, now with the training data fixed, the model can still maintain accuracy while reducing the number of units. This would not only make the training faster, but reduce inference time, making it more useful for real time applications. Could also consider if I need the layerNorm or the knob_to_h0. RTNeural doesn't support these layers, meaning likely a model can work with just conv -> GRU -> dense, all native layers to RTNeural. Things to also consider: reducing the conv features, the kernel size and the number of training windows.

Tried simplifying the model first — reducing the hidden units in the GRU from 128 to 32.

The training was around 20 minutes shorter at 42.6 mins with no early stop, around 8.7s per epoch. Val_loss hit 0.0001 by epoch 43. By epoch 85, val_loss was already hitting 0.0000. The LR scheduler stayed at 1e-3 all the way to epoch 206, meaning the model kept finding marginal improvements sporadically across that entire window. Stepped to 5e-4 at epoch 206, then 2.5e-4 at epoch 277. Both steps had diminishing returns as the model was already near its floor.

Best val_loss: 0.0000.

This resulted in this eval:

```bash
$ python eval_param_model.py best_model_param.pt 
Using device: cuda

 Freq (Hz)       ESR    ESR (dB)  Status
---------------------------------------------
        20    0.0018      -27.5dB  good
        60    0.0001      -40.9dB  good
       100    0.0000      -44.8dB  good
       125    0.0000      -45.6dB  good
       250    0.0000      -46.5dB  good
       500    0.0000      -48.6dB  good
       800    0.0000      -49.2dB  good
      1000    0.0000      -48.9dB  good
      2000    0.0000      -49.2dB  good
      4000    0.0000      -47.1dB  good
      8000    0.0000      -48.1dB  good
     12000    0.0000      -47.4dB  good
     16000    0.0000      -46.0dB  good
     20000    0.0001      -42.9dB  good
```

Not significantly worse, except at 20hz, where it was actually decently worse.

## Referencing other repos

During this training, I looked into how other implementations of RTNeural handle model architecture and training.

[This repo](https://github.com/GuitarML/NeuralSeed) shows rtneural running on the daisy seed.

The neural network architecture is explicitly designed to be as minimal as possible to run on the Daisy Seed's Cortex-M7 microcontroller, which has strict 480MHz CPU and 128KB Flash memory limits.

It uses a single GRU layer. 10 for a static model, 8 for parametrized models with 2-3 knobs. The units are reduced, as increasing the number of inputs scales the computation required.

This GRU layer feeds into a single dense layer for the sample output.

It uses a hardcoded skip connection, where the raw input audio is added to the output. This means the model only learns the difference rather than recreating the entire audio.

For parameter inputs, the knob positions, normalized from 0-1, are appended to the audio sample for a multidimensional input array at every sample — similar to my structure.

In terms of the training process, the training data is recorded at 5 intervals (e.g., 0%, 25%, 50%, 75%, 100%) for each physical knob. The 2nd and 3rd knob are recorded at only 3 positions, resulting in 45 total training files.

The model doesn't start from random weights, but from a pre-trained starting point, giving it a baseline understanding of audio processing. This is familiar from CV models I've used.

The actual training loop uses a warmup process like I do. The data set is also shuffled and divided into mini batches, with a default 200 sample warmup. The batches are processed in chunks of 1000 samples. The loss calc, gradient computation and weight update happens after each chunk:

```python
def process_batch(input_batch, target_batch, init_len, up_fr, optim, loss_fcn):
    self(input_batch[0:init_len, :, :])
    self.zero_grad()

    start_i = init_len
    batch_loss = 0

    for k in range(math.ceil((input_batch.shape[0] - init_len) / up_fr)):
        output = self(input_batch[start_i:start_i + up_fr, :, :])
        loss = loss_fcn(output, target_batch[start_i:start_i + up_fr, :, :])
        
        loss.backward()
        optim.step()

        # Detach hidden state to prevent gradient explosion over long sequences
        self.detach_hidden()
        self.zero_grad()

        start_i += up_fr
        batch_loss += loss

    return batch_loss
```

The loss calculation uses a mix of ESR and DC loss. ESRpre constitutes 75% of the error. The model's output and target audio goes through a pre-emphasis filter (a hpf by default), dividing the squared error by the total energy of the target signal. This is pretty similar to my ESR calc.

25% of the loss is DC loss. This calculates the squared difference between the mean value of the output and the mean value of the target, ensuring the DC offset is correct.

ESR calculates the shape of the waveform, the DC loss the absolute level.

For learning rate, it uses Adam optimizer, with an initial learning rate of 0.005 and a weight decay of 1e-4. Learning rate decay is managed dynamically using PyTorch's `ReduceLROnPlateau` scheduler. The scheduler has a patience of 5, and decreases lr by half.

The script defaults to 2000 epochs, with the google collab set to 300 epochs. The early stop is set to 25 cycles.

---

[This repo](https://github.com/spluta/RTNeural_Plugin) is a plugin for max, pd and super collider for rtneural.

The model architecture is very similar. This is because they both use the [CoreaudioML](https://github.com/Alec-Wright/CoreAudioML) submodule for the architecture. This is called the Automated Guitar Amp Modeling (AGAM) architecture.

By default, there are 40 GRU units, which the plugins use. This should be kept around 32-64 for desktop purposes. The plugins don't use skip connections, but it is best practice for virtual analog modelling.

This plugin provides 2 separate architectures: the MLP Oscillator architecture, and micro-TCN.

Micro-TCN uses a 1D convolution, which allows the model to look at past audio samples. This goes into a 1D batch normalization and a Parametric ReLU (PReLU) activation function.

The 1D conv has a kernel size and a dilation. The dilation puts gaps between the input values, meaning it processes the same number of samples but deeper in the past.

This block is then stacked multiple times with increased dilation. This means the convolutions act as memory, as it looks back in time for the audio effect's decay, sag and freq response.

This is in contrast to my current architecture's conv layer, which acts as a feature extractor and a preprocessor.

The advantages of micro-TCN is in its fast training. TCNs don't need to see each sample in sequence, so it can process every sample in parallel. This makes training faster compared to a RNN using Truncated Backpropagation Through Time.

Micro-TCNs also have a fixed receptive field, meaning it can see a set amount of the past. This makes it more fit for effects with long-term dependencies, where an RNN can fail as the gradients vanish over time.

The cons of a Micro-TCN are its high memory usage and incompatibility with IIR behavior. For a TCN to handle audio circuits that need long memory, it needs a long receptive field. This eats up a lot of RAM, as it needs to store the audio in memory.

Structurally, TCNs are non-linear FIR filters. This means they can't handle IIR, where there are feedback loops. RNNs on the other hand naturally handle feedback, as their output feeds back into their hidden state.

These plugins do not handle parameters. They are designed for snapshots only. They also don't use a pretrained model, spending the extra epochs to train from scratch.

Same exact loss calc, learning rate optimizer, and epochs as NeuralSeed, as it comes from CoreAudioML.

---

## Migration to standalone repo

All previous work was done inside a clone of the RTNeural repo, with files scattered across it. Moved everything to this dedicated repo for proper history tracking.

What changed in the migration:

- **File structure**: audio files in `audio/`, Python scripts in `python/`, C++ tools each in their own `src/<name>/` subdirectory
- **Dependencies**: RTNeural moved from being the host repo to a git submodule at `vendor/RTNeural/`; `dr_wav.h` vendored at `vendor/dr_wav.h` instead of FetchContent
- **CMake**: rebuilt from scratch — new project name `rtneuralCsound`, `add_tool()` function, build output to `build/bin/Release/`
- **Reference models**: archived by version in `ref/`
- **Build commands** in earlier diary entries reference the old RTNeural clone paths (`build/examples_out/`, `python/testSound.wav`, etc.) — they are preserved as-is for historical accuracy


## Simplification of the model architecture

From what I learned from the other repos, it seems 64 units on GRU is plenty. I will remove the knob_to_h0 initialization portion. LayerNorm is also likely safe to drop, as the knobs are scaled to 0-1 while the audio is -1-1. There isn't much normalizing to do. Conv1d is an honest improvement over the AGAM architecture that supplements the GRU's IIR memory with richer features. Dropping it would trade accuracy for minimalism that I don't need for a desktop use case.

Adding in the skip connection is also worthwhile, as it adds little complexity for standard practice for virtual analog.

This means my architecture is: Conv1d -> (concat knob at every step) -> GRU -> Dense -> output + skip from raw input.

This architecture also has the added bonus of all the layers being natively supported by RTNeural. This has 3 advantages:

1. RTneural can deserialize the model from the JSON without any custom inference code. The inference code becomes ultra lean.

2. Rtneural can use a templated static graph mode where the entire network is unrolled at compile time. This means the compiler inlines everything, aggressively optimizing the code.

3. RTNeural's native layers use XSIMD intrinsics.

In all, RTNeural can handle this model faster and easier, meaning implementation into Csound will be easier and more performative.

Now to see how well it trains and performs.

Training completed in 37.8 minutes (300 epochs, no early stop), around 7.5s per epoch, slightly faster than the previous 32-unit run at 42.6 minutes, since the model has fewer parameters without `knob_to_h0` and `LayerNorm`.

The LR scheduler stepped much more aggressively than previous runs. It stepped to 5e-4 at epoch 88 (vs epoch 206 last time), then to 2.5e-4 at epoch 144, then to 1.25e-4 at epoch 261, and finally to 6.25e-5 at epoch 284. This suggests the model was plateauing more frequently without the guidance that `knob_to_h0` provided. Best val_loss: 0.0000 (rounds to zero at 4 decimal places).

```bash
$ python eval_param_model.py best_model_param.pt
Using device: cuda

 Freq (Hz)       ESR    ESR (dB)  Status
---------------------------------------------
        20    0.0101      -19.9dB  ok
        60    0.0010      -30.2dB  good
       100    0.0004      -34.2dB  good
       125    0.0002      -36.5dB  good
       250    0.0002      -37.6dB  good
       500    0.0001      -39.0dB  good
       800    0.0001      -41.2dB  good
      1000    0.0001      -42.3dB  good
      2000    0.0000      -43.7dB  good
      4000    0.0000      -45.4dB  good
      8000    0.0000      -44.2dB  good
     12000    0.0000      -46.2dB  good
     16000    0.0001      -41.6dB  good
     20000    0.0001      -41.3dB  good
```

This is around 10dB worse than the previous best model across the board. 20Hz is borderline at -19.9dB. Everything else is technically "good" but well behind the -48 to -54dB range the previous model achieved.

The culprit is almost certainly the removal of `knob_to_h0`. That layer was seeding the GRU with an explicit representation of the target filter state before processing any audio. Without it, the GRU starts cold and has to infer the correct filter mode from the knob value in the input stream over the warmup period. The more aggressive LR scheduling pattern supports this: the model was struggling harder to find its footing.

The skip connection and removal of `LayerNorm` are likely minor contributors. The scale mismatch between audio and knob was not severe enough for `LayerNorm` to matter much.

Conclusion: `knob_to_h0` is doing real work and should be restored. Since the Csound opcode requires custom C++ inference regardless, there is no meaningful cost to keeping it. The "fully RTNeural native" goal was a nice-to-have, not a hard requirement.