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

The removal of `knob_to_h0` may have played a large role in this change. That layer was seeding the GRU with an explicit representation of the target filter state before processing any audio. Without it, the GRU starts cold and has to infer the correct filter mode from the knob value in the input stream over the warmup period. The more aggressive LR scheduling pattern supports this: the model was struggling harder to find its footing.

The skip connection and removal of `LayerNorm` are likely minor contributors. The scale mismatch between audio and knob was not severe enough for `LayerNorm` to matter much.

However, there is a methodological issue with attributing the full ~10dB regression solely to `knob_to_h0`: three things changed simultaneously in the simplification run:

| Removed | Added |
|---|---|
| `knob_to_h0` | Skip connection |
| `LayerNorm` | |
| 128→32 hidden units | |

Any combination of these could contribute to the lost accuracy. The 32-unit model *with* `knob_to_h0` from the earlier run already showed a ~15dB drop from the 128-unit model (e.g., 20Hz went from -36.9dB to -27.5dB). The simplified model dropped another ~6-10dB on top of that. Was it `knob_to_h0`, `LayerNorm`, or the interaction of both? You can't tell from a single compound change.

The argument that "audio is -1 to 1 and knob is 0-1, so normalization isn't needed" overlooks that the *conv features* (16 channels of arbitrary statistics) are what actually flow into the GRU alongside the knob — those could benefit from normalization even if the raw inputs don't strictly need it.


## Revised next steps

### 1. Ablation study: isolate `knob_to_h0` and `LayerNorm` (COMPLETE)

Result: `knob_to_h0` contributes ~3dB, `LayerNorm` contributes ~5–7dB. Both are load-bearing. See "Adding back in knob_to_h0" section above for full analysis.

### 2. Test dynamic cutoff behavior (HIGH PRIORITY — before any Csound work)

The model has only been trained and evaluated on static cutoffs. With moving parameters:

- The `knob_to_h0` seeding happens only once at initialization — it was never designed for parameter changes mid-stream. What happens to the GRU state when the knob changes abruptly?
- The 2048-sample warmup (42ms) is invisible during training loss calculation. In a Csound context, every parameter change means the model needs to "catch up" to a new filter state. Can it do that fast enough?
- Training data with sweeps/modulation might be necessary — and if so, the whole training pipeline needs rework.

**Approach**: Write a quick Python script that feeds a swept cutoff through the model, compare output against the real Moog filter on the same sweep. If it falls apart on moving parameters, the training strategy needs to change before the Csound opcode is built.

Rationale for prioritizing this over the opcode: if the model can't handle dynamic cutoffs well, the Csound opcode — which exists precisely to enable real-time parameter control — won't be useful.

### 3. Csound opcode (MUST-HAVE for conference paper)

The non-negotiable deliverable. Use whatever model architecture survives steps 1-3. The C++ inference code in `process_wav_torch_param.cpp` already demonstrates custom layer handling — the "fully RTNeural-native" constraint was always self-imposed, not a technical limitation. The opcode needs:

- Real-time audio I/O via Csound's opcode API
- `k-rate` cutoff parameter (updates every control cycle, not every sample)
- `a-rate` parameter would be ideal but can be a stretch goal
- Stateful inference: GRU hidden state persists across `ksmps`-sized blocks so the filter doesn't reset

### 4. Architecture sweet spot: try 64 units

The 128-unit model was overkill (LR halved six times, lots of spare capacity). The 32-unit model is slightly under-provisioned at low frequencies. 64 units with `knob_to_h0` restored would likely be the sweet spot: near the accuracy of 128 units but with lower inference cost. Worth a training run after the ablation results come in.

## Adding back in knob_to_h0

To more clearly see the effects of knob_to_h0, i did a training run with just it added back into the AGAM architecture.

The run lasted 48.4 minutes, which was longer than without, as it adds a small layer that runs every forward pass. Each epoch was around 10s vs the 7.5s from last run, costing 10 extra minutes.

The convergence pattern seen in the training was much less chaotic than the no-k2h0 model. Adding it back made the LR step only 3 times, starting at epochs 151, 212, and 267, all of which produced immediate and clean improvements in train and val loss. The no-k2h0 run stepped 4 times starting earlier at epoch 88, meaning the model was struggling to converge without the hidden state guidance.

```bash
$ python python/eval_param_model.py ref/09_moog_20-20k_AGAM+conv+knob_to_h0/best_model_param_k2h0.pt
Using device: cuda

 Freq (Hz)       ESR    ESR (dB)  Status
---------------------------------------------
        20    0.0054      -22.6dB  good
        60    0.0005      -33.4dB  good
       100    0.0002      -37.7dB  good
       125    0.0001      -38.6dB  good
       250    0.0001      -41.0dB  good
       500    0.0001      -42.6dB  good
       800    0.0000      -43.8dB  good
      1000    0.0000      -43.2dB  good
      2000    0.0001      -42.4dB  good
      4000    0.0000      -44.0dB  good
      8000    0.0001      -42.7dB  good
     12000    0.0001      -42.3dB  good
     16000    0.0001      -39.6dB  good
     20000    0.0000      -47.4dB  good
```

Comparing the three 32-unit runs side by side:

| Freq | 32u + k2h0 + LN | 32u, no k2h0, no LN | 32u + k2h0, no LN (this run) |
|------|-----------------|----------------------|-------------------------------|
| 20Hz | -27.5dB | -19.9dB | -22.6dB |
| 60Hz | -40.9dB | -30.2dB | -33.4dB |
| 500Hz | -48.6dB | -39.0dB | -42.6dB |
| 1kHz | -48.9dB | -42.3dB | -43.2dB |
| 8kHz | -48.1dB | -44.2dB | -42.7dB |

Restoring `knob_to_h0` recovered ~3dB across the board, with the largest gains at low frequencies (20–500Hz) where filter state matters most. This confirms it contributes meaningfully — but it is not the dominant factor.

The more significant finding is that this run is still ~5–7dB worse than the 32-unit model that had both `knob_to_h0` and `LayerNorm`. That gap can only be attributed to `LayerNorm`. The earlier assumption that audio (-1–1) and knob (0–1) are close enough in scale to not need normalization was wrong. It's not the raw inputs that matter, it's the Conv1d output (16 channels of arbitrary statistics) that flows into the GRU alongside the knob. Those features have no guaranteed scale and clearly benefit from normalization.

After joining ADC Japan 2026, I noticed many ML related talks visualized the loss val in the training with graphs as part of how their training went. To achieve this, I made the training script save the loss values as a CSV, and updated it to take an output directory as a CLI argument so all artifacts land in a versioned `ref/` folder automatically.

## Ablation: LayerNorm only, no knob_to_h0

To complete the ablation, ran a model with `LayerNorm` restored but `knob_to_h0` still removed. Architecture: `Conv1d → LayerNorm → (concat knob) → GRU → Dense + skip`.

Training ran the full 300 epochs in 42.6 minutes (~8.5s/epoch). The LR stepped 4 times: 1e-3 → 5e-4 at epoch 137, → 2.5e-4 at epoch 196, → 1.25e-4 at epoch 248, → 6.25e-5 at epoch 274. More steps than the k2h0-only run (3 steps) but with a later first step than the no-component run (epoch 88), suggesting `LayerNorm` provides some stabilization but less than `knob_to_h0`. Val loss was noticeably noisier at LR 1e-3 than the k2h0 run, with several large spikes (e.g. 0.0424 at epoch 26). Best val_loss: 0.0001.

```bash
$ python python/eval_param_model.py ref/10_moog_20-20k_AGAM+conv+LN/best_model.pt
Using device: cuda

 Freq (Hz)       ESR    ESR (dB)  Status
---------------------------------------------
        20    0.0115      -19.4dB  ok
        60    0.0009      -30.6dB  good
       100    0.0002      -36.8dB  good
       125    0.0002      -37.1dB  good
       250    0.0002      -37.3dB  good
       500    0.0002      -37.8dB  good
       800    0.0001      -39.1dB  good
      1000    0.0001      -40.0dB  good
      2000    0.0001      -41.6dB  good
      4000    0.0000      -45.8dB  good
      8000    0.0001      -41.6dB  good
     12000    0.0001      -39.6dB  good
     16000    0.0000      -46.2dB  good
     20000    0.0001      -41.2dB  good
```

## Full ablation results

All four 32-unit variants:

| Freq | 32u + k2h0 + LN | 32u no k2h0 no LN | 32u + k2h0, no LN | 32u + LN, no k2h0 |
|------|-----------------|-------------------|-------------------|--------------------|
| 20Hz | -27.5dB | -19.9dB | -22.6dB | -19.4dB |
| 60Hz | -40.9dB | -30.2dB | -33.4dB | -30.6dB |
| 500Hz | -48.6dB | -39.0dB | -42.6dB | -37.8dB |
| 1kHz | -48.9dB | -42.3dB | -43.2dB | -40.0dB |
| 8kHz | -48.1dB | -44.2dB | -42.7dB | -41.6dB |

**The LN-only result is the key finding.** It performs roughly the same as the no-component baseline, marginally better at some mid-high frequencies, slightly worse at 20Hz. `LayerNorm` alone contributes almost nothing.

This overturns the earlier assumption that `LayerNorm` was contributing ~5–7dB independently. The previous conclusion was based on the gap between the k2h0-only run and the both-components baseline, and incorrectly attributed that entire gap to `LayerNorm`. What's actually happening is that `knob_to_h0` and `LayerNorm` are **synergistic**, not additive:

- `knob_to_h0` alone: ~3dB improvement, mainly at low frequencies
- `LayerNorm` alone: negligible, roughly the same as no components
- Both together: ~10dB improvement over no components

The most likely explanation is that `LayerNorm` only helps when the GRU already starts in a meaningful state. With `knob_to_h0` seeding the hidden state to match the target cutoff, the GRU has prior knowledge of what filter behavior to expect. `LayerNorm` then makes the conv feature distribution consistent across cutoffs and audio content, amplifying that prior knowledge. Without `knob_to_h0`, the GRU starts cold regardless and the normalized inputs don't give it any additional signal about which filter mode to operate in.

**Revised ablation conclusions:**

- `knob_to_h0` is the primary driver of accuracy, particularly at low frequencies where filter state initialization matters most.
- `LayerNorm` is only beneficial in combination with `knob_to_h0`. Standalone it is ineffective.
- The RTNeural-native constraint (which requires dropping both) costs the full ~10dB.

The practical implication for the Csound implementation: the custom C++ inference path in `process_wav_torch_param.cpp` is the right approach for more accurate inference. Dropping `LayerNorm` and `knob_to_h0` for RTNeural compatibility sacrifices accuracy. The question is if the increased accuracy is worth the cost in inference time, and if it will work in real-time environments.

## Next steps

Two options are on the table:

**Option A: Test real-time performance first.** Build a Csound opcode using the existing custom inference path (with `knob_to_h0` + `LayerNorm`) and measure whether it runs at 48kHz within a real Csound context. This is the highest-priority path because the opcode is the non-negotiable conference paper deliverable. If the custom path is fast enough, the native architecture constraint was never necessary and the architecture question is settled.

**Option B: Train a 64-unit all-native model.** A 64-unit model without `knob_to_h0` or `LayerNorm` would be new data: the only all-native run so far was the 32-unit simplified model (run 08, ~-19 to -46dB). However, unit count alone cannot recover the synergistic benefit of `knob_to_h0` + `LayerNorm`, so the ceiling for a native model is likely in the -25 to -42dB range regardless of unit count. This option only becomes compelling if Option A reveals a real performance bottleneck that forces the native constraint.

**Decision: pursue Option A first.** The real-time test answers a binary question — does this work or not — and directly unblocks the conference paper. Option B is speculative and lower-value until there's a concrete reason the custom path can't be used.

Option B was also run overnight as run 11. See "Run 11: 64-unit all-native model" below.

## Run 11: 64-unit all-native model

Trained a 64-unit `Conv1d → GRU → Dense + skip` model (no `knob_to_h0`, no `LayerNorm`) to get a concrete data point on whether unit count alone can close the accuracy gap left by dropping both components.

Training ran the full 300 epochs without triggering early stopping, indicating the model continued making incremental progress through the end. Three distinct convergence phases visible in the loss history:

- Epochs 1–10: large initial drop, val loss 0.319 → 0.017
- Epochs 81–82: second sharp drop into the 8e-05 range
- Epochs 117–139: third drop settling around 2e-05
- Epochs 227+: final plateau at ~1.3–1.7e-05

Best val_loss: ~1.29e-05 (epoch 291), approximately **-48.9 dB ESR**.

```bash
$ python python/eval_param_model.py ref/11_moog_20-20k_AGAM+conv_64u/best_model.pt
Using device: cuda

 Freq (Hz)       ESR    ESR (dB)  Status
---------------------------------------------
        20    0.0032      -24.9dB  good
        60    0.0003      -35.0dB  good
       100    0.0001      -38.6dB  good
       125    0.0001      -39.6dB  good
       250    0.0001      -41.3dB  good
       500    0.0001      -42.8dB  good
       800    0.0000      -43.3dB  good
      1000    0.0000      -43.2dB  good
      2000    0.0000      -43.8dB  good
      4000    0.0001      -43.0dB  good
      8000    0.0000      -43.9dB  good
     12000    0.0000      -44.9dB  good
     16000    0.0000      -45.0dB  good
     20000    0.0000      -46.5dB  good
```

All 14 frequencies rated "good". Comparing against the 32-unit all-native baseline (run 08):

| Freq | 32u no k2h0 no LN | 64u no k2h0 no LN |
|------|-------------------|-------------------|
| 20Hz | -19.9dB | -24.9dB |
| 60Hz | -30.2dB | -35.0dB |
| 500Hz | -39.0dB | -42.8dB |
| 1kHz | -42.3dB | -43.2dB |
| 8kHz | -44.2dB | -43.9dB |

Doubling units recovered ~5dB at low frequencies, ~3-4dB at mid, and made essentially no difference at high frequencies where the 32-unit model was already accurate. Critically, 20Hz improved from "ok" (-19.9dB, borderline) to "good" (-24.9dB), which is a meaningful practical improvement.

**Extended ablation table** (all 32-unit variants + 64-unit native):

| Freq | 32u + k2h0 + LN | 32u + k2h0, no LN | 32u + LN, no k2h0 | 32u no k2h0 no LN | 64u no k2h0 no LN |
|------|-----------------|-------------------|-------------------|--------------------|-------------------|
| 20Hz | -27.5dB | -22.6dB | -19.4dB | -19.9dB | **-24.9dB** |
| 60Hz | -40.9dB | -33.4dB | -30.6dB | -30.2dB | **-35.0dB** |
| 500Hz | -48.6dB | -42.6dB | -37.8dB | -39.0dB | **-42.8dB** |
| 1kHz | -48.9dB | -43.2dB | -40.0dB | -42.3dB | **-43.2dB** |
| 8kHz | -48.1dB | -42.7dB | -41.6dB | -44.2dB | **-43.9dB** |

The 64-unit all-native model sits roughly on par with the 32-unit `k2h0`-only run at most frequencies, and slightly ahead at mid-high. It does not match the 32-unit `k2h0 + LN` baseline at low frequencies (5–7dB gap at 20-60Hz remains), confirming that unit count alone cannot recover the synergistic effect of both components. However, as the best fully RTNeural-native option, it is a viable deployment target if real-time performance testing rules out the custom inference path.

## Csound Opcode

Given that the Csound opcode is the non-negotiable deliverable, option 3 is likely the most pragmatic — the custom inference path already works, and accuracy should not be traded away for an architectural constraint that was always self-imposed.

Writing a Csound opcode can be done as a plugin. This would allow me to use C++ to write the opcode while using just the Csound header. This approach has the advantage that I only need to include the `include/` folder of the Csound repo. This means the entire Csound repo exists here, but the headers are used to build my opcode plugin and nothing else.

Apparently, making a Csound opcode doesn't use a `.lib` file, but a big struct called `CSOUND` which contains funciton pointers for everythin I need for a plugin. I should also keep in mind this struct changes between windows and linux/mac for future compatibility issues.

Here is what I need to do to get a working opcode:

- Decide on and train final native-architecture model (Conv1d → concat knob → GRU → Dense + skip, no LayerNorm, no knob_to_h0) — pick unit count (32 or 64)
- Add csound repo as git submodule at vendor/csound/
- Update CMakeLists.txt: add opcode MODULE target alongside existing tool targets
- Write src/rtneural_opcode.cpp: load JSON via parseJson<float>, stateful GRU inference across ksmps blocks, k-rate cutoff parameter
- Write test/test.csd and place model JSON in test/models/
- Build opcode DLL and verify it loads and processes audio correctly in Csound

In terms of the final model format, I will be using full native layers with 64 units. TBD

First add csound as a submodule:

```bash
git submodule add https://github.com/csound/csound vendor/csound
```

Added Csound as a submodule and started wiring up the build. The plan is to build a `.dll` plugin that Csound loads at runtime, using the CPOF (C++ Plugin Opcode Framework) which lives in `vendor/csound/include/plugin.h` and `modload.h`. The framework is header-only, so no need to actually build Csound. Just include the headers and let Csound's installed binary load the plugin at runtime.

First problem: Csound's headers depend on two generated files, `float-version.h` and `version.h`, that are normally produced by Csound's own CMake configure step. They're both `.h.in` templates in `vendor/csound/include/`. Rather than running Csound's full CMake build (which pulls in a ton of dependencies), I added two `configure_file` calls to my own root `CMakeLists.txt`. `float-version.h` just sets `USE_DOUBLE` (standard double-precision Csound), and `version.h` fills in the version numbers (7.0.0 from Csound's CMakeLists). Both get generated into `vendor/csound/include/` at configure time.

Also found that the root CMakeLists had two broken targets left over from the migration: `add_tool(process_wav_torch_param)` was looking for a file named `process_wav_torch_param.cpp` but the actual file was `process_wav_torch_param2.cpp`, and `add_tool(process_wav_torch_param2)` pointed at a directory that doesn't exist. Renamed the source file to drop the "2" and removed the dead target, so `add_tool(process_wav_torch_param)` works cleanly again.

Wrote the initial passthrough opcode in `src/csound_opcode/moognn.cpp`. The CPOF API is cleaner than I expected: inherit from `csnd::Plugin<nout, nin>`, implement `init()` and `aperf()`, register with `csnd::plugin<>()` in `on_load()`. The `sa_offset()` call happens automatically before `aperf()`, zeroing the offset/early regions of the output buffer for sample-accurate event scheduling. So the actual passthrough body is just a `std::copy` from input to output over `[offset, nsmps)`.

Added the `moognn` shared library target to CMakeLists with `PREFIX ""` so the output is `moognn.dll` not `libmoognn.dll` (Csound expects no prefix on Windows). Built successfully:

```
moognn.vcxproj -> build\bin\Release\moognn.dll
```

Wrote `test_passthrough.csd` using `--opcode-lib=build/bin/Release/moognn.dll` to load the plugin directly. Running it gave:

```
Loading command-line libraries:
  build/bin/Release/moognn.dll
error:  Unable to find opcode with name: moognn
```

The DLL loaded, as the C exports `csoundModuleCreate`/`Init`/`Destroy` were found and called, but the opcode never registered. The cause was a header mismatch: the vendor `csound` submodule is at commit `1a99bfddf` (a development snapshot labeled `6.17.0-3673`), while the installed Csound binary is the actual 7.0 release. These two versions have a different internal `CSOUND` struct layout. The struct is a large table of function pointers, including `AppendOpcode`, which is how `csnd::plugin<>()` registers opcodes at runtime. Compiling against the pre-7.0 headers put `AppendOpcode` at the wrong offset, so the call silently hit garbage and the opcode was never registered.

Fix: point the `moognn` build target at the installed Csound 7.0 headers (`C:/Program Files/Csound7/include/csound/`) instead of the vendor submodule. These headers already have `float-version.h` and `version.h` generated, so the `configure_file` steps were removed. The `CSOUND_INCLUDE_DIR` CMake cache variable defaults to the installed path but can be overridden. The vendor submodule stays for reading the Csound source, but cannot be used as the compile target.

Rebuilt against the correct headers and the DLL is ready to test.

Build the opcode DLL:

```bash
cmake -Bbuild
cmake --build build --config Release --target moognn
# output: build/bin/Release/moognn.dll
```

Test with the passthrough `.csd`:

```bash
csound test_passthrough.csd
```

The `.csd` uses `--opcode-lib=build/bin/Release/moognn.dll` to load the plugin directly without needing to install it or set `OPCODE6DIR64`. To install permanently, copy `moognn.dll` to `C:/Program Files/Csound7/plugins64/`.

## RTNeural inference in the opcode

With the plugin pipeline confirmed working, replaced the passthrough with real RTNeural inference using the 64-unit all-native model (run 11).

The model types mirror the architecture exactly: compile-time template parameters, same as in `process_wav_torch_param.cpp`, but 64 units and no `LayerNorm` or `knob_to_h0`:

```cpp
using ConvStage = RTNeural::ModelT<float, 1, 16,
    RTNeural::Conv1DT<float, 1, 16, 31, 1>>;

using RecurrentStage = RTNeural::ModelT<float, 17, 1,
    RTNeural::GRULayerT<float, 17, 64>,
    RTNeural::DenseT<float, 64, 1>>;
```

The opcode signature changed from `"ak"` to `"aSk"`: audio in, string model path, k-rate cutoff. In CPOF's `Param<>`, string args are stored as `STRINGDAT*` behind the `MYFLT*` pointer, so they're accessed via a cast: `(STRINGDAT*)inargs.begin()[1]`. Weights are loaded in `init()` using `torch_helpers`, same key prefixes as the training script (`"conv.conv."`, `"gru."`, `"dense."`). The model is heap-allocated with `new`/`delete` (freed in `deinit()`) to avoid stack overflow from Eigen's fixed-size matrices.

The `aperf()` loop runs per-sample inference over `[offset, nsmps)`: forward through conv, concat the normalized knob as the 17th feature, forward through GRU+Dense, then add the raw audio as the skip connection.

Hit one build error: including Csound headers before RTNeural caused macro collisions. Csound's C API defines macros that trample C++ reserved words, breaking `<chrono>` when Eigen/RTNeural pulls it in. Fix: include RTNeural and all standard library headers before the Csound headers.

Built successfully. Updated `test_passthrough.csd` to pass the model path and a `linseg` cutoff sweep (50Hz → 5kHz → 50Hz over 5 seconds) to test both inference and real-time parameter control in one pass:

```csound
ain   diskin "audio/bench_mono.wav", 1
klin = linseg:k(50, 2.5, 5000, 2.5, 50)
aout  moognn ain, "ref/11_moog_20-20k_AGAM+conv_64u/weights.json", klin
      out aout
```

Test output sounds good so far:

```bash
csound test_passthrough.csd
```

## CPOF API cleanup and fixing note-on clicks

Went back and cleaned up the opcode against the CPOF examples and README in `vendor/csound/examples/plugin/`.

Three things were wrong with how the opcode was written:

First, string args should be accessed via `inargs.str_data(1).data` rather than manually casting the raw pointer. The README shows this clearly and it's cleaner.

Second, audio buffer pointers should use `outargs.data(0)` and `inargs.data(0)` instead of `&outargs[0]` and `&inargs[0]`. Both work, but the former is the designed API.

Third, the README examples call `sa_offset(out)` at the top of `aperf()` to zero the `[0, offset)` region of the output buffer for sample-accurate scheduling. Added this call — but it immediately caused a build error:

```
error C2660: 'csnd::Plugin<1,3>::sa_offset': function does not take 1 arguments
```

Checking the installed Csound 7 `plugin.h`, `sa_offset()` takes no arguments in Csound 7 and is marked as called implicitly before `aperf()`. The README documents the Csound 6 API. The actual `opcodes.cpp` example file doesn't call `sa_offset()` at all, which is consistent with Csound 7 handling it automatically. Removed the call.

---

With the cleanup done, I made a second csound script that takes MIDI input to play a sawtooth wave in realtime. This way, I can use MIDI CC to adjust the cutoff while playing to see how the opcode feels using it.

There was a noticeable click at the start of every note. The cause is the GRU cold start: `init()` resets the hidden state to zeros, but the model was trained with a 2048-sample warmup period before loss was ever calculated. The GRU had never learned to produce correct output from a dead zero state, only from a warmed-up one. So the first ~42ms of every note is the GRU catching up, which manifests as a transient.

The fix is to pre-warm the GRU at `init()` time by running 2048 samples of silence through the model at the initial cutoff before any real audio arrives. This replicates the training warmup. Added to the end of `init()`:

```cpp
float knob = normalizeKnob((float)inargs[2]);
float convIn[1] = {0.0f};
float gruIn[17] = {};
gruIn[16] = knob;
for (int i = 0; i < 2048; i++) {
    model->conv.forward(convIn);
    std::copy(model->conv.getOutputs(), model->conv.getOutputs() + 16, gruIn);
    gruIn[16] = knob;
    model->rec.forward(gruIn);
}
```

This runs at i-time so there's no per-block overhead. The click is gone.

## Live MIDI test: test_midi_saw.csd

Wrote a live MIDI test file to play the opcode in real time. A sawtooth oscillator driven by MIDI note data goes through the moognn filter, with CC 110 controlling the cutoff.

The cutoff is mapped on a log scale from 20Hz to 20kHz, matching how the training data was generated:

```csound
kcc     ctrl7 1, 110, 0, 1
kcc     = portk:k(kcc, 0.01)    ; smooth out CC steps
kcutoff = 20 * pow(1000, kcc)
```

`portk` with a 10ms smoothing time prevents zipper noise from discrete CC steps.

Initial attempt had about half a second of latency. Csound defaults to large hardware buffers. Fixed by adding `-b 256 -B 4096` to CsOptions, which drops latency to ok levels.

Also `-M a` opens all MIDI input devices at once so any connected controller works without specifying a device number.

Using this program exposed another issue. When playing 3 or more notes at a time, the playback cuts out for a bit. I would guess that the init function can't process fast enough, causing audio dropouts when too many instances of the opcode are initialized at once. Could be the pre warm loop or more likely parsing of the JSON file.

To debug, add timing to init(). Add chrono as an include and time the file parse, alloc, weights and warmup, then print the timings.

this prints this:

```bash
moognn init: file+parse=7.99ms  alloc=0.01ms  weights=0.16ms  warmup=2.90ms  total=11.07ms
```

This shows that the json file IO and parsing is the big issue. with a ksmps of 64 at 48kHz, each block is 1.3ms. three notes causes 33ms of blocked init, which means 25 missed callbacks.

The fix is to cache the loaded model globally, keyed by file path. The first note pays the full ~11ms cost. Every subsequent note skips the file IO and JSON parse, and just copy-constructs from the cached model and calls `reset()` to clear the hidden state. RTNeural's ModelT uses Eigen fixed-size matrices which are copy-constructible, so this is effectively a fast memcpy of the weights with a fresh GRU state.

```cpp
static std::unordered_map<std::string, Model*> g_model_cache;

// In init():
if (g_model_cache.find(path) == g_model_cache.end()) {
    // load from disk and parse JSON — once per unique path
    Model *tmpl = new Model();
    // ... load weights ...
    g_model_cache[path] = tmpl;
}
model = new Model(*g_model_cache[path]);  // copy weights
model->conv.reset();
model->rec.reset();
// pre-warm as before
```

The copy-construction approach broke the opcode entirely, making it pass through audio with no filtering. Turned out RTNeural's `Conv1DT` has an `Eigen::Map` member called `outs` that gets initialized in the constructor to point to its own internal `outs_internal` raw array. The default copy constructor copies the Map's pointer as-is, so the copy's `outs` still points to the original's `outs_internal`. Every voice ends up reading from and writing to the same output buffer. The model outputs garbage, and with the skip connection (`result + input`), garbage near zero sounds like a passthrough.

The fix is to cache the parsed JSON instead of the Model. Each voice loads weights from the in-memory JSON (0.16ms) rather than from disk, allocates its own Model, and gets a clean unaliased output buffer. Per-note cost is now about 3ms (weight load from memory plus pre-warm), which is well within budget.

```cpp
static std::unordered_map<std::string, nlohmann::json> g_json_cache;

if (g_json_cache.find(path) == g_json_cache.end()) {
    std::ifstream f(path);
    g_json_cache[path] = nlohmann::json::parse(f);
}
const nlohmann::json& j = g_json_cache[path];
model = new Model();
// load weights from in-memory JSON as normal
```

Eventually, the model json can be fully embedded into the plugin, but it can be loaded like this for dev purposes.

## Fixing the note-on click properly

With the dropout fix in place, the clicking at note-on came back. The original silence pre-warm was removed and replaced with caching, which masked the issue. Time to understand it properly.

Checked `tensor_torch_param.py` to see exactly what the training warmup was doing:

```python
loss = esr_loss(pred[:, warmup_size:, :], yb[:, warmup_size:, :])
```

The model receives `warmup_size + window_size` samples of real audio, but loss is only calculated on the window portion. The GRU starts from `h0 = zeros` and sees 2048 samples of real audio before it needs to be accurate. It was never trained on silence during warmup.

The silence pre-warm I had added was based on wrong reasoning. It conditioned the GRU for silence rather than letting it converge on actual audio. When real audio then arrived, the GRU had to reconverge anyway, potentially making the transient worse. The devlog had already flagged this as a known limitation back when evaluating the model: the first 42ms of any new note is outside the training loss window, so the model has no guarantee of accuracy there.

The correct fix is an output fade-in that covers the convergence window. No pre-warm needed. The GRU starts from zero and sees real audio exactly as in training, and the fade hides the inaccurate startup period.

Tried 512 samples (10ms) and 1024 samples (21ms), both still had audible clicks. 2048 samples (42ms) eliminates the click completely, which makes sense since that matches the full training warmup length. Left it at 2048 for now.

```cpp
static constexpr uint32_t FADE_SAMPLES = 2048;

// in aperf():
float fade = (fade_counter < FADE_SAMPLES) ? (float)fade_counter / FADE_SAMPLES : 1.0f;
out[i] = (MYFLT)((result + convIn[0]) * fade);
if (fade_counter < FADE_SAMPLES) fade_counter++;
```

The fade is linear over 42ms. With a 10ms attack envelope on the oscillator input, the combined effect is a gentle ramp that covers the full GRU convergence window without sounding like a slow attack on normal playing.

42ms is too long to be practical without an ADSR. The goal is to get the fade short enough that the opcode can be used as a raw filter with no envelope and no audible click on note-on.

Two experiments are worth running to get there.

**Experiment 1: shorter warmup in training**

The `warmup_size = 2048` in `tensor_torch_param.py` defines the requirement: be accurate after X samples from a cold start. Reducing it forces the model to converge faster, because the loss kicks in sooner.

Try 512 first, then 256. If the retrained model is click-free at those fade lengths in the opcode, the problem is solved without touching the architecture. This is the lowest-risk experiment and should be run first.

There is no hard minimum floor from the conv1d. Looking at the training script, `CausalConv1d.forward()` zero-pads every window on the left by `kernel_size - 1 = 30` samples before applying the convolution. This means every training window starts with a zero-filled kernel buffer, which is exactly the same state as a freshly reset `Conv1DT` in RTNeural at note-on. The model has seen this condition for every window across the entire training run. The conv1d empty buffer at note-on is not a special edge case and is not contributing to the click. The warmup length is purely a GRU convergence problem.

**Experiment 2: remove conv1d**

No data exists in this project on a conv1d vs no-conv1d comparison. The decision to keep it was reasoned, not measured. This experiment is worth running to measure the accuracy difference, but removing conv1d will not help with warmup length since the conv buffer state at note-on already matches training exactly. The only motivation here is accuracy vs complexity, not cold-start behavior.

This experiment only makes sense if experiment 1 shows the architecture can't converge fast enough. The accuracy hit might not be worth it.

**What neither experiment addresses**

Without `knob_to_h0`, the GRU has no prior knowledge of the target filter state at note-on. It has to infer the correct filter mode from the knob value embedded in the audio stream over time. That inference cost is the fundamental cause of the long warmup, and it doesn't go away by shortening the training warmup or removing conv1d. Those just force the model to learn to converge faster, not to start in the right state.

The C++ processing script (`process_wav_torch_param.cpp`) demonstrates the solved case: `prepareModel()` seeds the GRU from `knob_to_h0` and needs zero warmup. The devlog ablation shows `knob_to_h0` contributes about 3dB of accuracy and provides much cleaner convergence behavior.

If both experiments fail to get below a practical warmup length, adding `knob_to_h0` back into the opcode is the right call. The custom inference path already exists in `process_wav_torch_param.cpp` and the architecture exists in the ref models. For the paper this is also the more interesting result. It directly validates the architectural decision from the ablation study with a real-world consequence.

## Training with 512 sample warmup

Training script already has a `warmup_size` var. Changing it then running the training is all that's needed to do this experiment.

Also added a tee in the script so all console output is automatically saved to `training.log` in the output dir.

Run 12: `warmup_size = 512`, same 64-unit architecture as run 11. Training ran 275 epochs (early stopped), 36.6 minutes. LR stepped 5 times: 1e-3 at start, then 5e-4 at epoch 116, 2.5e-4 at epoch 186, 1.25e-4 at epoch 209, 6.25e-5 at epoch 233, 3.13e-5 at epoch 256. Solid convergence, no major spikes.

```
 Freq (Hz)       ESR    ESR (dB)  Status
---------------------------------------------
        20    0.0123      -19.1dB  ok
        60    0.0003      -34.8dB  good
       100    0.0003      -35.4dB  good
       125    0.0003      -35.5dB  good
       250    0.0001      -38.9dB  good
       500    0.0000      -43.2dB  good
       800    0.0000      -45.7dB  good
      1000    0.0000      -46.5dB  good
      2000    0.0000      -47.4dB  good
      4000    0.0000      -46.1dB  good
      8000    0.0000      -45.8dB  good
     12000    0.0000      -47.1dB  good
     16000    0.0000      -47.3dB  good
     20000    0.0000      -46.3dB  good
```

Comparing against run 11 (same architecture, warmup 2048):

| Freq | Run 11 (warmup 2048) | Run 12 (warmup 512) |
|------|---------------------|---------------------|
| 20Hz | -24.9dB | -19.1dB |
| 60Hz | -35.0dB | -34.8dB |
| 250Hz | -41.3dB | -38.9dB |
| 500Hz | -42.8dB | -43.2dB |
| 1kHz | -43.2dB | -46.5dB |
| 8kHz | -43.9dB | -45.8dB |
| 20kHz | -46.5dB | -46.3dB |

The tradeoff is exactly what you'd expect. 20Hz dropped about 6dB and is now borderline "ok", and 60-250Hz lost a few dB. From 500Hz up, run 12 is actually slightly better than run 11. The shorter warmup forced the GRU to converge faster, and that learned behavior is stronger in the mid-high range where filter state history matters less.

The practical outcome is that `FADE_SAMPLES` in the opcode can drop from 2048 to 512 (42ms to ~10ms), which makes the note-on feel much more immediate. The cost is accuracy at very low cutoff frequencies.

Next: try `warmup_size = 256` to see how far the tradeoff goes.

## Testing run 12 in the MIDI opcode

Updated `FADE_SAMPLES` in `moognn.cpp` from 1024 to 512 to match the new training warmup, then pointed both test CSDs at `ref/12_moog_warmup512/weights.json`. Rebuilt the DLL and tested with `test_midi_saw.csd` using a MIDI controller with CC 110 mapped to cutoff.

The clicking was still pretty bad. Reducing the fade to 512 samples just exposes more of the convergence period instead of eliminating it.

The root cause is architectural, not about fade length. Without `knob_to_h0`, the GRU starts cold at every note-on with no knowledge of the target filter state. It has to infer the correct mode from the knob value in the input stream over time, and that inference takes longer than 512 samples regardless of how the training warmup is set. Shortening the training warmup just moves the goalposts.

The practical workaround is to run the opcode as an always-on send effect rather than instantiating it per note. One instance processes the mixed signal continuously, so the GRU state is never reset and the cold-start problem never happens. This is actually closer to how RTNeural is designed to be used: continuous streaming inference with a stateful GRU. It also mirrors how a real mono Moog works, where the filter is always on and all voices share it.

For true per-note independent filter instances, `knob_to_h0` is still the correct fix. The ablation showed it produces cleaner convergence and about 3dB better accuracy. The custom inference path for it already exists in `process_wav_torch_param.cpp` and the trained models with it are in the ref folder.

## Run 13: warmup_size = 256

Same 64-unit architecture, warmup halved again to 256 samples (~5ms). Early stopped at epoch 211 in 27.1 minutes. LR stepped 4 times: 5e-4 at epoch 109, 2.5e-4 at epoch 132, 1.25e-4 at epoch 152, 6.25e-5 at epoch 192. Best val_loss: 0.0002 — worse than run 12's 0.0001.

```
 Freq (Hz)       ESR    ESR (dB)  Status
---------------------------------------------
        20    0.0181      -17.4dB  ok
        60    0.0006      -32.0dB  good
       100    0.0005      -33.3dB  good
       125    0.0004      -33.6dB  good
       250    0.0003      -34.9dB  good
       500    0.0002      -37.4dB  good
       800    0.0001      -41.7dB  good
      1000    0.0000      -43.7dB  good
      2000    0.0000      -43.8dB  good
      4000    0.0000      -44.3dB  good
      8000    0.0000      -43.4dB  good
     12000    0.0001      -42.2dB  good
     16000    0.0001      -40.1dB  good
     20000    0.0001      -43.0dB  good
```

Full comparison across all three warmup runs:

| Freq | Run 11 (2048) | Run 12 (512) | Run 13 (256) |
|------|--------------|-------------|-------------|
| 20Hz | -24.9dB | -19.1dB | -17.4dB |
| 60Hz | -35.0dB | -34.8dB | -32.0dB |
| 250Hz | -41.3dB | -38.9dB | -34.9dB |
| 500Hz | -42.8dB | -43.2dB | -37.4dB |
| 1kHz | -43.2dB | -46.5dB | -43.7dB |
| 8kHz | -43.9dB | -45.8dB | -43.4dB |
| 16kHz | -45.0dB | -47.3dB | -40.1dB |

The trend inverted. Run 12 (512) had better mid-high accuracy than run 11 because the shorter window forced faster convergence. Run 13 (256) undoes that gain: 60-500Hz is now worse than both previous runs, and 12-16kHz dropped noticeably. The window is now short enough that the GRU doesn't get enough audio context during training to learn the filter behavior reliably. 20Hz continued its decline to -17.4dB.

Run 12 (warmup 512) was the sweet spot for this experiment series. Going shorter just degrades accuracy across the board.

Not even bothering with trying this model. I was training it while I was testing the previous one. The previous one didn't work so this one won't either, as in it'll still click when using the shorter warmup size.

## Research results

Did some research to see if any previous literature could guide me on what to do.

In regard to the cold start issue, there are many different approaches. Firstly, it has been refered to as "Exposure bias", where the GRU was trained on ground-truth hidden states, but runs on its own imperfect state at inference. TBPTT is the standard mitigation (what the warmup is a variation of), and state-matching is the next step (what k2h0 is). Techniques like HyperRNN do what knob_to_h0 do at init time during inference. FiLM is a more standard approach, where instead of generating weights like HyperRNN directly, it calculates scale and shifts that is applied to the feature activations. It has been found that placementof FiLM matters, and post-GRU and before Dense is the best.

Though FiLM is super light weight, it seems slightly overkill for this situation. For a single scalar knob, knob_to_h0 would be sufficient, and it won't disrupt the native RTNeural. However, I should consider adding it, as it conditions the GRU output at every sample, so it would help with dynamic parameter tracking.

So first, I will add k2h0 first to try to solve the cold start click, then add FiLM to better handle dynamic parameters and multiple parameters in the future.

I also researched how to best get training data. For automated, digital methods, adjusting parameters with an lfo seems standard. Instead of multiple values, you can collect data with each parameter at different lfo freqs from 1-20hz.

For analog systems, setting each param at 3-5 points and manually recording all combinations still seems to be sufficient.

The sound source is sine sweeps, white noise of different decibel levels, sine tones of different dB levels, music excerpts of different levels. The processes were varied between papers. No real consistent approach that I should follow, mainly targetting the effect that is being simulated (long gaps for reverbs, different decibels for compressions and distortions etc.).

Most literature agrees that LTSM and GRU are the best type of RNN for stateful effects. Some long history related effects like compressions that have 5 second attacks need to do creative architectures to handle longer memory, but won't be relevant for moog filters and distortions.

Also consider aliasing, especially for distortions. Antiderivative Antialiasing (ADAA) has become standard in the industry.

Consider teacher student models for simplification of the model to improve realtime performance. Can also help with anti aliasing.

Keep in mind sample rate. When a model is trained at a sample rate, it doesn't perform correctly in other sample rates at inference. There are solutions, but just note if it isn't mitigated.

There is a chirp-train metric that specifically measures how well a model tracks a sweeping parameter. This would provide a principled way to score a model's ability to handle dynamic parameters.

## Run 14: 64-unit model with knob_to_h0 and LayerNorm

Architecture: `Conv1d → LayerNorm → (concat knob) → GRU(17→64) → Dense(64→1) + skip`, with `knob_to_h0: Linear(1→64) + Tanh` seeding the GRU hidden state before each window. This restores both components from the ablation study and brings them together in the 64-unit configuration that the ablation suggested as the sweet spot.

Training ran the full 300 epochs in 52.2 minutes (~10.4s/epoch). The LR stepped 4 times: 1e-3 → 5e-4 at epoch 99, → 2.5e-4 at epoch 146, → 1.25e-4 at epoch 170, → 6.25e-5 at epoch 226. The val_loss dropped rapidly from 0.319 at epoch 1 to 0.0025 by epoch 23, then refined through the remaining LR steps before holding flat at 0.0001 from epoch 147 onward. Best val_loss: 0.0001.

Notable in the early training: epoch 1 had a train loss of 36.05, a gradient explosion on the first batch before clipping stabilised it. By epoch 2 it was already 0.35. This is normal for k2h0 at LR 1e-3 — the h0 seeding creates a sharp gradient signal initially.

```bash
$ python python/eval_param_model.py ref/14_moog_20-20k_64u_k2h0_LN/best_model.pt
Using device: cuda

 Freq (Hz)       ESR    ESR (dB)  Status
---------------------------------------------
        20    0.0032      -25.0dB  good
        60    0.0002      -36.6dB  good
       100    0.0001      -41.0dB  good
       125    0.0001      -41.8dB  good
       250    0.0001      -42.3dB  good
       500    0.0000      -43.7dB  good
       800    0.0000      -43.6dB  good
      1000    0.0000      -43.4dB  good
      2000    0.0001      -42.9dB  good
      4000    0.0000      -45.2dB  good
      8000    0.0000      -44.5dB  good
     12000    0.0000      -46.1dB  good
     16000    0.0000      -47.2dB  good
     20000    0.0000      -48.3dB  good
```

Comparing against the two most relevant prior runs:

| Freq | 32u + k2h0 + LN (run 09) | 64u no k2h0 no LN (run 11) | 64u + k2h0 + LN (run 14) |
|------|--------------------------|----------------------------|--------------------------|
| 20Hz | -27.5dB | -24.9dB | -25.0dB |
| 60Hz | -40.9dB | -35.0dB | -36.6dB |
| 500Hz | -48.6dB | -42.8dB | -43.7dB |
| 1kHz | -48.9dB | -43.2dB | -43.4dB |
| 8kHz | -48.1dB | -43.9dB | -44.5dB |
| 20kHz | — | -46.5dB | -48.3dB |

Run 14 is consistently 1–2dB better than run 11 across the board, and the high-frequency ceiling improved notably (−48.3dB at 20kHz vs −46.5dB). The mid-frequency scores are still behind the 32-unit run 09, which is counterintuitive given more units. The most likely explanation is the skip connection: run 09 didn't have it. Without the skip, the GRU had to model the full output signal, forcing more of the filter behavior into the GRU hidden state. With the skip connection, the model only needs to learn the residual, which reduces the pressure on the GRU to internalize the filter state — leaving less for k2h0 to leverage at low-mid frequencies.

The practical primary goal for this run wasn't raw accuracy — it was fixing the per-note click in the opcode. k2h0 seeds the GRU into the correct filter state before any audio arrives, so the cold-start problem that required a 2048-sample fade should be resolved. The next step is updating `moognn.cpp` to load and apply the k2h0 and LayerNorm weights.

## Opcode update: k2h0 and LayerNorm

Updated `moognn.cpp` to match the run 14 architecture. Added manual `LayerNorm` and `KnobToH0` structs (RTNeural has no native support for either). `init()` now loads `norm.weight/bias` and `knob_to_h0.0.weight/bias` from the JSON, resets conv and rec, then seeds `model->rec.get<0>().outs[]` from k2h0 before any audio arrives. `aperf()` applies LayerNorm between the conv copy and the GRU concat step.

Also added a 2048-sample fade-in on note-on. Even with k2h0 seeding the GRU correctly, the model was trained with a 2048-sample warmup before loss was computed, so it isn't guaranteed to produce accurate output from sample 0. The fade hides that convergence window.

Still clicking on note-on. Bypassing the opcode removes the click.

## Click debugging

Worked through several hypotheses to find the cause.

Added timing to `init()` to measure each section. Came back as `total=0.34ms` on the second note and beyond (JSON already cached). Far under the 1.33ms budget at ksmps=64/48kHz. Not the cause for those notes.

Added a cache warmup to `init()`: run one ksmps block of silence through the full inference path (conv + LayerNorm + GRU + Dense), then reset and re-seed h0. Idea was that cold Eigen matrix cache misses might spike the first `aperf()` call. Added first-aperf timing to confirm: `0.09ms`. Not the cause.

Added NaN/Inf detection on the first aperf output. Nothing reported. Ruled out NaN propagating through the fade (NaN * 0 = NaN by IEEE 754, which would bypass the fade).

Increasing ksmps from 64 to 256 didn't change anything.

Root cause found: on the first note of a session, before the JSON cache is populated, init prints `json=7.53ms, total=7.84ms`. This blows past the 1.33ms ksmps budget, causing a dropout that manifests as a click. Every subsequent note is `total=0.34ms` (cached) and click-free. ksmps=256 doesn't help because its budget is 5.33ms, still under 7.84ms.

Fix: pre-populate the JSON cache before any MIDI notes arrive. Can be done with a dummy instrument at score time 0 that instantiates the opcode once.

## Per-note click fix

The click debugging section concluded that subsequent notes were click-free, but that was wrong. The click was actually happening on every note from the third one onward, just masked during initial testing.

Added result-value logging to aperf() to print `raw`, `fade`, and `out` for the first 20 samples of a note's first aperf call. Playing the first two notes showed clean output: raw values around -0.12 to +0.15, faded output near zero. No transients.

Playing note 3 showed nothing at all. No s0-s19 lines. The `is_first` flag (`fade_counter == 0`) was never true, meaning aperf was running with fade_counter already at 2048 and the fade was at 1.0 from sample 0: full volume, no ramp, instant click.

Root cause: Csound reuses instrument memory blocks between notes without calling C++ constructors. The struct's default member initializer (`fade_counter = 0`) only runs at construction time. When note 1 ends and its memory is returned to Csound's pool, `fade_counter` is still sitting at 2048. When note 3 starts and gets that same memory, `fade_counter = 2048` and the entire fade is skipped.

Fix: explicitly reset `fade_counter = 0` at the top of `init()`. One line.

The first-note click from the 7.84ms JSON parse is still a separate unresolved issue.

## First-note JSON cache fix

Added a `moognn_preload` opcode to pre-warm the JSON cache before any MIDI notes arrive. It's i-rate only, takes a path string, and just does the JSON parse and cache insertion. No model allocation, no weight loading.

```csound
instr 99
  iignore moognn_preload "ref/14_moog_20-20k_64u_k2h0_LN/weights.json"
endin
```

```
i 99 0 0
```

Instr 99 fires at t=0, before any keys are pressed. The 7.84ms parse happens then with nothing playing, so the dropout is inaudible. Every `moognn init` after that shows `json=0.00ms`.

Hit a segfault on first attempt. The opcode was declared as `Plugin<0, 1>` (zero outputs) which causes CPOF to miscalculate the inarg pointer offsets. Fix: `Plugin<1, 1>` with a dummy i-rate output that gets assigned 0 and ignored by the CSD.

My main question now is that since the clicks were cause by a bug that made the fade in not work properly, maybe the previous models would work fine?

## CPU cache warmup loop removal

`init()` had a 64-sample silence loop before the h0 seed. The reasoning was that cold Eigen matrix cache misses on the first `aperf()` call could spike latency above the k-period budget. This was added as a precaution before the real click causes were found.

Added first-aperf timing to measure it properly. With the loop: 0.47ms, 0.42ms. Without: 0.37ms, 0.50ms, 0.38ms, 0.38ms, 0.40ms. No difference, all within normal jitter. The loop also reset the model state immediately after, so it had zero effect on audio. Removed.

## Run 15: k2h0 with 512 warmup samples

Running a training while i experiment with run 14 with 512 warmup samples. If this model works well, ideally bring it down to 256 samples next.
