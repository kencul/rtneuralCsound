import numpy as np
from scipy.io import wavfile

sample_rate = 48000
duration = 15
num_samples = sample_rate * duration

# Generate white noise, then shape its spectrum to 1/f (pink noise)
white = np.random.randn(num_samples)
fft = np.fft.rfft(white)
freqs = np.fft.rfftfreq(num_samples, d=1.0 / sample_rate)

# Avoid division by zero at DC; scale amplitudes by 1/sqrt(f) for pink spectrum
freqs[0] = 1
fft /= np.sqrt(freqs)
fft[0] = 0  # zero DC component

pink = np.fft.irfft(fft, n=num_samples)
pink /= np.max(np.abs(pink))  # normalize to [-1, 1]

wavfile.write('pink_noise.wav', sample_rate, pink.astype(np.float32))