import numpy as np
from scipy.io import wavfile

sample_rate = 48000
burst_samples = int(sample_rate * 0.01)   # 10 ms
silence_samples = int(sample_rate * 0.5)  # 500 ms

velocities = [0.10, 0.25, 0.50, 0.75, 1.00]

window = np.hanning(burst_samples)
silence = np.zeros(silence_samples)

segments = []
for v in velocities:
    noise = np.random.uniform(-1.0, 1.0, burst_samples)
    burst = window * noise * v
    segments.append(burst)
    segments.append(silence)

signal = np.concatenate(segments).astype(np.float32)
wavfile.write('velocity_bursts.wav', sample_rate, signal)