import numpy as np
from scipy.io import wavfile

sample_rate = 48000
num_samples = int(sample_rate * 0.)  # 100 ms

white = np.random.uniform(-1.0, 1.0, num_samples)

wavfile.write('white_noise.wav', sample_rate, white.astype(np.float32))