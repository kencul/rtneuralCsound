import numpy as np
from scipy.signal import chirp
from scipy.io import wavfile

# Create a time array for 1 second at 48kHz sample rate
t = np.linspace(0, 1, 48000 * 1)

# Generate a logarithmic sine sweep from 20Hz to 20kHz over the 1 second
sweep = chirp(t, f0=20, f1=20000, t1=1, method='logarithmic')

# Export the generated signal to a 32-bit floating point WAV file
wavfile.write('sweep.wav', 48000, sweep.astype(np.float32))