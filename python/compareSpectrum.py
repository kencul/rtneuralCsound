import sys
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <target.wav> <model_output.wav>")
    sys.exit(1)

target_audio, sr = librosa.load(sys.argv[1], sr=None)
model_audio, _ = librosa.load(sys.argv[2], sr=sr)

# Truncate to equal length to allow element-wise matrix subtraction
min_len = min(len(target_audio), len(model_audio))
target_audio = target_audio[:min_len]
model_audio = model_audio[:min_len]

# Compute Short-Time Fourier Transform to extract frequency magnitudes over time
n_fft = 2048
hop_length = 512
stft_target = librosa.stft(target_audio, n_fft=n_fft, hop_length=hop_length)
stft_model = librosa.stft(model_audio, n_fft=n_fft, hop_length=hop_length)

# Convert to decibel scale to match logarithmic human hearing perception
db_target = librosa.amplitude_to_db(np.abs(stft_target), ref=np.max)
db_model = librosa.amplitude_to_db(np.abs(stft_model), ref=np.max)

# Calculate absolute difference to isolate frequency ranges with high prediction error
db_diff = np.abs(db_target - db_model)

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
librosa.display.specshow(db_target, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
plt.title('Target Audio Spectrogram')
plt.colorbar(format='%+2.0f dB')

plt.subplot(3, 1, 2)
librosa.display.specshow(db_model, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
plt.title('Model Output Spectrogram')
plt.colorbar(format='%+2.0f dB')

plt.subplot(3, 1, 3)
librosa.display.specshow(db_diff, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', cmap='magma')
plt.title('Difference Spectrogram')
plt.colorbar(format='%+2.0f dB')

plt.tight_layout()
plt.show()