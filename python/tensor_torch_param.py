import torch
import torch.nn as nn
import numpy as np
import librosa
import json
import time
from json import JSONEncoder
from torch.utils.data import DataLoader, TensorDataset

FREQ_MIN = 20.0
FREQ_MAX = 20000.0

CUTOFF_FREQS = [20, 60, 100, 125, 250, 500, 800, 1000, 2000, 4000, 8000, 12000, 16000, 20000]


def normalize_knob(freq_hz):
    return (np.log(freq_hz) - np.log(FREQ_MIN)) / (np.log(FREQ_MAX) - np.log(FREQ_MIN))


class EncodeTensor(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super().default(obj)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)


GRU_HIDDEN = 32

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = CausalConv1d(in_channels=1, out_channels=16, kernel_size=31)
        self.norm = nn.LayerNorm(16)
        # maps knob scalar to GRU initial hidden state so the model starts in the right filter mode
        self.knob_to_h0 = nn.Sequential(nn.Linear(1, GRU_HIDDEN), nn.Tanh())
        self.gru = nn.GRU(17, GRU_HIDDEN, batch_first=True)
        self.dense = nn.Linear(GRU_HIDDEN, 1)

    def forward(self, x):
        # x: (batch, time, 2) - channel 0 = audio, channel 1 = knob
        audio = x[:, :, :1]  # (batch, time, 1)
        knob  = x[:, :, 1:]  # (batch, time, 1)

        audio = audio.permute(0, 2, 1)
        audio = self.conv(audio)
        audio = audio.permute(0, 2, 1)
        audio = self.norm(audio)  # normalize before mixing with knob

        gru_input = torch.cat([audio, knob], dim=-1)  # (batch, time, 17)

        # knob is constant per window - use first timestep to seed h0
        h0 = self.knob_to_h0(knob[:, 0, :]).unsqueeze(0)  # (1, batch, 128)
        out, _ = self.gru(gru_input, h0)
        return self.dense(out)


window_size = 8192
warmup_size = 2048


def load_conditioned_windows(dry_path, wet_path, knob_value_normalized, sr=None):
    x, sr_ = librosa.load(dry_path, sr=sr, mono=True)
    y, _   = librosa.load(wet_path, sr=sr_, mono=True)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    knob_channel = np.ones_like(x) * knob_value_normalized
    x_stacked = np.stack((x, knob_channel), axis=-1)  # shape (n, 2)

    total_size = window_size + warmup_size
    X_list, Y_list = [], []
    for i in range(0, n - total_size, window_size):
        X_list.append(x_stacked[i : i + total_size])
        Y_list.append(y[i : i + total_size])

    X = np.array(X_list).reshape(-1, total_size, 2)
    Y = np.array(Y_list).reshape(-1, total_size, 1)
    return X, Y, sr_


def load_all_conditioned(dry_path, wet_dir, wet_pattern, sr=None):
    all_X, all_Y = [], []
    detected_sr = sr
    for freq in CUTOFF_FREQS:
        wet_path = f"{wet_dir}/{wet_pattern.format(freq=freq)}"
        knob = normalize_knob(freq)
        X, Y, detected_sr = load_conditioned_windows(dry_path, wet_path, knob, sr=detected_sr)
        all_X.append(X)
        all_Y.append(Y)
        print(f"  {freq}Hz (knob={knob:.3f}): {len(X)} windows")
    return np.concatenate(all_X), np.concatenate(all_Y), detected_sr


print("Loading training data...")
X_train, Y_train, sr = load_all_conditioned(
    dry_path="audio/testSound_mono.wav",
    wet_dir="audio/filteredOutput/testSound",
    wet_pattern="testSound_mono_{freq}hz.wav",
)

print("Loading validation data...")
X_val, Y_val, _ = load_all_conditioned(
    dry_path="audio/bench_mono.wav",
    wet_dir="audio/filteredOutput/bench",
    wet_pattern="bench_mono_{freq}hz.wav",
    sr=sr,
)

print(f"Train windows: {len(X_train)}, Val windows: {len(X_val)}, Sample rate: {sr}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
X_val   = torch.tensor(X_val,   dtype=torch.float32).to(device)
Y_val   = torch.tensor(Y_val,   dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=64, shuffle=True)

model     = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, min_lr=1e-6)


def pre_emphasis(x, coef=0.95):
    return torch.cat([x[:, :1, :], x[:, 1:, :] - coef * x[:, :-1, :]], dim=1)


def esr_loss(pred, target):
    pred   = pre_emphasis(pred)
    target = pre_emphasis(target)
    # per-window ESR so low-cutoff (near-silence) windows don't get crushed by
    # high-energy windows sharing the same denominator
    error  = torch.mean((pred - target) ** 2, dim=(1, 2))
    # clamp to a real noise floor so near-silence windows don't blow up the loss
    energy = torch.clamp(torch.mean(target ** 2, dim=(1, 2)), min=1e-4)
    return torch.mean(error / energy)


best_val_loss = float('inf')
early_stop_patience = 40
epochs_without_improvement = 0
epochs = 300
train_start = time.time()
for epoch in range(epochs):
    epoch_start = time.time()
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = esr_loss(pred[:, warmup_size:, :], yb[:, warmup_size:, :])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= len(X_train)

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for xb, yb in DataLoader(TensorDataset(X_val, Y_val), batch_size=64):
            pred = model(xb)
            val_loss += esr_loss(pred[:, warmup_size:, :], yb[:, warmup_size:, :]).item() * len(xb)
        val_loss /= len(X_val)

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'best_model_param.pt')
    else:
        epochs_without_improvement += 1

    lr = optimizer.param_groups[0]['lr']
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - lr: {lr:.2e} - {epoch_time:.1f}s")

    if epochs_without_improvement >= early_stop_patience:
        print(f"Early stopping: val_loss has not improved for {early_stop_patience} epochs.")
        break

total_time = time.time() - train_start
print(f"Training complete in {total_time/60:.1f}m ({total_time:.0f}s)")
print(f"Best val_loss: {best_val_loss:.4f} — loading best weights for export")
model.load_state_dict(torch.load('best_model_param.pt'))

with open('rtneural_model_param_weights.json', 'w') as f:
    json.dump(model.state_dict(), f, cls=EncodeTensor, indent=4)

print("Model saved to rtneural_model_param_weights.json")
