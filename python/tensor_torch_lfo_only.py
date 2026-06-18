import torch
import numpy as np
import librosa
import json
import csv
import time
import sys
import os
from json import JSONEncoder
from model_concat import Model
from torch.utils.data import DataLoader, TensorDataset

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <output_dir>")
    print(f"  e.g. {sys.argv[0]} models/24_moog_lfo_only_128u_w256")
    sys.exit(1)

OUT_DIR = sys.argv[1]
os.makedirs(OUT_DIR, exist_ok=True)

class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
    def flush(self):
        for s in self._streams:
            s.flush()

_log = open(os.path.join(OUT_DIR, "training.log"), "w")
sys.stdout = _Tee(sys.__stdout__, _log)
sys.stderr = _Tee(sys.__stderr__, _log)

FREQ_MIN = 100.0
FREQ_MAX = 20000.0

CUTOFF_FREQS = [100, 125, 250, 500, 800, 1000, 2000, 4000, 8000, 12000, 16000, 20000]

VARIABLE_TRAIN_FILES = [
    (
        "audio/testSound_mono.wav",
        f"audio/filteredOutput/testSound/testSound_mono_lfo_{rate}_100-10khz.wav",
        f"audio/filteredOutput/testSound/testSound_mono_lfo_{rate}_100-10khz.csv",
    )
    for rate in ("1hz", "2hz", "fast", "10hz", "20hz")
]

VARIABLE_VAL_FILES = [
    (
        "audio/bench_mono.wav",
        "audio/filteredOutput/bench/bench_mono_lfo_fast_100-10khz.wav",
        "audio/filteredOutput/bench/bench_mono_lfo_fast_100-10khz.csv",
    ),
]


def normalize_knob(freq_hz):
    return (np.log(freq_hz) - np.log(FREQ_MIN)) / (np.log(FREQ_MAX) - np.log(FREQ_MIN))


class EncodeTensor(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super().default(obj)


GRU_HIDDEN = 128

window_size = 8192
warmup_size = 256


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


def load_conditioned_windows(dry_path, wet_path, knob_value_normalized, sr=None):
    x, sr_ = librosa.load(dry_path, sr=sr, mono=True)
    y, _   = librosa.load(wet_path, sr=sr_, mono=True)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    knob_channel = np.ones_like(x) * knob_value_normalized
    x_stacked = np.stack((x, knob_channel), axis=-1)

    total_size = window_size + warmup_size
    X_list, Y_list = [], []
    for i in range(0, n - total_size, window_size):
        X_list.append(x_stacked[i : i + total_size])
        Y_list.append(y[i : i + total_size])

    X = np.array(X_list).reshape(-1, total_size, 2)
    Y = np.array(Y_list).reshape(-1, total_size, 1)
    return X, Y, sr_


def load_knob_schedule(csv_path):
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    return data[:, 1].astype(np.float32)


def load_variable_windows(dry_path, wet_path, csv_path, sr=None):
    x, sr_ = librosa.load(dry_path, sr=sr, mono=True)
    y, _   = librosa.load(wet_path, sr=sr_, mono=True)
    knob   = load_knob_schedule(csv_path)
    n = min(len(x), len(y), len(knob))
    x, y, knob = x[:n], y[:n], knob[:n]

    total_size = window_size + warmup_size
    X_list, Y_list = [], []
    for i in range(0, n - total_size, window_size):
        x_slice    = x[i : i + total_size]
        knob_slice = knob[i : i + total_size]
        X_list.append(np.stack((x_slice, knob_slice), axis=-1))
        Y_list.append(y[i : i + total_size])

    X = np.array(X_list).reshape(-1, total_size, 2)
    Y = np.array(Y_list).reshape(-1, total_size, 1)
    return X, Y, sr_


print("Loading LFO training data...")
sr = None
X_train, Y_train = None, None
for dry_path, wet_path, csv_path in VARIABLE_TRAIN_FILES:
    X_var, Y_var, sr = load_variable_windows(dry_path, wet_path, csv_path, sr=sr)
    print(f"  {os.path.basename(wet_path)}: {len(X_var)} windows")
    if X_train is None:
        X_train, Y_train = X_var, Y_var
    else:
        X_train = np.concatenate([X_train, X_var])
        Y_train = np.concatenate([Y_train, Y_var])

print("Loading validation data...")
X_val, Y_val, _ = load_all_conditioned(
    dry_path="audio/bench_mono.wav",
    wet_dir="audio/filteredOutput/bench",
    wet_pattern="bench_mono_{freq}hz.wav",
    sr=sr,
)

print("Loading variable validation data...")
for dry_path, wet_path, csv_path in VARIABLE_VAL_FILES:
    X_vvar, Y_vvar, _ = load_variable_windows(dry_path, wet_path, csv_path, sr=sr)
    print(f"  {os.path.basename(wet_path)}: {len(X_vvar)} windows")
    X_val = np.concatenate([X_val, X_vvar])
    Y_val = np.concatenate([Y_val, Y_vvar])

print(f"Train windows: {len(X_train)}, Val windows: {len(X_val)}, Sample rate: {sr}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
X_val   = torch.tensor(X_val,   dtype=torch.float32).to(device)
Y_val   = torch.tensor(Y_val,   dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=64, shuffle=True)

model     = Model(gru_hidden=GRU_HIDDEN).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, min_lr=1e-6)


def pre_emphasis(x, coef=0.95):
    return torch.cat([x[:, :1, :], x[:, 1:, :] - coef * x[:, :-1, :]], dim=1)


def esr_loss(pred, target):
    pred   = pre_emphasis(pred)
    target = pre_emphasis(target)
    error  = torch.mean((pred - target) ** 2, dim=(1, 2))
    energy = torch.clamp(torch.mean(target ** 2, dim=(1, 2)), min=1e-4)
    return torch.mean(error / energy)


best_val_loss = float('inf')
early_stop_patience = 40
epochs_without_improvement = 0
epochs = 300
train_losses = []
val_losses   = []
train_start = time.time()
for epoch in range(epochs):
    epoch_start = time.time()
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred, _ = model(xb)
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
            pred, _ = model(xb)
            val_loss += esr_loss(pred[:, warmup_size:, :], yb[:, warmup_size:, :]).item() * len(xb)
        val_loss /= len(X_val)

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(
            {'model_state': model.state_dict(), 'arch': 'concat', 'gru_hidden': GRU_HIDDEN, 'freq_min': FREQ_MIN, 'freq_max': FREQ_MAX},
            os.path.join(OUT_DIR, 'best_model.pt')
        )
    else:
        epochs_without_improvement += 1

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    lr = optimizer.param_groups[0]['lr']
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - lr: {lr:.2e} - {epoch_time:.1f}s")

    if epochs_without_improvement >= early_stop_patience:
        print(f"Early stopping: val_loss has not improved for {early_stop_patience} epochs.")
        break

total_time = time.time() - train_start
print(f"Training complete in {total_time/60:.1f}m ({total_time:.0f}s)")
print(f"Best val_loss: {best_val_loss:.4f} -- loading best weights for export")
model.load_state_dict(torch.load(os.path.join(OUT_DIR, 'best_model.pt'))['model_state'])

weights_path = os.path.join(OUT_DIR, 'weights.json')
with open(weights_path, 'w') as f:
    json.dump(model.state_dict(), f, cls=EncodeTensor, indent=4)
print(f"Model saved to {weights_path}")

csv_path = os.path.join(OUT_DIR, 'loss_history.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'val_loss'])
    for i, (t, v) in enumerate(zip(train_losses, val_losses), 1):
        writer.writerow([i, t, v])
print(f"Loss history saved to {csv_path}")
