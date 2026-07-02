import torch
import torch.nn as nn
import numpy as np
import librosa
import json
import csv
import time
import sys
import os
from json import JSONEncoder
from model_distortion import Model
from torch.utils.data import DataLoader, TensorDataset

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print(f"Usage: {sys.argv[0]} <output_dir> [gru_hidden]")
    print(f"  e.g. {sys.argv[0]} models/dist_10_gru128 128")
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

GRU_HIDDEN  = int(sys.argv[2]) if len(sys.argv) == 3 else 128
CELL        = 'gru'
window_size = 8192
warmup_size = 256
VAL_FRAC    = 0.2
SEED        = 42

PAIRS = [
    ("audio/updatedDistortion/trainingDry.wav",    "audio/updatedDistortion/trainingWet.wav"),
    ("audio/updatedDistortion/training-10dBDry.wav", "audio/updatedDistortion/training-10dBWet.wav"),
]


class EncodeTensor(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super().default(obj)


def load_and_split(dry_path, wet_path, sr=None):
    x, sr_ = librosa.load(dry_path, sr=sr,  mono=True)
    y, _   = librosa.load(wet_path,  sr=sr_, mono=True)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    total_size = window_size + warmup_size
    X_list, Y_list = [], []
    for i in range(0, n - total_size, window_size):
        X_list.append(x[i : i + total_size])
        Y_list.append(y[i : i + total_size])

    X = np.array(X_list).reshape(-1, total_size, 1)
    Y = np.array(Y_list).reshape(-1, total_size, 1)

    rng     = np.random.default_rng(SEED)
    idx     = rng.permutation(len(X))
    n_val   = int(len(X) * VAL_FRAC)
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    return X[tr_idx], Y[tr_idx], X[val_idx], Y[val_idx], sr_


print("Loading training data...")
sr = None
X_train_parts, Y_train_parts = [], []
X_val_parts,   Y_val_parts   = [], []

for dry_path, wet_path in PAIRS:
    Xtr, Ytr, Xv, Yv, sr = load_and_split(dry_path, wet_path, sr=sr)
    X_train_parts.append(Xtr)
    Y_train_parts.append(Ytr)
    X_val_parts.append(Xv)
    Y_val_parts.append(Yv)
    print(f"  {os.path.basename(dry_path)}: {len(Xtr)} train, {len(Xv)} val windows")

X_train = np.concatenate(X_train_parts)
Y_train = np.concatenate(Y_train_parts)
X_val   = np.concatenate(X_val_parts)
Y_val   = np.concatenate(Y_val_parts)
print(f"Total: {len(X_train)} train, {len(X_val)} val windows  sr={sr}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
X_val   = torch.tensor(X_val,   dtype=torch.float32).to(device)
Y_val   = torch.tensor(Y_val,   dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=64, shuffle=True)

model     = Model(gru_hidden=GRU_HIDDEN, cell=CELL).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, min_lr=1e-6)

_MRSTFT_PARAMS = [
    (2048, 512,  2048),
    (1024, 256,  1024),
    ( 512, 128,   512),
]


def _stft_log_mag(x, fft_size, hop_size, win_length):
    w = torch.hann_window(win_length, device=x.device)
    X = torch.stft(x, fft_size, hop_size, win_length, w, return_complex=True)
    mag = torch.sqrt(torch.clamp(X.real**2 + X.imag**2, min=1e-8))
    return torch.log(mag)


# L1 + log-magnitude MR-STFT per Comunita et al. (ICASSP 2023).
# Spectral convergence omitted: its denominator norm can blow up on near-silent
# windows, causing NaN gradients that corrupt weights irreversibly.
def combined_loss(pred, target):
    p = pred.squeeze(2)
    t = target.squeeze(2)
    l_mae = nn.functional.l1_loss(p, t)
    l_mrstft = 0.0
    for fft_size, hop_size, win_length in _MRSTFT_PARAMS:
        l_mrstft += nn.functional.l1_loss(
            _stft_log_mag(p, fft_size, hop_size, win_length),
            _stft_log_mag(t, fft_size, hop_size, win_length),
        )
    l_mrstft = l_mrstft / len(_MRSTFT_PARAMS)
    return l_mae + l_mrstft, l_mae, l_mrstft


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
    train_loss = train_mae = train_mrstft = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred, _ = model(xb)
        loss, l_mae, l_mrstft = combined_loss(pred[:, warmup_size:, :], yb[:, warmup_size:, :])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        n = len(xb)
        train_loss   += loss.item()     * n
        train_mae    += l_mae.item()    * n
        train_mrstft += l_mrstft.item() * n
    train_loss   /= len(X_train)
    train_mae    /= len(X_train)
    train_mrstft /= len(X_train)

    model.eval()
    with torch.no_grad():
        val_loss = val_mae = val_mrstft = 0.0
        for xb, yb in DataLoader(TensorDataset(X_val, Y_val), batch_size=64):
            pred, _ = model(xb)
            loss, l_mae, l_mrstft = combined_loss(pred[:, warmup_size:, :], yb[:, warmup_size:, :])
            n = len(xb)
            val_loss   += loss.item()     * n
            val_mae    += l_mae.item()    * n
            val_mrstft += l_mrstft.item() * n
        val_loss   /= len(X_val)
        val_mae    /= len(X_val)
        val_mrstft /= len(X_val)

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(
            {'model_state': model.state_dict(), 'arch': f'distortion_{CELL}', 'gru_hidden': GRU_HIDDEN},
            os.path.join(OUT_DIR, 'best_model.pt')
        )
    else:
        epochs_without_improvement += 1

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    lr = optimizer.param_groups[0]['lr']
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} (mae {train_mae:.4f} mrstft {train_mrstft:.4f})"
          f" - val: {val_loss:.4f} (mae {val_mae:.4f} mrstft {val_mrstft:.4f})"
          f" - lr: {lr:.2e} - {epoch_time:.1f}s")

    if epochs_without_improvement >= early_stop_patience:
        print(f"Early stopping: val_loss has not improved for {early_stop_patience} epochs.")
        break

total_time = time.time() - train_start
print(f"Training complete in {total_time/60:.1f}m ({total_time:.0f}s)")
print(f"Best val_loss: {best_val_loss:.4f}")

checkpoint = torch.load(os.path.join(OUT_DIR, 'best_model.pt'))
model.load_state_dict(checkpoint['model_state'])

weights_path = os.path.join(OUT_DIR, 'weights.json')
with open(weights_path, 'w') as f:
    json.dump(model.state_dict(), f, cls=EncodeTensor, indent=4)
print(f"Weights exported to {weights_path}")

csv_path = os.path.join(OUT_DIR, 'loss_history.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'val_loss'])
    for i, (t, v) in enumerate(zip(train_losses, val_losses), 1):
        writer.writerow([i, t, v])
print(f"Loss history saved to {csv_path}")
