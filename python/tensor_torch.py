import torch
import torch.nn as nn
import numpy as np
import librosa
import json
import time
from json import JSONEncoder
from torch.utils.data import DataLoader, TensorDataset


class EncodeTensor(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super().default(obj)


# PyTorch has no built-in causal padding, so we pad manually on the left only
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = CausalConv1d(1, 16, 31)
        self.gru = nn.GRU(16, 64, batch_first=True)
        self.dense = nn.Linear(64, 1)

    def forward(self, x):
        # Conv1d expects (batch, channels, length), so permute from (batch, length, channels)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        return self.dense(x)


window_size = 8192
warmup_size = 2048

def load_windows(input_path, output_path, sr=None):
    x, sr_ = librosa.load(input_path, sr=sr, mono=True)
    y, _   = librosa.load(output_path, sr=sr_, mono=True)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    total_size = window_size + warmup_size
    X_list, Y_list = [], []
    for i in range(0, n - total_size, window_size):
        X_list.append(x[i : i + total_size])
        Y_list.append(y[i : i + total_size])
    X = np.array(X_list).reshape(-1, total_size, 1)
    Y = np.array(Y_list).reshape(-1, total_size, 1)
    return X, Y, sr_

X_train, Y_train, sr = load_windows('audio/testSound_mono.wav', 'audio/filteredOutput/testSound/testSound_mono_1000hz.wav')
X_val,   Y_val,   _  = load_windows('audio/bench_mono.wav',     'audio/filteredOutput/bench/bench_mono_1000hz.wav', sr=sr)

print(f"Train windows: {len(X_train)}, Val windows: {len(X_val)}, Sample rate: {sr}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
X_val   = torch.tensor(X_val,   dtype=torch.float32).to(device)
Y_val   = torch.tensor(Y_val,   dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=64, shuffle=True)

model     = Model().to(device)
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=1e-6)

def pre_emphasis(x, coef=0.95):
    # Boost high frequencies so the loss doesn't ignore the filter's rolloff region
    return torch.cat([x[:, :1, :], x[:, 1:, :] - coef * x[:, :-1, :]], dim=1)

def esr_loss(pred, target):
    pred   = pre_emphasis(pred)
    target = pre_emphasis(target)
    return torch.mean((pred - target) ** 2) / (torch.mean(target ** 2) + 1e-8)

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
        # Only compute loss on the target window, not the warmup portion
        loss = esr_loss(pred[:, warmup_size:, :], yb[:, warmup_size:, :])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= len(X_train)

    model.eval()
    with torch.no_grad():
        pred_val = model(X_val)
        val_loss = esr_loss(pred_val[:, warmup_size:, :], Y_val[:, warmup_size:, :]).item()

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'best_model.pt')
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
model.load_state_dict(torch.load('best_model.pt'))

with open('rtneural_model_weights.json', 'w') as f:
    json.dump(model.state_dict(), f, cls=EncodeTensor, indent=4)

print("Model saved to rtneural_model_weights.json")
