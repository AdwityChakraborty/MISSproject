import pandas as pd
import numpy as np
import time
import tracemalloc
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (precision_recall_curve, auc,
                              recall_score, confusion_matrix, f1_score)

print("=" * 60)
print("PHASE 3 STEP 3 — Sequence Detector (1D-CNN)")
print("=" * 60)

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW  = 10
EPOCHS  = 30
BATCH   = 128   # reduced from 256
MAX_TR  = 20_000  # subsample to keep memory safe on CPU
MAX_TE  = 8_000
print(f"  Device : {DEVICE}\n")


# ================================================================
# HELPERS
# ================================================================
def get_metrics(y_true, y_scores, y_pred, model_name, dataset):
    precision, recall_pts, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall_pts, precision)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    rec = recall_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn + 1e-9)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    print(f"    PR-AUC:{pr_auc:.4f}  Recall:{rec:.4f}  "
          f"FPR:{fpr:.4f}  F1:{f1:.4f}")
    return {"model": "1D-CNN", "dataset": dataset,
            "view": "sequence", "pr_auc": pr_auc,
            "recall": rec, "fpr": fpr, "f1": f1}


def bench_torch(model, X_tensor):
    tracemalloc.start()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(X_tensor[:500].to(DEVICE))  # bench on 500 samples
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    lat = ((t1 - t0) / 500) * 1000
    mem = peak / (1024 ** 2)
    print(f"    Latency:{lat:.4f} ms/flow  Memory:{mem:.2f} MB")
    return lat, mem


def subsample(X, y, n, seed=42):
    """Stratified subsample to n samples."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    return X[idx], y[idx]


# ================================================================
# 1D-CNN MODEL
# Input shape: (batch, window, n_features) — NOT flattened
# ================================================================
class CNN1D(nn.Module):
    def __init__(self, n_features, window=10):
        super().__init__()
        # Conv over the time (window) dimension
        # input: (batch, n_features, window)  after transpose
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=32,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)   # → (batch, 64, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        # x: (batch, window, n_features)
        x = x.permute(0, 2, 1)      # → (batch, n_features, window)
        x = self.conv(x)             # → (batch, 64, 1)
        x = x.squeeze(-1)            # → (batch, 64)
        return self.fc(x)            # → (batch, 2)


def train_cnn(model, X_tr, y_tr, n_features, epochs=EPOCHS,
              batch=BATCH, lr=1e-3):
    model.to(DEVICE)
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    # Reshape from (n, window*features) → (n, window, features)
    X_reshaped = X_tr.reshape(-1, WINDOW, n_features)
    ds = TensorDataset(
        torch.tensor(X_reshaped, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch, shuffle=True)

    for ep in range(epochs):
        model.train()
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            total += loss.item()
        if (ep+1) % 10 == 0:
            print(f"    Epoch {ep+1}/{epochs}  "
                  f"loss:{total/len(loader):.4f}")
    return model


def predict_cnn(model, X_te, n_features, batch=BATCH):
    model.eval()
    X_reshaped = X_te.reshape(-1, WINDOW, n_features)
    all_probs  = []
    loader = DataLoader(
        TensorDataset(torch.tensor(X_reshaped, dtype=torch.float32)),
        batch_size=batch, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(DEVICE))
            probs  = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs)


# ================================================================
# EXPERIMENT RUNNER
# ================================================================
def run_cnn_experiment(X_tr_raw, y_tr_raw, X_te_raw,
                        y_te_raw, exp_name):
    # Infer n_features from flattened dim
    n_features = X_tr_raw.shape[1] // WINDOW

    # Subsample to stay within memory limits
    X_tr, y_tr = subsample(X_tr_raw, y_tr_raw, MAX_TR)
    X_te, y_te = subsample(X_te_raw, y_te_raw, MAX_TE)

    print(f"\n  Train (subsampled):{X_tr.shape}  "
          f"Test (subsampled):{X_te.shape}")
    print(f"  n_features={n_features}  window={WINDOW}")
    print(f"  Label dist train: "
          f"{dict(zip(*np.unique(y_tr, return_counts=True)))}")

    model = CNN1D(n_features=n_features, window=WINDOW)
    model = train_cnn(model, X_tr, y_tr, n_features)

    # Bench on reshaped test tensor
    X_te_r = torch.tensor(
        X_te.reshape(-1, WINDOW, n_features), dtype=torch.float32)
    lat, mem = bench_torch(model, X_te_r)

    scores = predict_cnn(model, X_te, n_features)
    preds  = (scores >= 0.5).astype(int)

    r = get_metrics(y_te, scores, preds, "1D-CNN", exp_name)
    r.update({"latency_ms": lat, "memory_mb": mem})
    return r


all_results = []


# ================================================================
# E1 — NSL-KDD sequences
# ================================================================
print("\n" + "=" * 60)
print("E1 — NSL-KDD (1D-CNN sequence)")
print("=" * 60)

r = run_cnn_experiment(
    np.load("seq_nsl_X_train.npy"),
    np.load("seq_nsl_y_train.npy"),
    np.load("seq_nsl_X_test.npy"),
    np.load("seq_nsl_y_test.npy"),
    "E1:KDD→NSL(seq)")
all_results.append(r)


# ================================================================
# E3 — UNSW Temporal Drift sequences
# ================================================================
print("\n" + "=" * 60)
print("E3 — UNSW Temporal Drift (1D-CNN sequence)")
print("=" * 60)

r = run_cnn_experiment(
    np.load("seq_unsw_X_train.npy"),
    np.load("seq_unsw_y_train.npy"),
    np.load("seq_unsw_X_test.npy"),
    np.load("seq_unsw_y_test.npy"),
    "E3:T0→T2(seq)")
all_results.append(r)


# ================================================================
# E6 — ToN-IoT sequences
# ================================================================
print("\n" + "=" * 60)
print("E6 — ToN-IoT (1D-CNN sequence)")
print("=" * 60)

r = run_cnn_experiment(
    np.load("seq_ton_X_train.npy"),
    np.load("seq_ton_y_train.npy"),
    np.load("seq_ton_X_test.npy"),
    np.load("seq_ton_y_test.npy"),
    "E6:ToN-IoT(seq)")
all_results.append(r)


# ================================================================
# SAVE AND COMBINE
# ================================================================
print("\n" + "=" * 60)
print("SEQUENCE BASELINE RESULTS")
print("=" * 60)

df_seq = pd.DataFrame(all_results)
cols   = ["dataset","model","view","pr_auc","recall",
          "fpr","f1","latency_ms","memory_mb"]
df_seq = df_seq[cols]
print(df_seq.to_string(index=False))
df_seq.to_csv("baseline_sequence.csv", index=False)

df_all = pd.read_csv("baseline_all.csv")
df_all = pd.concat([df_all, df_seq], ignore_index=True)
df_all.to_csv("baseline_all.csv", index=False)

print("\n  baseline_sequence.csv  ✅  saved")
print("  baseline_all.csv       ✅  updated")
print("  Paste full output for verification")
