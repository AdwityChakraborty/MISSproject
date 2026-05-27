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
from sklearn.model_selection import train_test_split

print("=" * 60)
print("PHASE 3 STEP 2 — Deep Learning Detectors")
print("(Autoencoder, VAE, Deep SVDD)")
print("=" * 60)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device : {DEVICE}\n")


# ================================================================
# HELPERS
# ================================================================
def get_metrics(y_true, y_scores, threshold, model_name, dataset):
    preds = (y_scores >= threshold).astype(int)
    precision, recall_pts, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall_pts, precision)
    cm = confusion_matrix(y_true, preds)
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    rec = recall_score(y_true, preds, zero_division=0)
    fpr = fp / (fp + tn + 1e-9)
    f1  = f1_score(y_true, preds, zero_division=0)
    print(f"    PR-AUC:{pr_auc:.4f}  Recall:{rec:.4f}  "
          f"FPR:{fpr:.4f}  F1:{f1:.4f}")
    return {"model": model_name, "dataset": dataset,
            "view": "deep", "pr_auc": pr_auc,
            "recall": rec, "fpr": fpr, "f1": f1}


def bench_torch(model, X_tensor):
    tracemalloc.start()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(X_tensor.to(DEVICE))
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    lat = ((t1 - t0) / len(X_tensor)) * 1000
    mem = peak / (1024 ** 2)
    print(f"    Latency:{lat:.4f} ms/flow  Memory:{mem:.2f} MB")
    return lat, mem


def to_tensor(X):
    return torch.tensor(np.array(X), dtype=torch.float32)


def get_threshold(scores, contamination=0.15):
    """Set threshold at (1-contamination) percentile of training scores."""
    return np.percentile(scores, (1 - contamination) * 100)


# ================================================================
# MODEL DEFINITIONS
# ================================================================

# ── Autoencoder ──────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, n_feat, bottleneck=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feat, 32), nn.ReLU(),
            nn.Linear(32, 16),    nn.ReLU(),
            nn.Linear(16, bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 16), nn.ReLU(),
            nn.Linear(16, 32),         nn.ReLU(),
            nn.Linear(32, n_feat)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── VAE ──────────────────────────────────────────────────────────
class VAE(nn.Module):
    def __init__(self, n_feat, latent=8):
        super().__init__()
        self.fc1  = nn.Linear(n_feat, 32)
        self.fc_mu  = nn.Linear(32, latent)
        self.fc_var = nn.Linear(32, latent)
        self.dec  = nn.Sequential(
            nn.Linear(latent, 32), nn.ReLU(),
            nn.Linear(32, n_feat)
        )
    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_var(h)
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.dec(z), mu, log_var


def vae_loss(recon, x, mu, log_var):
    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kld


# ── Deep SVDD ────────────────────────────────────────────────────
class DeepSVDD(nn.Module):
    def __init__(self, n_feat, rep_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 32), nn.ReLU(),
            nn.Linear(32, rep_dim)
        )
    def forward(self, x):
        return self.net(x)


# ================================================================
# TRAINING FUNCTIONS
# ================================================================
def train_ae(model, X_tr, epochs=30, batch=256, lr=1e-3):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_tr.to(DEVICE)),
                        batch_size=batch, shuffle=True)
    for ep in range(epochs):
        model.train()
        total = 0
        for (x,) in loader:
            opt.zero_grad()
            loss = criterion(model(x), x)
            loss.backward(); opt.step()
            total += loss.item()
        if (ep+1) % 10 == 0:
            print(f"    Epoch {ep+1}/{epochs}  loss:{total/len(loader):.4f}")
    return model


def train_vae(model, X_tr, epochs=30, batch=256, lr=1e-3):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X_tr.to(DEVICE)),
                        batch_size=batch, shuffle=True)
    for ep in range(epochs):
        model.train()
        total = 0
        for (x,) in loader:
            opt.zero_grad()
            recon, mu, lv = model(x)
            loss = vae_loss(recon, x, mu, lv)
            loss.backward(); opt.step()
            total += loss.item()
        if (ep+1) % 10 == 0:
            print(f"    Epoch {ep+1}/{epochs}  loss:{total/len(loader):.4f}")
    return model


def train_svdd(model, X_tr, epochs=30, batch=256, lr=1e-3):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Compute center from initial forward pass
    with torch.no_grad():
        center = model(X_tr.to(DEVICE)).mean(dim=0)
    loader = DataLoader(TensorDataset(X_tr.to(DEVICE)),
                        batch_size=batch, shuffle=True)
    for ep in range(epochs):
        model.train()
        total = 0
        for (x,) in loader:
            opt.zero_grad()
            out  = model(x)
            loss = torch.mean(torch.sum((out - center) ** 2, dim=1))
            loss.backward(); opt.step()
            total += loss.item()
        if (ep+1) % 10 == 0:
            print(f"    Epoch {ep+1}/{epochs}  loss:{total/len(loader):.4f}")
    return model, center


def ae_scores(model, X_te):
    model.eval()
    with torch.no_grad():
        recon = model(X_te.to(DEVICE))
        return torch.mean((recon - X_te.to(DEVICE))**2,
                          dim=1).cpu().numpy()


def vae_scores(model, X_te):
    model.eval()
    with torch.no_grad():
        recon, _, _ = model(X_te.to(DEVICE))
        return torch.mean((recon - X_te.to(DEVICE))**2,
                          dim=1).cpu().numpy()


def svdd_scores(model, center, X_te):
    model.eval()
    with torch.no_grad():
        out = model(X_te.to(DEVICE))
        return torch.sum((out - center)**2,
                         dim=1).cpu().numpy()


# ================================================================
# EXPERIMENT RUNNER
# ================================================================
def run_deep_experiment(X_tr_np, X_te_np, y_te, exp_name,
                         contamination=0.15, epochs=30):
    """Train AE, VAE, Deep SVDD on X_tr; evaluate on X_te."""
    results = []
    n_feat  = X_tr_np.shape[1]
    X_tr_t  = to_tensor(X_tr_np)
    X_te_t  = to_tensor(X_te_np)

    print(f"\n  Train:{X_tr_np.shape}  Test:{X_te_np.shape}")

    # ── Autoencoder ──────────────────────────────────────────────
    print(f"\n  [Autoencoder | {exp_name}]")
    ae = Autoencoder(n_feat)
    ae = train_ae(ae, X_tr_t, epochs=epochs)
    lat, mem = bench_torch(ae, X_te_t)
    scores   = ae_scores(ae, X_te_t)
    # Threshold from training reconstruction errors
    tr_scores = ae_scores(ae, X_tr_t)
    thresh    = get_threshold(tr_scores, contamination)
    r = get_metrics(y_te, scores, thresh, "Autoencoder", exp_name)
    r.update({"latency_ms": lat, "memory_mb": mem})
    results.append(r)

    # ── VAE ──────────────────────────────────────────────────────
    print(f"\n  [VAE | {exp_name}]")
    vae = VAE(n_feat)
    vae = train_vae(vae, X_tr_t, epochs=epochs)
    lat, mem = bench_torch(vae, X_te_t)
    scores   = vae_scores(vae, X_te_t)
    tr_scores = vae_scores(vae, X_tr_t)
    thresh    = get_threshold(tr_scores, contamination)
    r = get_metrics(y_te, scores, thresh, "VAE", exp_name)
    r.update({"latency_ms": lat, "memory_mb": mem})
    results.append(r)

    # ── Deep SVDD ────────────────────────────────────────────────
    print(f"\n  [DeepSVDD | {exp_name}]")
    svdd_model = DeepSVDD(n_feat)
    svdd_model, center = train_svdd(svdd_model, X_tr_t, epochs=epochs)
    lat, mem = bench_torch(svdd_model, X_te_t)
    scores   = svdd_scores(svdd_model, center, X_te_t)
    tr_scores = svdd_scores(svdd_model, center, X_tr_t)
    thresh    = get_threshold(tr_scores, contamination)
    r = get_metrics(y_te, scores, thresh, "DeepSVDD", exp_name)
    r.update({"latency_ms": lat, "memory_mb": mem})
    results.append(r)

    return results


all_results = []


# ================================================================
# E1 — KDD'99 → NSL-KDD
# ================================================================
print("\n" + "=" * 60)
print("E1 — KDD'99 → NSL-KDD (deep detectors)")
print("=" * 60)

X_tr = pd.read_parquet("H_kdd_train.parquet").values
y_tr_labels = pd.read_parquet(
    "H_kdd_train_labels.parquet").iloc[:, 0]
X_te = pd.read_parquet("H_nsl_test.parquet").values
y_te = pd.read_parquet(
    "H_nsl_test_labels.parquet").iloc[:, 0]
y_te_bin = (y_te != "normal").astype(int).values

# Train only on normal samples for unsupervised detectors
normal_mask = (y_tr_labels == "normal").values
X_tr_normal = X_tr[normal_mask]

res = run_deep_experiment(X_tr_normal, X_te, y_te_bin,
                           "E1:KDD→NSL", contamination=0.15)
all_results.extend(res)


# ================================================================
# E2 — NSL-KDD → UNSW-NB15
# ================================================================
print("\n" + "=" * 60)
print("E2 — NSL-KDD → UNSW-NB15 (deep detectors)")
print("=" * 60)

X_tr  = pd.read_parquet("H_nsl_train.parquet").values
y_tr_labels = pd.read_parquet(
    "H_nsl_train_labels.parquet").iloc[:, 0]
X_te  = pd.read_parquet("H_unsw_T2.parquet").values
y_te  = pd.read_parquet(
    "H_unsw_T2_labels.parquet").iloc[:, 0].values

# Sample test for speed
idx   = np.random.RandomState(42).choice(len(X_te), 20_000,
                                          replace=False)
X_te_s = X_te[idx]; y_te_s = y_te[idx]

normal_mask = (y_tr_labels == "normal").values
X_tr_normal = X_tr[normal_mask]

res = run_deep_experiment(X_tr_normal, X_te_s, y_te_s,
                           "E2:NSL→UNSW", contamination=0.13)
all_results.extend(res)


# ================================================================
# E3 — UNSW Temporal Drift T0 → T2
# ================================================================
print("\n" + "=" * 60)
print("E3 — UNSW Temporal Drift T0→T2 (deep detectors)")
print("=" * 60)

X_T0  = pd.read_parquet("H_unsw_T0.parquet").values
y_T0  = pd.read_parquet("H_unsw_T0_labels.parquet").iloc[:, 0].values
X_T2  = pd.read_parquet("H_unsw_T2.parquet").values
y_T2  = pd.read_parquet("H_unsw_T2_labels.parquet").iloc[:, 0].values

# Sample
idx_tr = np.random.RandomState(42).choice(len(X_T0), 50_000,
                                           replace=False)
idx_te = np.random.RandomState(42).choice(len(X_T2), 15_000,
                                           replace=False)
X_T0_s = X_T0[idx_tr]; y_T0_s = y_T0[idx_tr]
X_T2_s = X_T2[idx_te]; y_T2_s = y_T2[idx_te]

# Train on normal only
normal_mask = (y_T0_s == 0)
X_T0_normal = X_T0_s[normal_mask]

res = run_deep_experiment(X_T0_normal, X_T2_s, y_T2_s,
                           "E3:T0→T2", contamination=0.13)
all_results.extend(res)


# ================================================================
# E6 — ToN-IoT
# ================================================================
print("\n" + "=" * 60)
print("E6 — ToN-IoT (deep detectors)")
print("=" * 60)

X_tr  = pd.read_parquet("H_ton_train.parquet").values
y_tr  = pd.read_parquet("H_ton_train_labels.parquet").iloc[:, 0].values
X_te  = pd.read_parquet("H_ton_test.parquet").values
y_te  = pd.read_parquet("H_ton_test_labels.parquet").iloc[:, 0].values

normal_mask = (y_tr == 0)
X_tr_normal = X_tr[normal_mask]

res = run_deep_experiment(X_tr_normal, X_te, y_te,
                           "E6:ToN-IoT", contamination=0.5)
all_results.extend(res)


# ================================================================
# SAVE AND COMBINE WITH TABULAR RESULTS
# ================================================================
print("\n" + "=" * 60)
print("DEEP LEARNING BASELINE RESULTS")
print("=" * 60)

df_deep = pd.DataFrame(all_results)
cols = ["dataset","model","view","pr_auc","recall",
        "fpr","f1","latency_ms","memory_mb"]
df_deep = df_deep[cols]
print(df_deep.to_string(index=False))
df_deep.to_csv("baseline_deep.csv", index=False)

# Merge with tabular results
df_tab = pd.read_csv("baseline_tabular.csv")
df_all = pd.concat([df_tab, df_deep], ignore_index=True)
df_all.to_csv("baseline_all.csv", index=False)

print("\n  baseline_deep.csv  ✅  saved")
print("  baseline_all.csv   ✅  saved (tabular + deep combined)")
print("  Paste full output for verification")
