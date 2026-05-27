import pandas as pd
import numpy as np
import time
import tracemalloc
import warnings

warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_curve, auc, recall_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb

print("=" * 60)
print("PHASE 3 STEP 1 — Tabular Detectors & Baselines (FIXED)")
print("=" * 60)


# ================================================================
# HELPERS (fixed)
# ================================================================
def get_metrics(y_true, y_scores, y_pred, model_name, dataset):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if len(np.unique(y_true)) < 2:
        pr_auc = 0.0
    else:
        precision, recall_pts, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall_pts, precision)

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    rec = recall_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn + 1e-9)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"    PR-AUC:{pr_auc:.4f}  Recall:{rec:.4f}  FPR:{fpr:.4f}  F1:{f1:.4f}")
    return {
        "model": model_name,
        "dataset": dataset,
        "view": "tabular",
        "pr_auc": pr_auc,
        "recall": rec,
        "fpr": fpr,
        "f1": f1
    }


def bench(model, X):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = model.predict(X)  # use this prediction, not call again
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    lat = ((t1 - t0) / len(X)) * 1000
    mem = peak / (1024 ** 2)
    print(f"    Latency:{lat:.4f} ms/flow  Memory:{mem:.2f} MB")
    return lat, mem, out


results = []


# ================================================================
# E1 — KDD'99 → NSL-KDD  (domain shift)
# ================================================================
print("\n" + "=" * 60)
print("E1 — KDD'99 → NSL-KDD")
print("=" * 60)

X_tr = pd.read_parquet("H_kdd_train.parquet")
y_tr = pd.read_parquet("H_kdd_train_labels.parquet").iloc[:, 0]
X_te = pd.read_parquet("H_nsl_test.parquet")
y_te = pd.read_parquet("H_nsl_test_labels.parquet").iloc[:, 0]
y_train_bin = (y_tr != "normal").astype(int)
y_test_bin = (y_te != "normal").astype(int)
print(f"  Train:{X_tr.shape}  Test:{X_te.shape}")

for name, clf in [
    ("IsolationForest", IsolationForest(n_estimators=100,
                          contamination=0.4, random_state=42, n_jobs=-1)),
    ("LightGBM", lgb.LGBMClassifier(n_estimators=200,
                   random_state=42, n_jobs=-1, verbose=-1)),
    ("XGBoost", xgb.XGBClassifier(n_estimators=200, random_state=42,
                  n_jobs=-1, verbosity=0, eval_metric='logloss')),
    ("MLP", MLPClassifier(hidden_layer_sizes=(64,32),
           max_iter=100, random_state=42, early_stopping=True)),
]:
    print(f"\n  [{name}]")
    if name == "IsolationForest":
        clf.fit(X_tr)
        lat, mem, out = bench(clf, X_te)
        scores = -clf.score_samples(X_te)
        preds = (out == -1).astype(int)  # use out, not clf.predict again
    else:
        clf.fit(X_tr, y_train_bin)
        lat, mem, preds = bench(clf, X_te)
        scores = clf.predict_proba(X_te)[:, 1]
    r = get_metrics(y_test_bin, scores, preds, name, "E1:KDD→NSL")
    r.update({"latency_ms": lat, "memory_mb": mem})
    results.append(r)


# ================================================================
# E2 — NSL-KDD → UNSW-NB15  (cross-dataset)
# ================================================================
print("\n" + "=" * 60)
print("E2 — NSL-KDD → UNSW-NB15")
print("=" * 60)

X_tr = pd.read_parquet("H_nsl_train.parquet")
y_tr = pd.read_parquet("H_nsl_train_labels.parquet").iloc[:, 0]
X_te = pd.read_parquet("H_unsw_T2.parquet")
y_te = pd.read_parquet("H_unsw_T2_labels.parquet").iloc[:, 0]
y_train_bin = (y_tr != "normal").astype(int)

idx = np.random.RandomState(42).choice(len(X_te), 30_000, replace=False)
X_te_s = X_te.iloc[idx]
y_te_s = y_te.iloc[idx]
print(f"  Train:{X_tr.shape}  Test (sampled):{X_te_s.shape}")

for name, clf in [
    ("IsolationForest", IsolationForest(n_estimators=100,
                          contamination=0.13, random_state=42, n_jobs=-1)),
    ("LightGBM", lgb.LGBMClassifier(n_estimators=200,
                   random_state=42, n_jobs=-1, verbose=-1)),
    ("XGBoost", xgb.XGBClassifier(n_estimators=200, random_state=42,
                  n_jobs=-1, verbosity=0, eval_metric='logloss')),
    ("MLP", MLPClassifier(hidden_layer_sizes=(64,32),
           max_iter=100, random_state=42, early_stopping=True)),
]:
    print(f"\n  [{name}]")
    if name == "IsolationForest":
        clf.fit(X_tr)
        lat, mem, out = bench(clf, X_te_s)
        scores = -clf.score_samples(X_te_s)
        preds = (out == -1).astype(int)
    else:
        clf.fit(X_tr, y_train_bin)
        lat, mem, preds = bench(clf, X_te_s)
        scores = clf.predict_proba(X_te_s)[:, 1]
    r = get_metrics(y_te_s, scores, preds, name, "E2:NSL→UNSW")
    r.update({"latency_ms": lat, "memory_mb": mem})
    results.append(r)


# ================================================================
# E3 — UNSW-NB15 Temporal Drift  T0 → T1 / T2
# ================================================================
print("\n" + "=" * 60)
print("E3 — UNSW-NB15 Temporal Drift T0→T1/T2")
print("=" * 60)

X_T0 = pd.read_parquet("H_unsw_T0.parquet")
y_T0 = pd.read_parquet("H_unsw_T0_labels.parquet").iloc[:, 0]
X_T1 = pd.read_parquet("H_unsw_T1.parquet")
y_T1 = pd.read_parquet("H_unsw_T1_labels.parquet").iloc[:, 0]
X_T2 = pd.read_parquet("H_unsw_T2.parquet")
y_T2 = pd.read_parquet("H_unsw_T2_labels.parquet").iloc[:, 0]

idx_tr = np.random.RandomState(42).choice(len(X_T0), 80_000, replace=False)
idx_t1 = np.random.RandomState(42).choice(len(X_T1), 20_000, replace=False)
idx_t2 = np.random.RandomState(42).choice(len(X_T2), 20_000, replace=False)
X_T0_s = X_T0.iloc[idx_tr]; y_T0_s = y_T0.iloc[idx_tr]

for name, clf in [
    ("LightGBM", lgb.LGBMClassifier(n_estimators=200,
                  random_state=42, n_jobs=-1, verbose=-1)),
    ("XGBoost", xgb.XGBClassifier(n_estimators=200, random_state=42,
                  n_jobs=-1, verbosity=0, eval_metric='logloss')),
]:
    print(f"\n  [{name}]")
    clf.fit(X_T0_s, y_T0_s)
    for split, X_sp, y_sp, idx_sp in [
        ("T1", X_T1, y_T1, idx_t1),
        ("T2", X_T2, y_T2, idx_t2),
    ]:
        X_s = X_sp.iloc[idx_sp]
        y_s = y_sp.iloc[idx_sp]
        lat, mem, preds = bench(clf, X_s)
        scores = clf.predict_proba(X_s)[:, 1]
        r = get_metrics(y_s, scores, preds, name, f"E3:T0→{split}")
        r.update({"latency_ms": lat, "memory_mb": mem})
        results.append(r)


# ================================================================
# E4 — UNSW-NB15 IoT Profile Subset
# ================================================================
print("\n" + "=" * 60)
print("E4 — UNSW-NB15 IoT Profile Subset")
print("=" * 60)

df_iot = pd.read_parquet("unsw_iot_profile.parquet")
feat = [c for c in df_iot.columns if c not in ["attack_cat", "binary_label"]]
X_iot = df_iot[feat]
y_iot = df_iot["binary_label"]

X_iot_tr, X_iot_te, y_iot_tr, y_iot_te = train_test_split(
    X_iot, y_iot, test_size=0.3, random_state=42, stratify=y_iot)
print(f"  Train:{X_iot_tr.shape}  Test:{X_iot_te.shape}")
print(f"  Labels: {y_iot_tr.value_counts().to_dict()}")

for name, clf in [
    ("IsolationForest", IsolationForest(n_estimators=100,
                          contamination=0.1, random_state=42, n_jobs=-1)),
    ("LightGBM", lgb.LGBMClassifier(n_estimators=200,
                   random_state=42, n_jobs=-1, verbose=-1)),
    ("XGBoost", xgb.XGBClassifier(n_estimators=200, random_state=42,
                  n_jobs=-1, verbosity=0, eval_metric='logloss')),
]:
    print(f"\n  [{name}]")
    if name == "IsolationForest":
        clf.fit(X_iot_tr)
        lat, mem, out = bench(clf, X_iot_te)
        scores = -clf.score_samples(X_iot_te)
        preds = (out == -1).astype(int)
    else:
        clf.fit(X_iot_tr, y_iot_tr)
        lat, mem, preds = bench(clf, X_iot_te)
        scores = clf.predict_proba(X_iot_te)[:, 1]
    r = get_metrics(y_iot_te, scores, preds, name, "E4:IoT-subset")
    r.update({"latency_ms": lat, "memory_mb": mem})
    results.append(r)


# ================================================================
# E6 — ToN-IoT Full Evaluation
# ================================================================
print("\n" + "=" * 60)
print("E6 — ToN-IoT")
print("=" * 60)

X_tr = pd.read_parquet("H_ton_train.parquet")
y_tr = pd.read_parquet("H_ton_train_labels.parquet").iloc[:, 0]
X_te = pd.read_parquet("H_ton_test.parquet")
y_te = pd.read_parquet("H_ton_test_labels.parquet").iloc[:, 0]
print(f"  Train:{X_tr.shape}  Test:{X_te.shape}")

for name, clf in [
    ("IsolationForest", IsolationForest(n_estimators=100,
                          contamination=0.5, random_state=42, n_jobs=-1)),
    ("LightGBM", lgb.LGBMClassifier(n_estimators=200,
                   random_state=42, n_jobs=-1, verbose=-1)),
    ("XGBoost", xgb.XGBClassifier(n_estimators=200, random_state=42,
                  n_jobs=-1, verbosity=0, eval_metric='logloss')),
    ("MLP", MLPClassifier(hidden_layer_sizes=(64,32),
           max_iter=100, random_state=42, early_stopping=True)),
]:
    print(f"\n  [{name}]")
    if name == "IsolationForest":
        clf.fit(X_tr)
        lat, mem, out = bench(clf, X_te)
        scores = -clf.score_samples(X_te)
        preds = (out == -1).astype(int)
    else:
        clf.fit(X_tr, y_tr)  # assume y_tr already 0/1; else use: (y_tr != "normal").astype(int)
        lat, mem, preds = bench(clf, X_te)
        scores = clf.predict_proba(X_te)[:, 1]
    r = get_metrics(y_te, scores, preds, name, "E6:ToN-IoT")
    r.update({"latency_ms": lat, "memory_mb": mem})
    results.append(r)


# ================================================================
# SAVE RESULTS to NEW CSV (do not overwrite baseline_tabular.csv)
# ================================================================
print("\n" + "=" * 60)
print("TABULAR BASELINE RESULTS (FIXED)")
print("=" * 60)
df_res = pd.DataFrame(results)
cols = ["dataset", "model", "view", "pr_auc", "recall",
        "fpr", "f1", "latency_ms", "memory_mb"]
df_res = df_res[cols]
print(df_res.to_string(index=False))
df_res.to_csv("baseline_tabular_fixed.csv", index=False)
print("\n  baseline_tabular_fixed.csv  ✅  saved")
print("  Paste full output for verification")
