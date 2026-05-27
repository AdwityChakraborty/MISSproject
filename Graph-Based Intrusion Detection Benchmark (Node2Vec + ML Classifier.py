import pandas as pd
import numpy as np
import time
import tracemalloc
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (precision_recall_curve, auc,
                              recall_score, confusion_matrix, f1_score)
import lightgbm as lgb

print("=" * 60)
print("PHASE 3 STEP 4 — Graph Detector (Node2Vec + Classifier)")
print("=" * 60)


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
    return {"model": model_name, "dataset": dataset,
            "view": "graph", "pr_auc": pr_auc,
            "recall": rec, "fpr": fpr, "f1": f1}


def bench_sklearn(model, X):
    tracemalloc.start()
    t0  = time.perf_counter()
    out = model.predict(X)
    t1  = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    lat = ((t1 - t0) / len(X)) * 1000
    mem = peak / (1024 ** 2)
    print(f"    Latency:{lat:.4f} ms/flow  Memory:{mem:.2f} MB")
    return lat, mem, out


# ================================================================
# EXPERIMENT RUNNER
# Graph embeddings are 64-dim per flow (32 src + 32 service)
# We train a LightGBM classifier on top of these embeddings
# ================================================================
def run_graph_experiment(X_tr, y_tr, X_te, y_te, exp_name):
    print(f"\n  Train:{X_tr.shape}  Test:{X_te.shape}")
    print(f"  Label dist train: "
          f"{dict(zip(*np.unique(y_tr, return_counts=True)))}")

    # ── LightGBM on graph embeddings ─────────────────────────────
    print(f"\n  [LightGBM+Graph | {exp_name}]")
    clf_lgb = lgb.LGBMClassifier(n_estimators=200, random_state=42,
                                   n_jobs=-1, verbose=-1)
    clf_lgb.fit(X_tr, y_tr)
    lat, mem, preds = bench_sklearn(clf_lgb, X_te)
    scores = clf_lgb.predict_proba(X_te)[:, 1]
    r1 = get_metrics(y_te, scores, preds,
                     "LightGBM+Graph", exp_name)
    r1.update({"latency_ms": lat, "memory_mb": mem})

    # ── MLP on graph embeddings ───────────────────────────────────
    print(f"\n  [MLP+Graph | {exp_name}]")
    clf_mlp = MLPClassifier(hidden_layer_sizes=(64, 32),
                             max_iter=100, random_state=42,
                             early_stopping=True)
    clf_mlp.fit(X_tr, y_tr)
    lat, mem, preds = bench_sklearn(clf_mlp, X_te)
    scores = clf_mlp.predict_proba(X_te)[:, 1]
    r2 = get_metrics(y_te, scores, preds,
                     "MLP+Graph", exp_name)
    r2.update({"latency_ms": lat, "memory_mb": mem})

    return [r1, r2]


all_results = []


# ================================================================
# E1 — NSL-KDD graph embeddings
# ================================================================
print("\n" + "=" * 60)
print("E1 — NSL-KDD (graph view)")
print("=" * 60)

X_tr = np.load("graph_nsl_X_train.npy")
y_tr = np.load("graph_nsl_y_train.npy")
X_te = np.load("graph_nsl_X_test.npy")
y_te = np.load("graph_nsl_y_test.npy")

res = run_graph_experiment(X_tr, y_tr, X_te, y_te,
                            "E1:KDD→NSL(graph)")
all_results.extend(res)


# ================================================================
# E3 — UNSW graph embeddings
# ================================================================
print("\n" + "=" * 60)
print("E3 — UNSW (graph view)")
print("=" * 60)

X_tr = np.load("graph_unsw_X_train.npy")
y_tr = np.load("graph_unsw_y_train.npy")
X_te = np.load("graph_unsw_X_test.npy")
y_te = np.load("graph_unsw_y_test.npy")

res = run_graph_experiment(X_tr, y_tr, X_te, y_te,
                            "E3:T0→T2(graph)")
all_results.extend(res)


# ================================================================
# E6 — ToN-IoT graph embeddings
# ================================================================
print("\n" + "=" * 60)
print("E6 — ToN-IoT (graph view)")
print("=" * 60)

X_tr = np.load("graph_ton_X_train.npy")
y_tr = np.load("graph_ton_y_train.npy")
X_te = np.load("graph_ton_X_test.npy")
y_te = np.load("graph_ton_y_test.npy")

res = run_graph_experiment(X_tr, y_tr, X_te, y_te,
                            "E6:ToN-IoT(graph)")
all_results.extend(res)


# ================================================================
# SAVE AND COMBINE ALL BASELINES
# ================================================================
print("\n" + "=" * 60)
print("GRAPH BASELINE RESULTS")
print("=" * 60)

df_graph = pd.DataFrame(all_results)
cols = ["dataset","model","view","pr_auc","recall",
        "fpr","f1","latency_ms","memory_mb"]
df_graph = df_graph[cols]
print(df_graph.to_string(index=False))
df_graph.to_csv("baseline_graph.csv", index=False)

df_all = pd.read_csv("baseline_all.csv")
df_all = pd.concat([df_all, df_graph], ignore_index=True)
df_all.to_csv("baseline_all.csv", index=False)

print("\n  baseline_graph.csv   ✅  saved")
print("  baseline_all.csv     ✅  updated (all views combined)")


# ================================================================
# FINAL COMPLETE BASELINE TABLE
# ================================================================
print("\n" + "=" * 60)
print("COMPLETE BASELINE TABLE — ALL MODELS ALL VIEWS")
print("=" * 60)
df_final = pd.read_csv("baseline_all.csv")
print(df_final.to_string(index=False))
print(f"\n  Total baseline entries : {len(df_final)}")
print("\n  PHASE 3 COMPLETE — paste output for verification")
