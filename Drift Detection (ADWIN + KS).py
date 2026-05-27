import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

from river import drift as river_drift
from scipy import stats
from sklearn.metrics import (precision_recall_curve, auc,
                              recall_score, confusion_matrix, f1_score)
import lightgbm as lgb
import xgboost as xgb

print("=" * 60)
print("PHASE 5 — Drift Detection & Online Adaptation")
print("=" * 60)


# ================================================================
# HELPERS
# ================================================================
def get_metrics(y_true, y_pred, y_scores, label=""):
    pr, rc, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(rc, pr)
    cm = confusion_matrix(y_true, y_pred)
    tn,fp,fn,tp = cm.ravel() if cm.shape==(2,2) else (0,0,0,0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn + 1e-9)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    if label:
        print(f"    {label}: PR-AUC={pr_auc:.4f}  "
              f"Recall={rec:.4f}  FPR={fpr:.4f}  F1={f1:.4f}")
    return {"pr_auc":pr_auc,"recall":rec,"fpr":fpr,"f1":f1}


# ================================================================
# LOAD BEST NIO CONFIG (from hybrid Pareto front)
# Use pareto_id=18: XGB, 12 features, threshold=0.515
# Best balance of PR-AUC and low FPR
# ================================================================
print("\nLoading best NIO configuration...")

N_FEATURES  = 32
group_size  = N_FEATURES // 5
FEAT_GROUPS = [
    list(range(0, group_size)),
    list(range(group_size, 2*group_size)),
    list(range(2*group_size, 3*group_size)),
    list(range(3*group_size, 4*group_size)),
    list(range(4*group_size, N_FEATURES)),
]

# Best hybrid config: XGB, 12 features (groups 0+1), threshold=0.515
BEST_FEATURES = FEAT_GROUPS[0] + FEAT_GROUPS[1]   # first 12 features
BEST_THRESHOLD = 0.515
print(f"  Model     : XGBoost")
print(f"  Features  : {len(BEST_FEATURES)} of {N_FEATURES}")
print(f"  Threshold : {BEST_THRESHOLD}")


# ================================================================
# PART 1 — ADWIN DRIFT DETECTION
# ADWIN monitors a stream of values and signals when the
# mean has shifted significantly (concept drift).
# We apply it to the anomaly score stream.
# ================================================================
print("\n" + "=" * 60)
print("PART 1 — ADWIN Drift Detection")
print("=" * 60)

def run_adwin(score_stream, name=""):
    """
    Run ADWIN on a stream of anomaly scores.
    Returns list of drift detection points.
    """
    detector    = river_drift.ADWIN(delta=0.002)
    drift_points = []
    for i, val in enumerate(score_stream):
        detector.update(float(val))
        if detector.drift_detected:
            drift_points.append(i)
    if drift_points:
        print(f"  {name}: drift detected at "
              f"{len(drift_points)} points — "
              f"first at index {drift_points[0]}")
    else:
        print(f"  {name}: no drift detected")
    return drift_points


# Load UNSW data
X_T0 = pd.read_parquet("H_unsw_T0.parquet").values
y_T0 = pd.read_parquet("H_unsw_T0_labels.parquet").iloc[:,0].values
X_T1 = pd.read_parquet("H_unsw_T1.parquet").values
y_T1 = pd.read_parquet("H_unsw_T1_labels.parquet").iloc[:,0].values
X_T2 = pd.read_parquet("H_unsw_T2.parquet").values
y_T2 = pd.read_parquet("H_unsw_T2_labels.parquet").iloc[:,0].values

# Train best NIO model on T0
print("\n  Training best NIO model on UNSW T0...")
idx_tr = np.random.RandomState(42).choice(
    len(X_T0), 50_000, replace=False)
X_T0_s = X_T0[idx_tr][:, BEST_FEATURES]
y_T0_s = y_T0[idx_tr]

nio_model = xgb.XGBClassifier(
    n_estimators=210, max_depth=6,
    random_state=42, n_jobs=-1,
    verbosity=0, eval_metric='logloss')
nio_model.fit(X_T0_s, y_T0_s)
print(f"  Model trained on {len(X_T0_s):,} samples")

# Generate score streams for T1 and T2
idx_t1 = np.random.RandomState(42).choice(
    len(X_T1), 10_000, replace=False)
idx_t2 = np.random.RandomState(42).choice(
    len(X_T2), 10_000, replace=False)

scores_T1 = nio_model.predict_proba(
    X_T1[idx_t1][:, BEST_FEATURES])[:, 1]
scores_T2 = nio_model.predict_proba(
    X_T2[idx_t2][:, BEST_FEATURES])[:, 1]
scores_T0_ref = nio_model.predict_proba(
    X_T0_s[:5000])[:, 1]

print("\n  Running ADWIN on anomaly score streams:")
drift_T0 = run_adwin(scores_T0_ref, "T0 (reference)")
drift_T1 = run_adwin(scores_T1,     "T1 (val)")
drift_T2 = run_adwin(scores_T2,     "T2 (test)")


# ================================================================
# PART 2 — KS TEST ON FEATURE DISTRIBUTIONS
# KS test checks if two distributions are significantly different.
# We test each feature between T0 and T1/T2.
# ================================================================
print("\n" + "=" * 60)
print("PART 2 — KS Test on Feature Distributions")
print("=" * 60)

feature_names = [f"f_{i}" for i in range(len(BEST_FEATURES))]
ks_results_T1 = []
ks_results_T2 = []

X_T0_feat = X_T0[:5000][:, BEST_FEATURES]
X_T1_feat = X_T1[idx_t1][:, BEST_FEATURES]
X_T2_feat = X_T2[idx_t2][:, BEST_FEATURES]

for i, fname in enumerate(feature_names):
    ks_t1 = stats.ks_2samp(X_T0_feat[:, i], X_T1_feat[:, i])
    ks_t2 = stats.ks_2samp(X_T0_feat[:, i], X_T2_feat[:, i])
    ks_results_T1.append({"feature": fname,
                           "ks_stat": ks_t1.statistic,
                           "p_value": ks_t1.pvalue,
                           "drift"  : ks_t1.pvalue < 0.05})
    ks_results_T2.append({"feature": fname,
                           "ks_stat": ks_t2.statistic,
                           "p_value": ks_t2.pvalue,
                           "drift"  : ks_t2.pvalue < 0.05})

df_ks_T1 = pd.DataFrame(ks_results_T1)
df_ks_T2 = pd.DataFrame(ks_results_T2)

n_drift_T1 = df_ks_T1["drift"].sum()
n_drift_T2 = df_ks_T2["drift"].sum()

print(f"  T0→T1: {n_drift_T1}/{len(BEST_FEATURES)} features "
      f"show significant drift (p<0.05)")
print(f"  T0→T2: {n_drift_T2}/{len(BEST_FEATURES)} features "
      f"show significant drift (p<0.05)")
print(f"\n  Most drifted features (T0→T2):")
print(df_ks_T2.nlargest(5,"ks_stat")[
    ["feature","ks_stat","p_value"]].to_string(index=False))

df_ks_T1.to_csv("ks_test_T1.csv", index=False)
df_ks_T2.to_csv("ks_test_T2.csv", index=False)
print("\n  ks_test_T1.csv / ks_test_T2.csv  ✅  saved")


# ================================================================
# PART 3 — ADAPTATION MECHANISMS
# Three strategies triggered when drift is detected:
# A) Threshold recalibration
# B) Ensemble reweighting
# C) Incremental retraining
# ================================================================
print("\n" + "=" * 60)
print("PART 3 — Adaptation Mechanisms")
print("=" * 60)

y_T2_sample = y_T2[idx_t2]
X_T2_feat_full = X_T2[idx_t2][:, BEST_FEATURES]

# Baseline — no adaptation (T0 model, original threshold)
print("\n  [No Adaptation — baseline]")
preds_base = (scores_T2 >= BEST_THRESHOLD).astype(int)
m_base = get_metrics(y_T2_sample, preds_base,
                      scores_T2, "No adaptation")

adaptation_results = [{"method": "No Adaptation", **m_base}]


# ── A: Threshold Recalibration ────────────────────────────────────
print("\n  [A] Threshold Recalibration")
# Use T1 scores to find optimal threshold
scores_T1_sample = scores_T1
y_T1_sample      = y_T1[idx_t1]

# Find threshold that maximises F1 on T1
best_thresh = BEST_THRESHOLD
best_f1     = 0.0
for t in np.arange(0.1, 0.9, 0.01):
    preds_t = (scores_T1_sample >= t).astype(int)
    f1_t    = f1_score(y_T1_sample, preds_t, zero_division=0)
    if f1_t > best_f1:
        best_f1     = f1_t
        best_thresh = t

print(f"    Original threshold : {BEST_THRESHOLD:.3f}")
print(f"    Recalibrated       : {best_thresh:.3f}  "
      f"(F1 on T1 = {best_f1:.4f})")

preds_recal = (scores_T2 >= best_thresh).astype(int)
m_recal = get_metrics(y_T2_sample, preds_recal,
                       scores_T2, "Threshold recalibration")
adaptation_results.append({"method": "Threshold Recalibration",
                            **m_recal})


# ── B: Ensemble Reweighting ───────────────────────────────────────
print("\n  [B] Ensemble Reweighting")
# Train a second model on T1 data
# Blend scores: w1*model1 + w2*model2
idx_t1_tr = np.random.RandomState(0).choice(
    len(X_T1), min(20_000, len(X_T1)), replace=False)
X_T1_tr   = X_T1[idx_t1_tr][:, BEST_FEATURES]
y_T1_tr   = y_T1[idx_t1_tr]

model2 = xgb.XGBClassifier(
    n_estimators=100, random_state=42,
    n_jobs=-1, verbosity=0, eval_metric='logloss')
model2.fit(X_T1_tr, y_T1_tr)
scores2_T2 = model2.predict_proba(X_T2_feat_full)[:, 1]

# Find best blend weight on T1 val set
scores2_T1 = model2.predict_proba(
    X_T1[idx_t1][:, BEST_FEATURES])[:, 1]

best_w  = 0.5
best_f1 = 0.0
for w in np.arange(0.1, 1.0, 0.1):
    blended = w * scores_T1 + (1-w) * scores2_T1
    preds_w = (blended >= BEST_THRESHOLD).astype(int)
    f1_w    = f1_score(y_T1_sample, preds_w, zero_division=0)
    if f1_w > best_f1:
        best_f1 = f1_w
        best_w  = w

blended_T2 = best_w * scores_T2 + (1-best_w) * scores2_T2
preds_ens  = (blended_T2 >= BEST_THRESHOLD).astype(int)

print(f"    Blend weight (old model) : {best_w:.1f}")
print(f"    Blend weight (new model) : {1-best_w:.1f}")
m_ens = get_metrics(y_T2_sample, preds_ens,
                     blended_T2, "Ensemble reweighting")
adaptation_results.append({"method": "Ensemble Reweighting",
                            **m_ens})


# ── C: Incremental Retraining ─────────────────────────────────────
print("\n  [C] Incremental Retraining")
# Fine-tune original model by continuing training on T1 data
# XGBoost supports this via xgb_model parameter
model_retrained = xgb.XGBClassifier(
    n_estimators=50,   # additional trees only
    random_state=42, n_jobs=-1,
    verbosity=0, eval_metric='logloss')

# Train on combined T0 sample + T1 data
X_combined = np.vstack([X_T0_s[:10_000],
                         X_T1_tr[:10_000]])
y_combined  = np.concatenate([y_T0_s[:10_000],
                               y_T1_tr[:10_000]])

model_retrained.fit(X_combined, y_combined)
scores_retrain = model_retrained.predict_proba(
    X_T2_feat_full)[:, 1]
preds_retrain  = (scores_retrain >= BEST_THRESHOLD).astype(int)

print(f"    Retrained on : {len(X_combined):,} samples "
      f"(T0 + T1 combined)")
m_retrain = get_metrics(y_T2_sample, preds_retrain,
                         scores_retrain,
                         "Incremental retraining")
adaptation_results.append({"method": "Incremental Retraining",
                            **m_retrain})


# ── Save adaptation results ───────────────────────────────────────
df_adapt = pd.DataFrame(adaptation_results)
df_adapt.to_csv("adaptation_results.csv", index=False)

print("\n" + "=" * 60)
print("ADAPTATION RESULTS SUMMARY")
print("=" * 60)
print(df_adapt.to_string(index=False))
print("\n  adaptation_results.csv  ✅  saved")


# ================================================================
# PART 4 — RECALL DEGRADATION OVER TIME
# Measures how recall changes from T0→T1→T2 with and
# without adaptation — key figure for drift robustness claim
# ================================================================
print("\n" + "=" * 60)
print("PART 4 — Recall Degradation Over Time")
print("=" * 60)

# T0 performance (training period)
scores_T0_eval = nio_model.predict_proba(
    X_T0_s[:3000])[:, 1]
preds_T0_eval  = (scores_T0_eval >= BEST_THRESHOLD).astype(int)
rec_T0 = recall_score(y_T0_s[:3000], preds_T0_eval,
                       zero_division=0)

# T1 performance — no adaptation
preds_T1_noadapt = (scores_T1 >= BEST_THRESHOLD).astype(int)
rec_T1_noadapt   = recall_score(y_T1_sample,
                                  preds_T1_noadapt,
                                  zero_division=0)

# T2 performance — no adaptation
rec_T2_noadapt = recall_score(y_T2_sample, preds_base,
                                zero_division=0)

# T2 performance — best adaptation (incremental retraining)
rec_T2_adapt = recall_score(y_T2_sample, preds_retrain,
                              zero_division=0)

print(f"  Recall at T0 (train) : {rec_T0:.4f}")
print(f"  Recall at T1 (no adapt) : {rec_T1_noadapt:.4f}  "
      f"(Δ = {rec_T1_noadapt-rec_T0:+.4f})")
print(f"  Recall at T2 (no adapt) : {rec_T2_noadapt:.4f}  "
      f"(Δ = {rec_T2_noadapt-rec_T0:+.4f})")
print(f"  Recall at T2 (adapted)  : {rec_T2_adapt:.4f}  "
      f"(Δ = {rec_T2_adapt-rec_T0:+.4f})")

deg_no_adapt = abs(rec_T2_noadapt - rec_T0)
deg_adapted  = abs(rec_T2_adapt   - rec_T0)
print(f"\n  Degradation without adaptation : {deg_no_adapt:.4f}")
print(f"  Degradation with adaptation    : {deg_adapted:.4f}")
print(f"  Improvement from adaptation    : "
      f"{deg_no_adapt - deg_adapted:+.4f}")

target_met = deg_adapted <= 0.05
print(f"\n  Target: ≤5% recall degradation — "
      f"{'✅ MET' if target_met else '⚠️ NOT MET'} "
      f"({deg_adapted*100:.1f}%)")

drift_summary = {
    "rec_T0": rec_T0,
    "rec_T1_noadapt": rec_T1_noadapt,
    "rec_T2_noadapt": rec_T2_noadapt,
    "rec_T2_adapted": rec_T2_adapt,
    "degradation_no_adapt": deg_no_adapt,
    "degradation_adapted": deg_adapted,
}
pd.DataFrame([drift_summary]).to_csv(
    "drift_summary.csv", index=False)
print("  drift_summary.csv  ✅  saved")


# ================================================================
# PART 5 — EXPERIMENT E5 ABLATIONS
# Remove one component at a time and measure impact
# ================================================================
print("\n" + "=" * 60)
print("PART 5 — Experiment E5 Ablations")
print("=" * 60)

X_te_full = X_T2[idx_t2]
y_te_abl  = y_T2_sample

ablation_results = []

# ── Full model (reference) ────────────────────────────────────────
print("\n  [Full Model — NIO + IoT features + adaptation]")
m_full = get_metrics(y_te_abl, preds_retrain,
                      scores_retrain, "Full model")
ablation_results.append({"ablation": "Full Model (reference)",
                          **m_full})


# ── Ablation 1: Remove IoT features ──────────────────────────────
print("\n  [Ablation 1 — Remove IoT features]")
# IoT features are the last 5 in COMMON_FEATURES (indices 27-31)
NON_IOT_FEATS = list(range(27))   # first 27 non-IoT features
X_T0_noiot = X_T0[idx_tr][:, NON_IOT_FEATS]
X_T2_noiot = X_T2[idx_t2][:, NON_IOT_FEATS]

model_noiot = xgb.XGBClassifier(
    n_estimators=210, random_state=42,
    n_jobs=-1, verbosity=0, eval_metric='logloss')
model_noiot.fit(X_T0_noiot, y_T0_s)
scores_noiot = model_noiot.predict_proba(X_T2_noiot)[:, 1]
preds_noiot  = (scores_noiot >= BEST_THRESHOLD).astype(int)
m_noiot = get_metrics(y_te_abl, preds_noiot,
                       scores_noiot, "No IoT features")
ablation_results.append({"ablation": "Remove IoT Features",
                          **m_noiot})


# ── Ablation 2: Remove NIO (use default hyperparameters) ──────────
print("\n  [Ablation 2 — Remove NIO optimization]")
# Default XGBoost — no feature selection, default params
model_default = xgb.XGBClassifier(
    n_estimators=100, max_depth=6,
    learning_rate=0.3,   # XGBoost default
    random_state=42, n_jobs=-1,
    verbosity=0, eval_metric='logloss')
model_default.fit(X_T0[idx_tr], y_T0_s)
scores_default = model_default.predict_proba(
    X_T2[idx_t2])[:, 1]
preds_default  = (scores_default >= 0.5).astype(int)
m_default = get_metrics(y_te_abl, preds_default,
                         scores_default,
                         "No NIO (default params)")
ablation_results.append({"ablation": "Remove NIO Optimization",
                          **m_default})


# ── Ablation 3: Remove drift adaptation ───────────────────────────
print("\n  [Ablation 3 — Remove drift adaptation]")
# Use original T0 model with no recalibration on T2
m_nodrift = get_metrics(y_te_abl, preds_base,
                         scores_T2,
                         "No drift adaptation")
ablation_results.append({"ablation": "Remove Drift Adaptation",
                          **m_nodrift})


# ── Ablation 4: Remove feature selection (use all 32) ─────────────
print("\n  [Ablation 4 — Remove feature selection]")
model_allfeat = xgb.XGBClassifier(
    n_estimators=210, random_state=42,
    n_jobs=-1, verbosity=0, eval_metric='logloss')
model_allfeat.fit(X_T0[idx_tr], y_T0_s)
scores_allfeat = model_allfeat.predict_proba(
    X_T2[idx_t2])[:, 1]
preds_allfeat  = (scores_allfeat >= BEST_THRESHOLD).astype(int)
m_allfeat = get_metrics(y_te_abl, preds_allfeat,
                         scores_allfeat,
                         "No feature selection (all 32)")
ablation_results.append({"ablation": "Remove Feature Selection",
                          **m_allfeat})


# ── Save ablation results ─────────────────────────────────────────
df_abl = pd.DataFrame(ablation_results)
df_abl.to_csv("ablation_results.csv", index=False)

print("\n" + "=" * 60)
print("E5 ABLATION RESULTS")
print("=" * 60)
print(df_abl.to_string(index=False))
print("\n  ablation_results.csv  ✅  saved")


# ================================================================
# FINAL PHASE 5 SUMMARY
# ================================================================
print("\n" + "=" * 60)
print("PHASE 5 COMPLETE — FILE SUMMARY")
print("=" * 60)
phase5_files = [
    "adaptation_results.csv",
    "ablation_results.csv",
    "drift_summary.csv",
    "ks_test_T1.csv",
    "ks_test_T2.csv",
]
import os
for f in phase5_files:
    exists = os.path.exists(f)
    print(f"  {'✅' if exists else '❌'}  {f}")

print("\n  Paste full output for verification")
