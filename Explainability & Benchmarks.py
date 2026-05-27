import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import warnings
warnings.filterwarnings("ignore")

import shap
import lime
import lime.lime_tabular
from sklearn.metrics import f1_score
import xgboost as xgb

print("=" * 60)
print("PHASE 6 — Explainability & Benchmarks")
print("=" * 60)


# ================================================================
# SETUP — Load data and retrain best NIO model
# ================================================================
print("\nLoading data and retraining best NIO model...")

N_FEATURES  = 32
group_size  = N_FEATURES // 5
FEAT_GROUPS = [
    list(range(0, group_size)),
    list(range(group_size, 2*group_size)),
    list(range(2*group_size, 3*group_size)),
    list(range(3*group_size, 4*group_size)),
    list(range(4*group_size, N_FEATURES)),
]
BEST_FEATURES = FEAT_GROUPS[0] + FEAT_GROUPS[1]  # 12 features
BEST_THRESHOLD = 0.515

# COMMON_FEATURES list (same order as harmonized schema)
COMMON_FEATURES = [
    'f_duration','f_src_bytes','f_dst_bytes','f_src_pkts',
    'f_dst_pkts','f_total_bytes','f_bytes_per_pkt',
    'f_src_load','f_dst_load','f_serror_rate','f_rerror_rate',
    'f_same_srv_rate','f_diff_srv_rate','f_proto_tcp',
    'f_proto_udp','f_proto_icmp','f_proto_other',
    'f_svc_http','f_svc_ftp','f_svc_smtp','f_svc_dns',
    'f_svc_ssh','f_svc_other','f_svc_none',
    'f_state_established','f_state_rejected','f_state_other',
    'f_iot_iat_variance','f_iot_burstiness',
    'f_iot_service_sparsity','f_iot_fan_out','f_iot_fan_in'
]
SELECTED_FEAT_NAMES = [COMMON_FEATURES[i] for i in BEST_FEATURES]

# Load data
X_tr = pd.read_parquet("H_nsl_train.parquet").values
y_tr = pd.read_parquet(
    "H_nsl_train_labels.parquet").iloc[:,0]
y_tr_bin = (y_tr != "normal").astype(int).values
X_te = pd.read_parquet("H_nsl_test.parquet").values
y_te = pd.read_parquet(
    "H_nsl_test_labels.parquet").iloc[:,0]
y_te_bin = (y_te != "normal").astype(int).values

X_tr_sel = X_tr[:, BEST_FEATURES]
X_te_sel = X_te[:, BEST_FEATURES]

# Train best NIO model
nio_model = xgb.XGBClassifier(
    n_estimators=210, max_depth=6,
    random_state=42, n_jobs=-1,
    verbosity=0, eval_metric='logloss')
nio_model.fit(X_tr_sel, y_tr_bin)
print(f"  Model trained : XGBoost, "
      f"{len(BEST_FEATURES)} features")


# ================================================================
# PART 1 — SHAP Analysis
# ================================================================
print("\n" + "=" * 60)
print("PART 1 — SHAP Feature Importance")
print("=" * 60)

# Sample for SHAP (full dataset is slow)
np.random.seed(42)
idx_shap = np.random.choice(len(X_te_sel),
                              min(2000, len(X_te_sel)),
                              replace=False)
X_shap = X_te_sel[idx_shap]
y_shap = y_te_bin[idx_shap]

print(f"  Computing SHAP values on {len(X_shap)} samples...")

# Fix for SHAP 0.49 + XGBoost 2.x compatibility issue
# Use model_output='raw' to bypass base_score parsing bug
try:
    explainer = shap.TreeExplainer(
        nio_model,
        model_output='raw')
    shap_values = explainer.shap_values(X_shap)
except ValueError:
    # Fallback: retrain with older booster format
    import lightgbm as lgb
    print("  XGBoost SHAP failed — using LightGBM for SHAP")
    lgb_shap = lgb.LGBMClassifier(
        n_estimators=210, random_state=42,
        n_jobs=-1, verbose=-1)
    lgb_shap.fit(X_tr_sel, y_tr_bin)
    explainer   = shap.TreeExplainer(lgb_shap)
    shap_values = explainer.shap_values(X_shap)
    # LightGBM returns list — take class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    nio_model = lgb_shap   # use LGB for LIME too

# If shap_values is a list (binary), take class 1
if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values

print(f"  SHAP values shape : {sv.shape}")

# Mean absolute SHAP per feature
mean_shap = np.abs(sv).mean(axis=0)
shap_df   = pd.DataFrame({
    "feature"   : SELECTED_FEAT_NAMES,
    "mean_shap" : mean_shap
}).sort_values("mean_shap", ascending=False)

print("\n  Top 12 features by mean |SHAP|:")
print(shap_df.to_string(index=False))
shap_df.to_csv("shap_importance.csv", index=False)


# ── SHAP Plot 1: Bar chart ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#d62728' if 'iot' in f else '#1f77b4'
          for f in shap_df["feature"]]
bars = ax.barh(shap_df["feature"],
               shap_df["mean_shap"],
               color=colors, alpha=0.85)
ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
ax.set_title("SHAP Feature Importance — Best NIO Model\n"
             "(XGBoost, 12 features, NSL-KDD)",
             fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.grid(True, axis='x', alpha=0.3)

# Legend
p1 = mpatches.Patch(color='#d62728', label='IoT-specific feature')
p2 = mpatches.Patch(color='#1f77b4', label='Standard feature')
ax.legend(handles=[p1, p2], fontsize=10)

for bar, val in zip(bars, shap_df["mean_shap"]):
    ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig("shap_importance_bar.png", dpi=150,
            bbox_inches='tight')
plt.show()
print("  shap_importance_bar.png  ✅  saved")


# ── SHAP Plot 2: Beeswarm ─────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 7))
shap.summary_plot(sv, X_shap,
                  feature_names=SELECTED_FEAT_NAMES,
                  show=False, plot_size=None)
plt.title("SHAP Beeswarm Plot — Feature Impact Distribution",
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=150,
            bbox_inches='tight')
plt.show()
print("  shap_beeswarm.png  ✅  saved")


# ── SHAP by attack category ───────────────────────────────────────
print("\n  SHAP by attack category (top 3 features):")
top3_feats  = shap_df["feature"].iloc[:3].tolist()
top3_idx    = [SELECTED_FEAT_NAMES.index(f) for f in top3_feats]

attack_cats  = y_te[idx_shap]
unique_cats  = pd.Series(attack_cats).value_counts().head(5).index

for cat in unique_cats:
    mask     = (attack_cats == cat)
    if mask.sum() < 5:
        continue
    cat_shap = np.abs(sv[mask][:, top3_idx]).mean(axis=0)
    vals     = "  ".join([f"{top3_feats[i]}={cat_shap[i]:.4f}"
                           for i in range(len(top3_feats))])
    print(f"    {cat:20s} : {vals}")


# ================================================================
# PART 2 — LIME Local Explanations
# ================================================================
print("\n" + "=" * 60)
print("PART 2 — LIME Local Explanations")
print("=" * 60)

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_tr_sel,
    feature_names=SELECTED_FEAT_NAMES,
    class_names=["Normal", "Attack"],
    mode="classification",
    random_state=42
)

# Explain 3 samples — one normal, two attack types
normal_idx = np.where(y_te_bin == 0)[0][:1]
attack_idx = np.where(y_te_bin == 1)[0][:2]
explain_idx = np.concatenate([normal_idx, attack_idx])

lime_results = []
for i, idx in enumerate(explain_idx):
    sample   = X_te_sel[idx]
    true_lbl = "Normal" if y_te_bin[idx] == 0 else "Attack"
    pred_lbl = "Attack" if nio_model.predict(
        sample.reshape(1,-1))[0] == 1 else "Normal"

    exp = lime_explainer.explain_instance(
        sample,
        nio_model.predict_proba,
        num_features=6,
        num_samples=500
    )
    exp_list = exp.as_list()

    print(f"\n  Sample {i+1} — True:{true_lbl}  "
          f"Pred:{pred_lbl}")
    for feat_rule, weight in exp_list[:6]:
        direction = "→Attack" if weight > 0 else "→Normal"
        print(f"    {feat_rule:35s}  "
              f"weight={weight:+.4f}  {direction}")

    lime_results.append({
        "sample_id"  : idx,
        "true_label" : true_lbl,
        "pred_label" : pred_lbl,
        "top_feature": exp_list[0][0] if exp_list else "N/A",
        "top_weight" : exp_list[0][1] if exp_list else 0,
    })

    # Save individual explanation plot
    fig_lime = exp.as_pyplot_figure()
    plt.title(f"LIME Explanation — Sample {i+1} "
              f"(True:{true_lbl} Pred:{pred_lbl})",
              fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"lime_explanation_{i+1}.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  lime_explanation_{i+1}.png  ✅  saved")

pd.DataFrame(lime_results).to_csv("lime_results.csv",
                                    index=False)
print("\n  lime_results.csv  ✅  saved")


# ================================================================
# PART 3 — IoT Gateway Latency Benchmark
# ================================================================
print("\n" + "=" * 60)
print("PART 3 — IoT Gateway Latency Benchmark")
print("=" * 60)
print("  Simulating constrained IoT gateway:")
print("  (single-threaded, no parallelism)\n")

import tracemalloc

benchmark_results = []

# Test at different throughput levels
flow_counts = [100, 500, 1000, 2000, 5000]

for n_flows in flow_counts:
    X_bench = X_te_sel[:n_flows]

    # Single-threaded inference
    tracemalloc.start()
    t0    = time.perf_counter()
    preds = nio_model.predict(X_bench)
    t1    = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_ms     = (t1 - t0) * 1000
    per_flow_ms  = total_ms / n_flows
    peak_mb      = peak / (1024 ** 2)
    throughput   = n_flows / (t1 - t0)

    meets_lat    = per_flow_ms <= 20.0
    meets_mem    = peak_mb <= 200.0

    print(f"  {n_flows:5d} flows : "
          f"{per_flow_ms:.4f} ms/flow  "
          f"peak={peak_mb:.1f} MB  "
          f"throughput={throughput:.0f} flows/sec  "
          f"{'✅' if meets_lat else '❌'} latency  "
          f"{'✅' if meets_mem else '❌'} memory")

    benchmark_results.append({
        "n_flows"      : n_flows,
        "per_flow_ms"  : round(per_flow_ms, 5),
        "total_ms"     : round(total_ms, 2),
        "peak_mb"      : round(peak_mb, 2),
        "throughput"   : round(throughput, 1),
        "meets_latency": meets_lat,
        "meets_memory" : meets_mem,
    })

df_bench = pd.DataFrame(benchmark_results)
df_bench.to_csv("latency_benchmark.csv", index=False)
print("\n  latency_benchmark.csv  ✅  saved")

# Check overall IoT targets
all_lat = all(df_bench["meets_latency"])
all_mem = all(df_bench["meets_memory"])
print(f"\n  IoT Target — ≤20ms latency  : "
      f"{'✅ ALL PASS' if all_lat else '❌ SOME FAIL'}")
print(f"  IoT Target — ≤200MB memory  : "
      f"{'✅ ALL PASS' if all_mem else '❌ SOME FAIL'}")


# ── Latency plot ──────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(9, 5))
ax3.plot(df_bench["n_flows"],
         df_bench["per_flow_ms"],
         'b-o', linewidth=2, markersize=7,
         label="NIO Model (XGBoost 12-feat)")
ax3.axhline(y=20, color='red', linestyle='--',
            alpha=0.7, label="IoT target (20ms)")
ax3.fill_between(df_bench["n_flows"],
                  df_bench["per_flow_ms"], 20,
                  where=[v < 20 for v in df_bench["per_flow_ms"]],
                  alpha=0.1, color='green',
                  label="Safe zone")
ax3.set_xlabel("Number of Flows", fontsize=12)
ax3.set_ylabel("Latency (ms/flow)", fontsize=12)
ax3.set_title("IoT Gateway Latency Benchmark\n"
              "Per-flow inference latency vs throughput",
              fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xscale('log')
plt.tight_layout()
plt.savefig("latency_benchmark.png", dpi=150,
            bbox_inches='tight')
plt.show()
print("  latency_benchmark.png  ✅  saved")


# ================================================================
# PART 4 — Ablation Bar Chart
# ================================================================
print("\n" + "=" * 60)
print("PART 4 — Ablation Bar Chart")
print("=" * 60)

df_abl = pd.read_csv("ablation_results.csv")

fig4, ax4 = plt.subplots(figsize=(12, 6))
x      = np.arange(len(df_abl))
width  = 0.22
bars_p = ax4.bar(x - width*1.5, df_abl["pr_auc"],
                  width, label='PR-AUC',
                  color='steelblue', alpha=0.85)
bars_r = ax4.bar(x - width*0.5, df_abl["recall"],
                  width, label='Recall',
                  color='darkorange', alpha=0.85)
bars_f = ax4.bar(x + width*0.5, df_abl["fpr"],
                  width, label='FPR',
                  color='tomato', alpha=0.85)
bars_1 = ax4.bar(x + width*1.5, df_abl["f1"],
                  width, label='F1',
                  color='green', alpha=0.85)

# Value labels
for bars in [bars_p, bars_r, bars_f, bars_1]:
    for bar in bars:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2,
                 h + 0.002, f'{h:.3f}',
                 ha='center', va='bottom',
                 fontsize=6.5, rotation=45)

labels = [a.replace(" ", "\n") for a in df_abl["ablation"]]
ax4.set_xticks(x)
ax4.set_xticklabels(labels, fontsize=8)
ax4.set_ylabel("Score", fontsize=12)
ax4.set_title("E5 Ablation Study Results\n"
              "Impact of removing each framework component",
              fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.set_ylim(0, 1.15)
ax4.grid(True, axis='y', alpha=0.3)
ax4.axhline(y=0.95, color='gray',
            linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig("ablation_chart.png", dpi=150,
            bbox_inches='tight')
plt.show()
print("  ablation_chart.png  ✅  saved")


# ================================================================
# FINAL PHASE 6 SUMMARY
# ================================================================
print("\n" + "=" * 60)
print("PHASE 6 COMPLETE — FILE SUMMARY")
print("=" * 60)
import os
phase6_files = [
    "shap_importance.csv",
    "shap_importance_bar.png",
    "shap_beeswarm.png",
    "lime_results.csv",
    "lime_explanation_1.png",
    "lime_explanation_2.png",
    "lime_explanation_3.png",
    "latency_benchmark.csv",
    "latency_benchmark.png",
    "ablation_chart.png",
]
for f in phase6_files:
    exists = os.path.exists(f)
    print(f"  {'✅' if exists else '❌'}  {f}")

print("\n  Paste full output for verification")
