import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

print("Generating Pareto Front Plots...")

# ================================================================
# LOAD ALL THREE RESULT SETS
# ================================================================
df_baseline_raw = pd.read_csv("baseline_all.csv")
df_nsga  = pd.read_csv("pareto_front.csv")
df_pso   = pd.read_csv("pso_refinement.csv")
df_hybrid= pd.read_csv("hybrid_pareto_front.csv")

# Best baseline per experiment for reference point
baseline = {"pr_auc": 0.9182, "recall": 0.8913, "fpr": 0.1350}

# ================================================================
# FIGURE 1 — Pareto Front: PR-AUC vs FPR
# (main contribution figure for your paper)
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Pareto Front Analysis — NIO Optimization Results",
             fontsize=14, fontweight='bold', y=1.02)

# ── Plot 1: PR-AUC vs FPR ────────────────────────────────────────
ax = axes[0]

# NSGA-II points
ax.scatter(df_nsga["fpr"], df_nsga["pr_auc"],
           c='steelblue', s=80, alpha=0.8,
           label="NSGA-II Pareto", zorder=3)

# Hybrid points
ax.scatter(df_hybrid["fpr"], df_hybrid["pr_auc"],
           c='darkorange', s=80, marker='D', alpha=0.8,
           label="Hybrid NSGA-II+PSO", zorder=3)

# PSO refined points
ax.scatter(df_pso["pso_fpr"], df_pso["pso_pr_auc"],
           c='green', s=120, marker='*', alpha=0.9,
           label="PSO Refined", zorder=4)

# Baseline reference
ax.scatter(baseline["fpr"], baseline["pr_auc"],
           c='red', s=150, marker='X', zorder=5,
           label="Best Baseline (MLP)")

# Annotate best hybrid
best_h = df_hybrid.iloc[0]
ax.annotate(f"Best Hybrid\nPR-AUC={best_h['pr_auc']:.3f}\nFPR={best_h['fpr']:.3f}",
            xy=(best_h["fpr"], best_h["pr_auc"]),
            xytext=(best_h["fpr"]+0.02, best_h["pr_auc"]-0.04),
            fontsize=8, color='darkorange',
            arrowprops=dict(arrowstyle='->', color='darkorange'))

ax.set_xlabel("False Positive Rate (FPR) ↓", fontsize=11)
ax.set_ylabel("PR-AUC ↑", fontsize=11)
ax.set_title("PR-AUC vs FPR\n(lower-right = better)", fontsize=11)
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.01, max(df_nsga["fpr"].max(),
                        baseline["fpr"]) + 0.05)


# ── Plot 2: Recall vs FPR ─────────────────────────────────────────
ax = axes[1]

ax.scatter(df_nsga["fpr"], df_nsga["recall"],
           c='steelblue', s=80, alpha=0.8,
           label="NSGA-II Pareto", zorder=3)
ax.scatter(df_hybrid["fpr"], df_hybrid["recall"],
           c='darkorange', s=80, marker='D', alpha=0.8,
           label="Hybrid NSGA-II+PSO", zorder=3)
ax.scatter(df_pso["pso_fpr"], df_pso["pso_recall"],
           c='green', s=120, marker='*', alpha=0.9,
           label="PSO Refined", zorder=4)
ax.scatter(baseline["fpr"], baseline["recall"],
           c='red', s=150, marker='X', zorder=5,
           label="Best Baseline")

# Draw ideal region box
ax.axhspan(0.85, 1.0, alpha=0.05, color='green',
           label="Target recall zone")
ax.axvspan(0.0, 0.09, alpha=0.05, color='green')

ax.set_xlabel("False Positive Rate (FPR) ↓", fontsize=11)
ax.set_ylabel("Recall ↑", fontsize=11)
ax.set_title("Recall vs FPR\n(upper-left = better)", fontsize=11)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)


# ── Plot 3: PR-AUC vs Latency (efficiency frontier) ───────────────
ax = axes[2]

# Color by model type for NSGA-II
colors_nsga = ['steelblue' if m == 'XGB' else
               'cornflowerblue' for m in df_nsga["model_type"]]
ax.scatter(df_nsga["latency_ms"], df_nsga["pr_auc"],
           c=colors_nsga, s=80, alpha=0.8, zorder=3)

colors_hybrid = ['darkorange' if m == 'XGB' else
                 'moccasin' for m in df_hybrid["model_type"]]
ax.scatter(df_hybrid["latency_ms"], df_hybrid["pr_auc"],
           c=colors_hybrid, s=80, marker='D', alpha=0.8, zorder=3)

ax.scatter(0.009, baseline["pr_auc"],
           c='red', s=150, marker='X', zorder=5)

# IoT latency target line
ax.axvline(x=20, color='red', linestyle='--',
           alpha=0.7, label="IoT target (20ms)")

# Legend patches
p1 = mpatches.Patch(color='steelblue',    label='NSGA-II XGB')
p2 = mpatches.Patch(color='cornflowerblue', label='NSGA-II LGB')
p3 = mpatches.Patch(color='darkorange',   label='Hybrid XGB')
p4 = mpatches.Patch(color='moccasin',     label='Hybrid LGB')
p5 = mpatches.Patch(color='red',          label='Baseline')
ax.legend(handles=[p1,p2,p3,p4,p5], fontsize=8)

ax.set_xlabel("Latency (ms/flow) ↓", fontsize=11)
ax.set_ylabel("PR-AUC ↑", fontsize=11)
ax.set_title("PR-AUC vs Latency\n(upper-left = better)", fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pareto_front_plots.png", dpi=150,
            bbox_inches='tight')
plt.show()
print("  pareto_front_plots.png  ✅  saved")


# ================================================================
# FIGURE 2 — Three-way comparison bar chart
# ================================================================
fig2, ax2 = plt.subplots(figsize=(10, 6))

best_nsga_row   = df_nsga.iloc[0]
best_hybrid_row = df_hybrid.iloc[0]
best_pso_row    = df_pso.loc[
    (df_pso["pso_pr_auc"] + df_pso["pso_recall"]
     - df_pso["pso_fpr"]).idxmax()]

methods  = ["Baseline\n(MLP)", "NSGA-II\nBest",
            "PSO\nRefined", "Hybrid\nBest"]
pr_aucs  = [baseline["pr_auc"],
            best_nsga_row["pr_auc"],
            best_pso_row["pso_pr_auc"],
            best_hybrid_row["pr_auc"]]
recalls  = [baseline["recall"],
            best_nsga_row["recall"],
            best_pso_row["pso_recall"],
            best_hybrid_row["recall"]]
fprs     = [baseline["fpr"],
            best_nsga_row["fpr"],
            best_pso_row["pso_fpr"],
            best_hybrid_row["fpr"]]

x     = np.arange(len(methods))
width = 0.25

bars1 = ax2.bar(x - width, pr_aucs,  width,
                label='PR-AUC',  color='steelblue',  alpha=0.85)
bars2 = ax2.bar(x,          recalls, width,
                label='Recall',  color='darkorange',  alpha=0.85)
bars3 = ax2.bar(x + width,  fprs,    width,
                label='FPR',     color='tomato',      alpha=0.85)

# Value labels on bars
for bar in [*bars1, *bars2, *bars3]:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 0.005,
             f'{h:.3f}', ha='center', va='bottom',
             fontsize=7, rotation=45)

ax2.set_ylabel("Score", fontsize=12)
ax2.set_title("Three-Way Comparison: Baseline vs NSGA-II vs PSO vs Hybrid",
              fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(methods, fontsize=11)
ax2.legend(fontsize=10)
ax2.set_ylim(0, 1.15)
ax2.grid(True, axis='y', alpha=0.3)
ax2.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig("three_way_comparison.png", dpi=150,
            bbox_inches='tight')
plt.show()
print("  three_way_comparison.png  ✅  saved")


# ================================================================
# FIGURE 3 — NSGA-II Convergence curve
# ================================================================
fig3, ax3 = plt.subplots(figsize=(9, 5))

# Reconstruct from generation logs in output
# Using the values printed during NSGA-II run
gen_prauc = [0.9648,0.9657,0.9666,0.9698,0.9718,
             0.9718,0.9718,0.9708,0.9669,0.9733,
             0.9733,0.9738,0.9721,0.9693,0.9734,
             0.9738,0.9719,0.9731,0.9738,0.9739]
gen_fpr   = [0.0227,0.0223,0.0195,0.0195,0.0204,
             0.0181,0.0107,0.0107,0.0181,0.0107,
             0.0107,0.0190,0.0107,0.0107,0.0107,
             0.0107,0.0107,0.0107,0.0107,0.0107]
gens = list(range(1, 21))

ax3_twin = ax3.twinx()
l1, = ax3.plot(gens, gen_prauc, 'b-o', markersize=5,
               label='PR-AUC (left)', linewidth=2)
l2, = ax3_twin.plot(gens, gen_fpr, 'r-s', markersize=5,
                    label='FPR (right)', linewidth=2)
ax3.axhline(y=baseline["pr_auc"], color='blue',
            linestyle='--', alpha=0.5,
            label=f'Baseline PR-AUC ({baseline["pr_auc"]})')
ax3_twin.axhline(y=baseline["fpr"], color='red',
                 linestyle='--', alpha=0.5,
                 label=f'Baseline FPR ({baseline["fpr"]})')

ax3.set_xlabel("Generation", fontsize=12)
ax3.set_ylabel("PR-AUC ↑", fontsize=12, color='blue')
ax3_twin.set_ylabel("FPR ↓", fontsize=12, color='red')
ax3.set_title("NSGA-II Convergence Curve",
              fontsize=12, fontweight='bold')
ax3.legend(handles=[l1, l2], loc='center right', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(gens)

plt.tight_layout()
plt.savefig("convergence_curve.png", dpi=150, bbox_inches='tight')
plt.show()
print("  convergence_curve.png  ✅  saved")

print("\n  ALL FIGURES SAVED:")
print("  pareto_front_plots.png     — 3-panel Pareto front")
print("  three_way_comparison.png   — bar chart comparison")
print("  convergence_curve.png      — NSGA-II convergence")
print("\n  These are your main paper figures for Phase 4")
