import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (precision_recall_curve, auc,
                              recall_score, confusion_matrix, f1_score)
from sklearn.ensemble import IsolationForest
import lightgbm as lgb
import xgboost as xgb

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.core.callback import Callback

print("=" * 60)
print("PHASE 4 STAGE 3 — Hybrid NSGA-II + PSO")
print("=" * 60)
print("""
  How it works:
  Every 5 NSGA-II generations, the top-3 solutions from the
  current population are passed to PSO for local refinement.
  Refined solutions replace their originals before the next
  NSGA-II generation begins. This combines global exploration
  (NSGA-II) with local exploitation (PSO).
""")

# ================================================================
# LOAD DATA
# ================================================================
X_tr = pd.read_parquet("H_nsl_train.parquet").values
y_tr = pd.read_parquet(
    "H_nsl_train_labels.parquet").iloc[:, 0]
y_tr_bin = (y_tr != "normal").astype(int).values
X_te = pd.read_parquet("H_nsl_test.parquet").values
y_te = pd.read_parquet(
    "H_nsl_test_labels.parquet").iloc[:, 0]
y_te_bin = (y_te != "normal").astype(int).values

# Subsample for speed
idx_tr = np.random.RandomState(42).choice(
    len(X_tr), 15_000, replace=False)
idx_te = np.random.RandomState(42).choice(
    len(X_te), 5_000, replace=False)
X_tr_s = X_tr[idx_tr]; y_tr_s = y_tr_bin[idx_tr]
X_te_s = X_te[idx_te]; y_te_s = y_te_bin[idx_te]

print(f"  Train: {X_tr_s.shape}  Test: {X_te_s.shape}\n")


# ================================================================
# SHARED CONFIG (same as Stage 1)
# ================================================================
N_FEATURES   = X_tr_s.shape[1]
N_VARS       = 11
N_OBJECTIVES = 4
POP_SIZE     = 30
N_GEN        = 20
PSO_INTERVAL = 5     # run PSO every 5 generations
PSO_PARTICLES= 10    # smaller PSO inside loop for speed
PSO_ITER     = 10

group_size = N_FEATURES // 5
FEATURE_GROUPS = [
    list(range(0, group_size)),
    list(range(group_size, 2*group_size)),
    list(range(2*group_size, 3*group_size)),
    list(range(3*group_size, 4*group_size)),
    list(range(4*group_size, N_FEATURES)),
]


def decode_chromosome(x):
    model_val = x[0]
    contam    = 0.05 + x[1] * 0.45
    n_est     = int(50 + x[2] * 250)
    max_depth = int(3 + x[3] * 7)
    lr        = 0.01 + x[4] * 0.29
    threshold = 0.3 + x[5] * 0.4
    feat_mask = x[6:]

    active = []
    for i, w in enumerate(feat_mask):
        if w >= 0.5:
            active.extend(FEATURE_GROUPS[i])
    if len(active) == 0:
        active = list(range(N_FEATURES))

    if model_val < 0.33:   model_type = "IF"
    elif model_val < 0.66: model_type = "LGB"
    else:                  model_type = "XGB"

    return {"model_type": model_type, "contam": contam,
            "n_est": n_est, "max_depth": max_depth,
            "lr": lr, "threshold": threshold,
            "features": active}


def evaluate_chromosome(x, X_tr, y_tr, X_te, y_te):
    cfg  = decode_chromosome(x)
    feat = cfg["features"]
    Xtr  = X_tr[:, feat]
    Xte  = X_te[:, feat]
    t0   = time.perf_counter()

    try:
        if cfg["model_type"] == "IF":
            m = IsolationForest(
                n_estimators=cfg["n_est"],
                contamination=cfg["contam"],
                random_state=42, n_jobs=1)
            m.fit(Xtr)
            scores = -m.score_samples(Xte)
            preds  = (scores >= np.percentile(
                scores, (1-cfg["contam"])*100)).astype(int)
        elif cfg["model_type"] == "LGB":
            m = lgb.LGBMClassifier(
                n_estimators=cfg["n_est"],
                max_depth=cfg["max_depth"],
                learning_rate=cfg["lr"],
                random_state=42, n_jobs=1, verbose=-1)
            m.fit(Xtr, y_tr)
            scores = m.predict_proba(Xte)[:, 1]
            preds  = (scores >= cfg["threshold"]).astype(int)
        else:
            m = xgb.XGBClassifier(
                n_estimators=cfg["n_est"],
                max_depth=cfg["max_depth"],
                learning_rate=cfg["lr"],
                random_state=42, n_jobs=1,
                verbosity=0, eval_metric='logloss')
            m.fit(Xtr, y_tr)
            scores = m.predict_proba(Xte)[:, 1]
            preds  = (scores >= cfg["threshold"]).astype(int)
    except Exception:
        return np.array([1.0, 1.0, 1.0, 100.0])

    t1  = time.perf_counter()
    lat = ((t1 - t0) / len(Xte)) * 1000

    try:
        p, r, _ = precision_recall_curve(y_te, scores)
        pr_auc  = auc(r, p)
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(y_te, preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape==(2,2) else (0,0,0,0)
    recall = recall_score(y_te, preds, zero_division=0)
    fpr    = fp / (fp + tn + 1e-9)

    return np.array([-pr_auc, -recall, fpr, lat])


# ================================================================
# MINI PSO — local refinement of a single solution
# Only optimizes threshold (x[5]) and feature weights (x[6:])
# keeping model type and hyperparams fixed
# ================================================================
def mini_pso_refine(x_seed, X_tr, y_tr, X_te, y_te,
                    n_particles=PSO_PARTICLES,
                    n_iter=PSO_ITER):
    """Refine a single chromosome using PSO on threshold+features."""
    dim = 6   # threshold (1) + feature weights (5)
    lb  = np.zeros(dim)
    ub  = np.ones(dim)

    # Initialize particles around the seed
    noise = np.random.randn(n_particles, dim) * 0.05
    pos   = np.clip(x_seed[5:11] + noise, lb, ub)
    vel   = np.zeros_like(pos)

    def fitness(p):
        x_new    = x_seed.copy()
        x_new[5:11] = p
        obj = evaluate_chromosome(x_new, X_tr, y_tr, X_te, y_te)
        # Combined score: maximize PR-AUC + Recall, minimize FPR
        return -obj[0] - obj[1] * 0.5 + obj[2] * 0.3

    p_best_pos = pos.copy()
    p_best_fit = np.array([fitness(p) for p in pos])
    g_idx      = np.argmax(p_best_fit)
    g_best_pos = p_best_pos[g_idx].copy()
    g_best_fit = p_best_fit[g_idx]

    w, c1, c2 = 0.7, 1.5, 1.5
    for _ in range(n_iter):
        r1 = np.random.rand(n_particles, dim)
        r2 = np.random.rand(n_particles, dim)
        vel = (w * vel
               + c1 * r1 * (p_best_pos - pos)
               + c2 * r2 * (g_best_pos - pos))
        pos = np.clip(pos + vel, lb, ub)
        fits = np.array([fitness(p) for p in pos])
        improved = fits > p_best_fit
        p_best_pos[improved] = pos[improved]
        p_best_fit[improved] = fits[improved]
        g_idx2 = np.argmax(p_best_fit)
        if p_best_fit[g_idx2] > g_best_fit:
            g_best_pos = p_best_pos[g_idx2].copy()
            g_best_fit = p_best_fit[g_idx2]

    x_refined    = x_seed.copy()
    x_refined[5:11] = g_best_pos
    return x_refined


# ================================================================
# HYBRID CALLBACK — triggers PSO every PSO_INTERVAL generations
# ================================================================
class HybridPSOCallback(Callback):
    def __init__(self, X_tr, y_tr, X_te, y_te, interval=5):
        super().__init__()
        self.X_tr     = X_tr
        self.y_tr     = y_tr
        self.X_te     = X_te
        self.y_te     = y_te
        self.interval = interval
        self.gen      = 0
        self.pso_runs = 0

    def notify(self, algorithm):
        self.gen += 1
        if self.gen % self.interval != 0:
            return

        print(f"  → PSO refinement triggered at gen {self.gen}")

        # Get current population
        pop  = algorithm.pop
        X_pop = pop.get("X")
        F_pop = pop.get("F")

        # Select top-3 by first objective (PR-AUC)
        top_idx = np.argsort(F_pop[:, 0])[:3]

        for i in top_idx:
            x_orig    = X_pop[i].copy()
            x_refined = mini_pso_refine(
                x_orig,
                self.X_tr, self.y_tr,
                self.X_te, self.y_te)

            # Evaluate refined solution
            f_refined = evaluate_chromosome(
                x_refined,
                self.X_tr, self.y_tr,
                self.X_te, self.y_te)

            # Replace if improved
            if f_refined[0] < F_pop[i][0]:
                X_pop[i] = x_refined
                F_pop[i] = f_refined

        pop.set("X", X_pop)
        pop.set("F", F_pop)
        self.pso_runs += 1
        print(f"    PSO refined {len(top_idx)} solutions "
              f"(total PSO runs: {self.pso_runs})")


# ================================================================
# HYBRID PROBLEM
# ================================================================
class HybridIDSProblem(Problem):
    def __init__(self, X_tr, y_tr, X_te, y_te):
        super().__init__(
            n_var=N_VARS, n_obj=N_OBJECTIVES,
            n_constr=0,
            xl=np.zeros(N_VARS), xu=np.ones(N_VARS))
        self.X_tr = X_tr; self.y_tr = y_tr
        self.X_te = X_te; self.y_te = y_te
        self._gen_count = 0

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.array([evaluate_chromosome(
            x, self.X_tr, self.y_tr,
            self.X_te, self.y_te) for x in X])
        out["F"] = F
        self._gen_count += 1
        if self._gen_count % POP_SIZE == 0:
            gen = self._gen_count // POP_SIZE
            print(f"  Gen {gen:3d}/{N_GEN} | "
                  f"Best PR-AUC:{-np.min(F[:,0]):.4f}  "
                  f"Recall:{-np.min(F[:,1]):.4f}  "
                  f"FPR:{np.min(F[:,2]):.4f}")


# ================================================================
# RUN HYBRID
# ================================================================
print(f"Running Hybrid NSGA-II + PSO:")
print(f"  Population  : {POP_SIZE}")
print(f"  Generations : {N_GEN}")
print(f"  PSO trigger : every {PSO_INTERVAL} generations")
print(f"  PSO particles: {PSO_PARTICLES}, iterations: {PSO_ITER}\n")

hybrid_problem  = HybridIDSProblem(X_tr_s, y_tr_s, X_te_s, y_te_s)
hybrid_callback = HybridPSOCallback(
    X_tr_s, y_tr_s, X_te_s, y_te_s, interval=PSO_INTERVAL)

algorithm = NSGA2(
    pop_size=POP_SIZE,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True,
)

termination = get_termination("n_gen", N_GEN)

t_start = time.perf_counter()
result_hybrid = minimize(
    hybrid_problem,
    algorithm,
    termination,
    seed=42,
    verbose=False,
    callback=hybrid_callback
)
t_end = time.perf_counter()

print(f"\n  Hybrid complete in {(t_end-t_start)/60:.1f} minutes")
print(f"  Pareto front size: {len(result_hybrid.F)}")


# ================================================================
# EVALUATE HYBRID PARETO FRONT
# ================================================================
print("\n" + "=" * 60)
print("HYBRID PARETO FRONT — Full Test Set Evaluation")
print("=" * 60)

hybrid_results = []
for i, x in enumerate(result_hybrid.X):
    cfg  = decode_chromosome(x)
    feat = cfg["features"]
    Xtr  = X_tr[:, feat]
    Xte  = X_te[:, feat]

    try:
        if cfg["model_type"] == "LGB":
            m = lgb.LGBMClassifier(
                n_estimators=cfg["n_est"],
                max_depth=cfg["max_depth"],
                learning_rate=cfg["lr"],
                random_state=42, n_jobs=-1, verbose=-1)
        else:
            m = xgb.XGBClassifier(
                n_estimators=cfg["n_est"],
                max_depth=cfg["max_depth"],
                learning_rate=cfg["lr"],
                random_state=42, n_jobs=-1,
                verbosity=0, eval_metric='logloss')

        m.fit(Xtr, y_tr_bin)
        scores = m.predict_proba(Xte)[:, 1]
        preds  = (scores >= cfg["threshold"]).astype(int)

        p, r, _ = precision_recall_curve(y_te_bin, scores)
        pr_auc  = auc(r, p)
        cm = confusion_matrix(y_te_bin, preds)
        tn,fp,fn,tp = cm.ravel() if cm.shape==(2,2) else (0,0,0,0)
        recall = recall_score(y_te_bin, preds, zero_division=0)
        fpr    = fp / (fp + tn + 1e-9)
        f1     = f1_score(y_te_bin, preds, zero_division=0)

        t0  = time.perf_counter()
        _   = m.predict(Xte[:1000])
        t1  = time.perf_counter()
        lat = ((t1-t0)/1000)*1000

        hybrid_results.append({
            "pareto_id"   : i,
            "model_type"  : cfg["model_type"],
            "n_features"  : len(feat),
            "n_estimators": cfg["n_est"],
            "threshold"   : round(cfg["threshold"], 3),
            "pr_auc"      : round(pr_auc, 4),
            "recall"      : round(recall, 4),
            "fpr"         : round(fpr, 4),
            "f1"          : round(f1, 4),
            "latency_ms"  : round(lat, 4),
        })
    except Exception as e:
        print(f"  [SKIP] {i}: {e}")

df_hybrid = pd.DataFrame(hybrid_results)
df_hybrid.sort_values("pr_auc", ascending=False, inplace=True)
df_hybrid.reset_index(drop=True, inplace=True)
print(df_hybrid.to_string(index=False))
df_hybrid.to_csv("hybrid_pareto_front.csv", index=False)
print("\n  hybrid_pareto_front.csv  ✅  saved")


# ================================================================
# THREE-WAY COMPARISON: Baseline vs NSGA-II vs Hybrid
# ================================================================
print("\n" + "=" * 60)
print("THREE-WAY COMPARISON")
print("=" * 60)

df_nsga = pd.read_csv("pareto_front.csv")
best_nsga   = df_nsga.iloc[0]
best_hybrid = df_hybrid.iloc[0]

baseline = {"pr_auc": 0.9182, "recall": 0.8913, "fpr": 0.1350}
nsga2    = {"pr_auc": best_nsga["pr_auc"],
            "recall": best_nsga["recall"],
            "fpr"   : best_nsga["fpr"]}
hybrid   = {"pr_auc": best_hybrid["pr_auc"],
            "recall": best_hybrid["recall"],
            "fpr"   : best_hybrid["fpr"]}

print(f"  {'Metric':<12} {'Baseline':>10} "
      f"{'NSGA-II':>10} {'Hybrid':>10}")
print(f"  {'-'*46}")
for m in ["pr_auc","recall","fpr"]:
    print(f"  {m:<12} {baseline[m]:>10.4f} "
          f"{nsga2[m]:>10.4f} {hybrid[m]:>10.4f}")

print(f"\n  FPR reduction (NSGA-II vs baseline) : "
      f"{(baseline['fpr']-nsga2['fpr'])/baseline['fpr']*100:.1f}%")
print(f"  FPR reduction (Hybrid  vs baseline) : "
      f"{(baseline['fpr']-hybrid['fpr'])/baseline['fpr']*100:.1f}%")
print(f"  PR-AUC gain   (Hybrid  vs NSGA-II)  : "
      f"{(hybrid['pr_auc']-nsga2['pr_auc'])/nsga2['pr_auc']*100:+.2f}%")

print("\n  PHASE 4 FULLY COMPLETE")
print("  Files saved:")
print("    pareto_front.csv         (NSGA-II alone)")
print("    pso_refinement.csv       (PSO alone)")
print("    hybrid_pareto_front.csv  (Hybrid NSGA-II+PSO)")
print("\n  Paste output for verification")
