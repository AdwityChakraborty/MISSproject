import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("PHASE 2 — Sequence View (Sliding Window)")
print("=" * 60)

# ================================================================
# CONCEPT
# A sliding window of size W is applied over flows ordered by
# source IP. Each sample becomes a (W x F) matrix representing
# the last W flows from that source — capturing temporal patterns.
# We flatten to (W*F,) for tabular storage; the 1D-CNN/TCN will
# reshape back to (W, F) during training.
# ================================================================

WINDOW_SIZE = 10   # 10 consecutive flows per sequence
                   # kept small so it works on all datasets


def build_sequences(X, y, window_size=10, src_ip_col=None,
                    dataset_name=""):
    """
    Build sliding window sequences from a feature matrix.
    If src_ip_col is provided, windows are built per source IP.
    Otherwise windows are built over the full sorted DataFrame.

    Returns:
        X_seq : np.ndarray of shape (n_samples, window_size * n_features)
        y_seq : np.ndarray of shape (n_samples,)
               label is taken from the LAST row of each window
    """
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    n_features = X.shape[1]

    X_seq_list = []
    y_seq_list = []

    if src_ip_col is not None and src_ip_col in X.columns:
        groups = X.groupby(src_ip_col).groups
        for ip, idx in groups.items():
            idx = sorted(idx)
            arr = X.iloc[idx].values
            lbl = y.iloc[idx].values
            for i in range(len(arr) - window_size + 1):
                window = arr[i:i + window_size].flatten()
                X_seq_list.append(window)
                y_seq_list.append(lbl[i + window_size - 1])
    else:
        arr = X.values
        lbl = y.values
        for i in range(len(arr) - window_size + 1):
            window = arr[i:i + window_size].flatten()
            X_seq_list.append(window)
            y_seq_list.append(lbl[i + window_size - 1])

    X_seq = np.array(X_seq_list, dtype=np.float32)
    y_seq = np.array(y_seq_list)

    print(f"  {dataset_name}")
    print(f"    Input  : {X.shape}")
    print(f"    Output : {X_seq.shape}  "
          f"(samples x window*features)")
    print(f"    Label distribution : "
          f"{pd.Series(y_seq).value_counts().to_dict()}\n")

    return X_seq, y_seq


# ================================================================
# NSL-KDD — sequence view
# ================================================================
print("─" * 60)
print("NSL-KDD")
print("─" * 60)

df_nsl_tr = pd.read_parquet("nsl_train.parquet")
df_nsl_te = pd.read_parquet("nsl_test.parquet")

feat_nsl  = [c for c in df_nsl_tr.columns
             if c not in ["label", "label_cat", "binary_label"]]
X_nsl_tr  = df_nsl_tr[feat_nsl]
y_nsl_tr  = df_nsl_tr["binary_label"]
X_nsl_te  = df_nsl_te[feat_nsl]
y_nsl_te  = df_nsl_te["binary_label"]

# Align columns
X_nsl_tr, X_nsl_te = X_nsl_tr.align(X_nsl_te, join='inner', axis=1)

X_nsl_seq_tr, y_nsl_seq_tr = build_sequences(
    X_nsl_tr, y_nsl_tr, WINDOW_SIZE, dataset_name="NSL-KDD train")
X_nsl_seq_te, y_nsl_seq_te = build_sequences(
    X_nsl_te, y_nsl_te, WINDOW_SIZE, dataset_name="NSL-KDD test")

np.save("seq_nsl_X_train.npy", X_nsl_seq_tr)
np.save("seq_nsl_y_train.npy", y_nsl_seq_tr)
np.save("seq_nsl_X_test.npy",  X_nsl_seq_te)
np.save("seq_nsl_y_test.npy",  y_nsl_seq_te)
print("  Saved seq_nsl_X/y_train/test.npy  ✅\n")


# ================================================================
# UNSW-NB15 T0 — sequence view (sample for memory)
# ================================================================
print("─" * 60)
print("UNSW-NB15 T0")
print("─" * 60)

df_unsw_T0 = pd.read_parquet("unsw_T0_with_iot.parquet")
df_unsw_T2 = pd.read_parquet("unsw_T2_with_iot.parquet")

feat_unsw  = [c for c in df_unsw_T0.columns
              if c not in ["attack_cat", "binary_label"]]

# Sample to keep memory manageable
df_unsw_T0_s = df_unsw_T0.sample(n=60_000, random_state=42)
df_unsw_T2_s = df_unsw_T2.sample(n=20_000, random_state=42)

X_unsw_tr_s = df_unsw_T0_s[feat_unsw]
y_unsw_tr_s = df_unsw_T0_s["binary_label"]
X_unsw_te_s = df_unsw_T2_s[feat_unsw]
y_unsw_te_s = df_unsw_T2_s["binary_label"]

X_unsw_seq_tr, y_unsw_seq_tr = build_sequences(
    X_unsw_tr_s, y_unsw_tr_s, WINDOW_SIZE,
    dataset_name="UNSW-NB15 T0 train (sampled)")
X_unsw_seq_te, y_unsw_seq_te = build_sequences(
    X_unsw_te_s, y_unsw_te_s, WINDOW_SIZE,
    dataset_name="UNSW-NB15 T2 test (sampled)")

np.save("seq_unsw_X_train.npy", X_unsw_seq_tr)
np.save("seq_unsw_y_train.npy", y_unsw_seq_tr)
np.save("seq_unsw_X_test.npy",  X_unsw_seq_te)
np.save("seq_unsw_y_test.npy",  y_unsw_seq_te)
print("  Saved seq_unsw_X/y_train/test.npy  ✅\n")


# ================================================================
# ToN-IoT — sequence view
# ================================================================
print("─" * 60)
print("ToN-IoT")
print("─" * 60)

df_ton_tr = pd.read_parquet("ton_train_with_iot.parquet")
df_ton_te = pd.read_parquet("ton_test_with_iot.parquet")

feat_ton  = [c for c in df_ton_tr.columns
             if c not in ["label", "type", "binary_label"]]
X_ton_tr  = df_ton_tr[feat_ton]
y_ton_tr  = (df_ton_tr["label"].astype(str)
              .str.lower() != "normal").astype(int)
X_ton_te  = df_ton_te[feat_ton]
y_ton_te  = (df_ton_te["label"].astype(str)
              .str.lower() != "normal").astype(int)

X_ton_tr, X_ton_te = X_ton_tr.align(X_ton_te, join='inner', axis=1)

X_ton_seq_tr, y_ton_seq_tr = build_sequences(
    X_ton_tr, y_ton_tr, WINDOW_SIZE, dataset_name="ToN-IoT train")
X_ton_seq_te, y_ton_seq_te = build_sequences(
    X_ton_te, y_ton_te, WINDOW_SIZE, dataset_name="ToN-IoT test")

np.save("seq_ton_X_train.npy", X_ton_seq_tr)
np.save("seq_ton_y_train.npy", y_ton_seq_tr)
np.save("seq_ton_X_test.npy",  X_ton_seq_te)
np.save("seq_ton_y_test.npy",  y_ton_seq_te)
print("  Saved seq_ton_X/y_train/test.npy  ✅\n")


# ================================================================
# VERIFY all sequence files
# ================================================================
import os
print("=" * 60)
print("SEQUENCE VIEW FILE INVENTORY")
print("=" * 60)
seq_files = [
    "seq_nsl_X_train.npy",  "seq_nsl_y_train.npy",
    "seq_nsl_X_test.npy",   "seq_nsl_y_test.npy",
    "seq_unsw_X_train.npy", "seq_unsw_y_train.npy",
    "seq_unsw_X_test.npy",  "seq_unsw_y_test.npy",
    "seq_ton_X_train.npy",  "seq_ton_y_train.npy",
    "seq_ton_X_test.npy",   "seq_ton_y_test.npy",
]
for f in seq_files:
    if os.path.exists(f):
        arr = np.load(f)
        print(f"  ✅  {f:35s}  shape: {arr.shape}")
    else:
        print(f"  ❌  {f:35s}  MISSING")

print(f"\n  Window size used : {WINDOW_SIZE}")
print(f"  Each sequence shape when reshaped for 1D-CNN : "
      f"({WINDOW_SIZE}, n_features)")
print("\n  SEQUENCE VIEW COMPLETE — paste output for verification")
