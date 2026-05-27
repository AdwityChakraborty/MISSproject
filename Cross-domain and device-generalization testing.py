import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (precision_recall_curve, auc,
                              recall_score, confusion_matrix,
                              f1_score)
import lightgbm as lgb
import xgboost as xgb

print("=" * 60)
print("E6 — ToN-IoT Device-Level Splits")
print("=" * 60)

# ================================================================
# Load raw ToN-IoT with device type labels
# ================================================================
df_ton = pd.read_csv(r"train_test_network.csv",
                      low_memory=False)
print(f"  Loaded : {df_ton.shape}")
print(f"  Attack types:\n"
      f"{df_ton['type'].value_counts().to_string()}\n")

# ================================================================
# HARMONIZED FEATURES for cross-dataset experiment
# ================================================================
N_FEATURES   = 32
group_size   = N_FEATURES // 5
BEST_FEATURES= list(range(group_size*2))  # first 12

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


def get_metrics(y_true, y_scores, y_pred, label=""):
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
# E6a — Device-level splits
# Train on: backdoor, ddos, dos, injection, password
# Test on:  ransomware, scanning, xss, mitm
# ================================================================
print("─" * 60)
print("E6a — Device-type generalization split")
print("─" * 60)

TRAIN_TYPES = ['backdoor','ddos','dos',
               'injection','password','normal']
TEST_TYPES  = ['ransomware','scanning','xss','mitm','normal']

df_train_dev = df_ton[df_ton['type'].isin(TRAIN_TYPES)].copy()
df_test_dev  = df_ton[df_ton['type'].isin(TEST_TYPES)].copy()

print(f"  Train types : {TRAIN_TYPES}")
print(f"  Test types  : {TEST_TYPES}")
print(f"  Train size  : {len(df_train_dev):,}")
print(f"  Test size   : {len(df_test_dev):,}")
print(f"  Train distribution:\n"
      f"{df_train_dev['type'].value_counts().to_string()}")
print(f"  Test distribution:\n"
      f"{df_test_dev['type'].value_counts().to_string()}\n")

# Binary labels
y_tr_dev = (df_train_dev['type'] != 'normal').astype(int)
y_te_dev = (df_test_dev['type']  != 'normal').astype(int)

# Drop high-cardinality and label columns
drop_cols = ['src_ip','dst_ip','dns_query','ssl_subject',
             'ssl_issuer','http_uri','http_user_agent',
             'http_orig_mime_types','http_resp_mime_types',
             'weird_name','weird_addl','ssl_cipher',
             'conn_state','label','type']

df_train_dev.drop(columns=[c for c in drop_cols
                             if c in df_train_dev.columns],
                   inplace=True)
df_test_dev.drop(columns=[c for c in drop_cols
                            if c in df_test_dev.columns],
                  inplace=True)

# Fill nulls and encode
for col in df_train_dev.columns:
    if df_train_dev[col].dtype == 'object':
        df_train_dev[col].fillna('-', inplace=True)
        df_test_dev[col].fillna('-', inplace=True)
    else:
        df_train_dev[col].fillna(0, inplace=True)
        df_test_dev[col].fillna(0, inplace=True)

df_train_dev = pd.get_dummies(df_train_dev)
df_test_dev  = pd.get_dummies(df_test_dev)
df_train_dev, df_test_dev = df_train_dev.align(
    df_test_dev, join='left', axis=1, fill_value=0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_tr_dev = scaler.fit_transform(df_train_dev)
X_te_dev = scaler.transform(df_test_dev)

e6a_results = []
for name, clf in [
    ("LightGBM", lgb.LGBMClassifier(
        n_estimators=200, random_state=42,
        n_jobs=-1, verbose=-1)),
    ("XGBoost",  xgb.XGBClassifier(
        n_estimators=200, random_state=42,
        n_jobs=-1, verbosity=0,
        eval_metric='logloss')),
]:
    print(f"\n  [{name} | E6a device-split]")
    clf.fit(X_tr_dev, y_tr_dev)
    scores = clf.predict_proba(X_te_dev)[:, 1]
    preds  = clf.predict(X_te_dev)
    m = get_metrics(y_te_dev, scores, preds,
                    f"{name} device-split")
    e6a_results.append({"experiment":"E6a:device-split",
                         "model":name, **m})


# ================================================================
# E6b — Cross-dataset: ToN-IoT → UNSW-NB15
# Train on full ToN-IoT harmonized
# Test on UNSW-NB15 T2 harmonized
# ================================================================
print("\n" + "─" * 60)
print("E6b — Cross-dataset: ToN-IoT → UNSW-NB15")
print("─" * 60)

X_ton_tr = pd.read_parquet("H_ton_train.parquet").values
y_ton_tr = pd.read_parquet(
    "H_ton_train_labels.parquet").iloc[:,0].values
X_unsw_te= pd.read_parquet("H_unsw_T2.parquet").values
y_unsw_te= pd.read_parquet(
    "H_unsw_T2_labels.parquet").iloc[:,0].values

# Sample test for speed
idx_te = np.random.RandomState(42).choice(
    len(X_unsw_te), 20_000, replace=False)
X_unsw_te_s = X_unsw_te[idx_te]
y_unsw_te_s = y_unsw_te[idx_te]

print(f"  Train (ToN-IoT)   : {X_ton_tr.shape}")
print(f"  Test  (UNSW-NB15) : {X_unsw_te_s.shape}")

e6b_results = []
for name, clf in [
    ("LightGBM", lgb.LGBMClassifier(
        n_estimators=200, random_state=42,
        n_jobs=-1, verbose=-1)),
    ("XGBoost",  xgb.XGBClassifier(
        n_estimators=200, random_state=42,
        n_jobs=-1, verbosity=0,
        eval_metric='logloss')),
]:
    print(f"\n  [{name} | E6b ToN→UNSW]")
    clf.fit(X_ton_tr, y_ton_tr)
    scores = clf.predict_proba(X_unsw_te_s)[:, 1]
    preds  = clf.predict(X_unsw_te_s)
    m = get_metrics(y_unsw_te_s, scores, preds,
                    f"{name} ToN→UNSW")
    e6b_results.append({"experiment":"E6b:ToN→UNSW",
                         "model":name, **m})


# ================================================================
# SAVE ALL E6 RESULTS
# ================================================================
df_e6 = pd.DataFrame(e6a_results + e6b_results)
df_e6.to_csv("e6_results.csv", index=False)

print("\n" + "=" * 60)
print("E6 COMPLETE RESULTS")
print("=" * 60)
print(df_e6.to_string(index=False))
print("\n  e6_results.csv  ✅  saved")


# ================================================================
# UPDATE baseline_all.csv with E6 device results
# ================================================================
df_all = pd.read_csv("baseline_all.csv")
df_e6["view"] = "tabular"
df_e6["latency_ms"] = 0.0
df_e6["memory_mb"]  = 0.0
df_e6.rename(columns={"experiment":"dataset"},
              inplace=True)
df_all = pd.concat([df_all, df_e6], ignore_index=True)
df_all.to_csv("baseline_all.csv", index=False)
print("  baseline_all.csv  ✅  updated with E6 results")
print("\n  Paste output for verification")
