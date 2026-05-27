import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("HARMONIZED META-SCHEMA")
print("=" * 60)

# ================================================================
# The common feature set is built from features that exist
# across KDD'99, NSL-KDD, UNSW-NB15 and ToN-IoT.
# Features missing in a dataset are zero-filled.
# IoT-specific features are included but zero-filled for
# KDD'99 and NSL-KDD (data unavailable for those datasets).
# ================================================================

# ── Common core features shared by KDD/NSL and UNSW/ToN ─────────
# KDD'99 and NSL-KDD share these with UNSW-NB15 via mapping:
#   duration        → dur
#   src_bytes       → sbytes
#   dst_bytes       → dbytes
#   src_pkts proxy  → Spkts
#   dst_pkts proxy  → Dpkts
#   serror_rate     → ct_srv_src (proxy)
#   same_srv_rate   → ct_srv_dst (proxy)

COMMON_FEATURES = [
    # Connection statistics
    'f_duration',       # dur / duration
    'f_src_bytes',      # sbytes / src_bytes
    'f_dst_bytes',      # dbytes / dst_bytes
    'f_src_pkts',       # Spkts / src_pkts
    'f_dst_pkts',       # Dpkts / dst_pkts
    'f_total_bytes',    # sbytes+dbytes / src_bytes+dst_bytes

    # Traffic ratios
    'f_bytes_per_pkt',  # total_bytes / total_pkts
    'f_src_load',       # Sload / src_bytes/duration
    'f_dst_load',       # Dload / dst_bytes/duration
    'f_serror_rate',    # serror_rate / ct_srv_src proxy
    'f_rerror_rate',    # rerror_rate
    'f_same_srv_rate',  # same_srv_rate / ct_srv_dst proxy
    'f_diff_srv_rate',  # diff_srv_rate

    # Protocol one-hot (present in all datasets)
    'f_proto_tcp',
    'f_proto_udp',
    'f_proto_icmp',
    'f_proto_other',

    # Service one-hot (common services)
    'f_svc_http',
    'f_svc_ftp',
    'f_svc_smtp',
    'f_svc_dns',
    'f_svc_ssh',
    'f_svc_other',
    'f_svc_none',

    # Connection state
    'f_state_established',  # SF/FIN/CON → established
    'f_state_rejected',     # REJ/RST     → rejected
    'f_state_other',

    # IoT-specific (zero-filled for KDD'99 and NSL-KDD)
    'f_iot_iat_variance',
    'f_iot_burstiness',
    'f_iot_service_sparsity',
    'f_iot_fan_out',
    'f_iot_fan_in',
]

print(f"  Total harmonized features : {len(COMMON_FEATURES)}")
print(f"  Features: {COMMON_FEATURES}\n")


# ================================================================
# MAPPING FUNCTIONS — one per dataset
# ================================================================

def map_kdd(df):
    """Map KDD'99 or NSL-KDD columns to harmonized schema."""
    out = pd.DataFrame(index=df.index)

    out['f_duration']      = df.get('duration', 0)
    out['f_src_bytes']     = df.get('src_bytes', 0)
    out['f_dst_bytes']     = df.get('dst_bytes', 0)
    out['f_src_pkts']      = df.get('count', 0)       # best proxy
    out['f_dst_pkts']      = df.get('srv_count', 0)
    out['f_total_bytes']   = df.get('src_bytes', 0) + df.get('dst_bytes', 0)
    out['f_bytes_per_pkt'] = (out['f_total_bytes'] /
                               (out['f_src_pkts'] + out['f_dst_pkts'] + 1e-9))
    out['f_src_load']      = df.get('src_bytes', 0)   # no direct load col
    out['f_dst_load']      = df.get('dst_bytes', 0)
    out['f_serror_rate']   = df.get('serror_rate', 0)
    out['f_rerror_rate']   = df.get('rerror_rate', 0)
    out['f_same_srv_rate'] = df.get('same_srv_rate', 0)
    out['f_diff_srv_rate'] = df.get('diff_srv_rate', 0)

    # Protocol — check one-hot columns produced by get_dummies
    out['f_proto_tcp']  = df.get('protocol_type_tcp',  0)
    out['f_proto_udp']  = df.get('protocol_type_udp',  0)
    out['f_proto_icmp'] = df.get('protocol_type_icmp', 0)
    out['f_proto_other']= 1 - (out['f_proto_tcp'] +
                                out['f_proto_udp'] +
                                out['f_proto_icmp']).clip(0, 1)

    # Service
    out['f_svc_http']  = df.get('service_http',     0)
    out['f_svc_ftp']   = df.get('service_ftp',      0)
    out['f_svc_smtp']  = df.get('service_smtp',     0)
    out['f_svc_dns']   = df.get('service_domain',   0)
    out['f_svc_ssh']   = df.get('service_ssh',      0)
    out['f_svc_none']  = df.get('service_private',  0)
    out['f_svc_other'] = 1 - (out[['f_svc_http','f_svc_ftp','f_svc_smtp',
                                    'f_svc_dns','f_svc_ssh','f_svc_none']]
                               .sum(axis=1)).clip(0, 1)

    # Connection state via flag column
    out['f_state_established'] = df.get('flag_SF',  0)
    out['f_state_rejected']    = df.get('flag_REJ', 0)
    out['f_state_other']       = 1 - (out['f_state_established'] +
                                       out['f_state_rejected']).clip(0, 1)

    # IoT features — zero-filled (not available in KDD datasets)
    for col in ['f_iot_iat_variance','f_iot_burstiness',
                'f_iot_service_sparsity','f_iot_fan_out','f_iot_fan_in']:
        out[col] = 0.0

    return out[COMMON_FEATURES].fillna(0).astype(float)


def map_unsw(df):
    """Map UNSW-NB15 columns to harmonized schema."""
    out = pd.DataFrame(index=df.index)

    out['f_duration']      = df.get('dur', 0)
    out['f_src_bytes']     = df.get('sbytes', 0)
    out['f_dst_bytes']     = df.get('dbytes', 0)
    out['f_src_pkts']      = df.get('Spkts', 0)
    out['f_dst_pkts']      = df.get('Dpkts', 0)
    out['f_total_bytes']   = df.get('sbytes', 0) + df.get('dbytes', 0)
    out['f_bytes_per_pkt'] = (out['f_total_bytes'] /
                               (out['f_src_pkts'] + out['f_dst_pkts'] + 1e-9))
    out['f_src_load']      = df.get('Sload', 0)
    out['f_dst_load']      = df.get('Dload', 0)
    out['f_serror_rate']   = df.get('ct_srv_src', 0)
    out['f_rerror_rate']   = df.get('ct_srv_dst', 0)
    out['f_same_srv_rate'] = df.get('ct_dst_ltm', 0)
    out['f_diff_srv_rate'] = df.get('ct_src_ltm', 0)

    out['f_proto_tcp']   = df.get('proto_tcp',  0)
    out['f_proto_udp']   = df.get('proto_udp',  0)
    out['f_proto_icmp']  = df.get('proto_icmp', 0)
    out['f_proto_other'] = 1 - (out['f_proto_tcp'] +
                                 out['f_proto_udp'] +
                                 out['f_proto_icmp']).clip(0, 1)

    out['f_svc_http']  = df.get('service_http',     0)
    out['f_svc_ftp']   = df.get('service_ftp',      0)
    out['f_svc_smtp']  = df.get('service_smtp',     0)
    out['f_svc_dns']   = df.get('service_dns',      0)
    out['f_svc_ssh']   = df.get('service_ssh',      0)
    out['f_svc_none']  = df.get('service_-',        0)
    out['f_svc_other'] = 1 - (out[['f_svc_http','f_svc_ftp','f_svc_smtp',
                                    'f_svc_dns','f_svc_ssh','f_svc_none']]
                               .sum(axis=1)).clip(0, 1)

    out['f_state_established'] = df.get('state_FIN', 0) + df.get('state_CON', 0)
    out['f_state_rejected']    = df.get('state_RST', 0) + df.get('state_REQ', 0)
    out['f_state_other']       = 1 - (out['f_state_established'] +
                                       out['f_state_rejected']).clip(0, 1)

    # IoT features — available
    out['f_iot_iat_variance']    = df.get('iot_iat_variance',    0)
    out['f_iot_burstiness']      = df.get('iot_burstiness',      0)
    out['f_iot_service_sparsity']= df.get('iot_service_sparsity',0)
    out['f_iot_fan_out']         = df.get('iot_fan_out',         0)
    out['f_iot_fan_in']          = df.get('iot_fan_in',          0)

    return out[COMMON_FEATURES].fillna(0).astype(float)


def map_ton(df):
    """Map ToN-IoT columns to harmonized schema."""
    out = pd.DataFrame(index=df.index)

    out['f_duration']      = df.get('duration', 0)
    out['f_src_bytes']     = df.get('src_bytes', 0)
    out['f_dst_bytes']     = df.get('dst_bytes', 0)
    out['f_src_pkts']      = df.get('src_pkts', 0)
    out['f_dst_pkts']      = df.get('dst_pkts', 0)
    out['f_total_bytes']   = df.get('src_bytes', 0) + df.get('dst_bytes', 0)
    out['f_bytes_per_pkt'] = (out['f_total_bytes'] /
                               (out['f_src_pkts'] + out['f_dst_pkts'] + 1e-9))
    out['f_src_load']      = df.get('src_bytes', 0)
    out['f_dst_load']      = df.get('dst_bytes', 0)
    out['f_serror_rate']   = df.get('src_ip_bytes', 0)
    out['f_rerror_rate']   = df.get('dst_ip_bytes', 0)
    out['f_same_srv_rate'] = df.get('missed_bytes', 0)
    out['f_diff_srv_rate'] = df.get('src_port',    0)

    out['f_proto_tcp']   = df.get('proto_tcp',  0)
    out['f_proto_udp']   = df.get('proto_udp',  0)
    out['f_proto_icmp']  = df.get('proto_icmp', 0)
    out['f_proto_other'] = 1 - (out['f_proto_tcp'] +
                                 out['f_proto_udp'] +
                                 out['f_proto_icmp']).clip(0, 1)

    out['f_svc_http']  = df.get('service_http',  0)
    out['f_svc_ftp']   = df.get('service_ftp',   0)
    out['f_svc_smtp']  = df.get('service_smtp',  0)
    out['f_svc_dns']   = df.get('service_dns',   0)
    out['f_svc_ssh']   = df.get('service_ssh',   0)
    out['f_svc_none']  = df.get('service_-',     0)
    out['f_svc_other'] = 1 - (out[['f_svc_http','f_svc_ftp','f_svc_smtp',
                                    'f_svc_dns','f_svc_ssh','f_svc_none']]
                               .sum(axis=1)).clip(0, 1)

    out['f_state_established'] = df.get('conn_state_SF',  0)
    out['f_state_rejected']    = df.get('conn_state_REJ', 0)
    out['f_state_other']       = 1 - (out['f_state_established'] +
                                       out['f_state_rejected']).clip(0, 1)

    out['f_iot_iat_variance']    = df.get('iot_iat_variance',    0)
    out['f_iot_burstiness']      = df.get('iot_burstiness',      0)
    out['f_iot_service_sparsity']= df.get('iot_service_sparsity',0)
    out['f_iot_fan_out']         = df.get('iot_fan_out',         0)
    out['f_iot_fan_in']          = df.get('iot_fan_in',          0)

    return out[COMMON_FEATURES].fillna(0).astype(float)


# ================================================================
# APPLY MAPPING TO ALL DATASETS AND SAVE
# ================================================================

# ── KDD'99 ───────────────────────────────────────────────────────
print("  Mapping KDD'99...")
X_kdd_tr = pd.read_parquet("kdd99_train_smote.parquet")
X_kdd_te = pd.read_parquet("kdd99_test.parquet")
y_kdd_tr = pd.read_parquet("kdd99_train_smote_labels.parquet").iloc[:, 0]
y_kdd_te = pd.read_parquet("kdd99_test_labels.parquet").iloc[:, 0]

H_kdd_tr = map_kdd(X_kdd_tr)
H_kdd_te = map_kdd(X_kdd_te)

# Normalize
scaler = MinMaxScaler()
H_kdd_tr = pd.DataFrame(scaler.fit_transform(H_kdd_tr),
                          columns=COMMON_FEATURES)
H_kdd_te = pd.DataFrame(scaler.transform(H_kdd_te),
                          columns=COMMON_FEATURES)
H_kdd_tr.to_parquet("H_kdd_train.parquet")
H_kdd_te.to_parquet("H_kdd_test.parquet")
y_kdd_tr.to_frame().to_parquet("H_kdd_train_labels.parquet")
y_kdd_te.to_frame().to_parquet("H_kdd_test_labels.parquet")
print(f"  KDD'99  → train {H_kdd_tr.shape} | test {H_kdd_te.shape}  ✓")


# ── NSL-KDD ──────────────────────────────────────────────────────
print("  Mapping NSL-KDD...")
df_nsl_tr = pd.read_parquet("nsl_train.parquet")
df_nsl_te = pd.read_parquet("nsl_test.parquet")
y_nsl_tr  = df_nsl_tr["label_cat"]
y_nsl_te  = df_nsl_te["label_cat"]

H_nsl_tr = map_kdd(df_nsl_tr)
H_nsl_te = map_kdd(df_nsl_te)

scaler = MinMaxScaler()
H_nsl_tr = pd.DataFrame(scaler.fit_transform(H_nsl_tr),
                          columns=COMMON_FEATURES)
H_nsl_te = pd.DataFrame(scaler.transform(H_nsl_te),
                          columns=COMMON_FEATURES)
H_nsl_tr.to_parquet("H_nsl_train.parquet")
H_nsl_te.to_parquet("H_nsl_test.parquet")
y_nsl_tr.to_frame().to_parquet("H_nsl_train_labels.parquet")
y_nsl_te.to_frame().to_parquet("H_nsl_test_labels.parquet")
print(f"  NSL-KDD → train {H_nsl_tr.shape} | test {H_nsl_te.shape}  ✓")


# ── UNSW-NB15 ────────────────────────────────────────────────────
print("  Mapping UNSW-NB15...")
df_unsw_T0 = pd.read_parquet("unsw_T0_with_iot.parquet")
df_unsw_T1 = pd.read_parquet("unsw_T1_with_iot.parquet")
df_unsw_T2 = pd.read_parquet("unsw_T2_with_iot.parquet")
y_unsw_T0  = df_unsw_T0["binary_label"]
y_unsw_T1  = df_unsw_T1["binary_label"]
y_unsw_T2  = df_unsw_T2["binary_label"]

H_unsw_T0 = map_unsw(df_unsw_T0)
H_unsw_T1 = map_unsw(df_unsw_T1)
H_unsw_T2 = map_unsw(df_unsw_T2)

scaler = MinMaxScaler()
H_unsw_T0 = pd.DataFrame(scaler.fit_transform(H_unsw_T0),
                           columns=COMMON_FEATURES)
H_unsw_T1 = pd.DataFrame(scaler.transform(H_unsw_T1),
                           columns=COMMON_FEATURES)
H_unsw_T2 = pd.DataFrame(scaler.transform(H_unsw_T2),
                           columns=COMMON_FEATURES)

H_unsw_T0.to_parquet("H_unsw_T0.parquet")
H_unsw_T1.to_parquet("H_unsw_T1.parquet")
H_unsw_T2.to_parquet("H_unsw_T2.parquet")
y_unsw_T0.to_frame().to_parquet("H_unsw_T0_labels.parquet")
y_unsw_T1.to_frame().to_parquet("H_unsw_T1_labels.parquet")
y_unsw_T2.to_frame().to_parquet("H_unsw_T2_labels.parquet")
print(f"  UNSW T0 → {H_unsw_T0.shape} | T1 → {H_unsw_T1.shape} "
      f"| T2 → {H_unsw_T2.shape}  ✓")


# ── ToN-IoT ──────────────────────────────────────────────────────
print("  Mapping ToN-IoT...")
df_ton_tr = pd.read_parquet("ton_train_with_iot.parquet")
df_ton_te = pd.read_parquet("ton_test_with_iot.parquet")
y_ton_tr  = (df_ton_tr["label"].astype(str).str.lower() != "normal").astype(int)
y_ton_te  = (df_ton_te["label"].astype(str).str.lower() != "normal").astype(int)

H_ton_tr = map_ton(df_ton_tr)
H_ton_te = map_ton(df_ton_te)

scaler = MinMaxScaler()
H_ton_tr = pd.DataFrame(scaler.fit_transform(H_ton_tr),
                          columns=COMMON_FEATURES)
H_ton_te = pd.DataFrame(scaler.transform(H_ton_te),
                          columns=COMMON_FEATURES)
H_ton_tr.to_parquet("H_ton_train.parquet")
H_ton_te.to_parquet("H_ton_test.parquet")
y_ton_tr.to_frame().to_parquet("H_ton_train_labels.parquet")
y_ton_te.to_frame().to_parquet("H_ton_test_labels.parquet")
print(f"  ToN-IoT → train {H_ton_tr.shape} | test {H_ton_te.shape}  ✓")


# ================================================================
# VERIFICATION — all datasets must have identical columns
# ================================================================
print("\n" + "=" * 60)
print("VERIFICATION — all H_ files must have same 31 columns")
print("=" * 60)
for name, df in [("KDD train",   H_kdd_tr),
                  ("NSL train",   H_nsl_tr),
                  ("UNSW T0",     H_unsw_T0),
                  ("ToN train",   H_ton_tr)]:
    match = (list(df.columns) == COMMON_FEATURES)
    print(f"  {name:15s} : {df.shape}  columns match={match}")

print("\n  H_ parquet files are ready for cross-dataset experiments")
print("  Paste output for verification")
