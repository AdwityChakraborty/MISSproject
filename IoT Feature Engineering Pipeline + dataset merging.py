import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("PHASE 2 — IoT-Specific Feature Engineering")
print("=" * 60)


# ================================================================
# HELPER — compute IoT features from a dataframe
# Requires columns: src_ip, dst_ip, service, src_pkts, dst_pkts,
#                   src_bytes, dst_bytes, duration (or Sintpkt)
# ================================================================
def compute_iot_features(df, src_ip_col, dst_ip_col,
                          service_col, sintpkt_col,
                          spkts_col, dpkts_col,
                          sbytes_col, dbytes_col):

    df = df.copy()

    # ── 1. Periodicity — IAT variance per source IP ──────────────
    iat_var = (df.groupby(src_ip_col)[sintpkt_col]
                 .transform('var')
                 .fillna(0))
    df['iot_iat_variance'] = iat_var

    # ── 2. Burstiness index — peak / mean packet rate per src IP ─
    # peak  = max pkts in any single flow for that src IP
    # mean  = average pkts across all flows for that src IP
    src_pkts_total = df[spkts_col].astype(float)
    peak  = df.groupby(src_ip_col)[spkts_col].transform('max').astype(float)
    mean  = df.groupby(src_ip_col)[spkts_col].transform('mean').astype(float)
    df['iot_burstiness'] = (peak / mean.replace(0, np.nan)).fillna(0)

    # ── 3. Service sparsity — unique services per source IP ──────
    df['iot_service_sparsity'] = (
        df.groupby(src_ip_col)[service_col]
          .transform('nunique')
          .astype(float)
    )

    # ── 4. Fan-out — unique destinations per source IP ───────────
    df['iot_fan_out'] = (
        df.groupby(src_ip_col)[dst_ip_col]
          .transform('nunique')
          .astype(float)
    )

    # ── 5. Fan-in — unique sources per destination IP ────────────
    df['iot_fan_in'] = (
        df.groupby(dst_ip_col)[src_ip_col]
          .transform('nunique')
          .astype(float)
    )

    new_cols = ['iot_iat_variance','iot_burstiness',
                'iot_service_sparsity','iot_fan_out','iot_fan_in']
    print(f"  IoT features added: {new_cols}")
    print(f"  Sample stats:\n"
          f"{df[new_cols].describe().round(4).to_string()}\n")

    return df, new_cols


# ================================================================
# UNSW-NB15 — reload raw merged file to get IP columns
# (parquet files had IPs dropped during preprocessing)
# ================================================================
print("─" * 60)
print("UNSW-NB15 IoT Features")
print("─" * 60)

unsw_cols = [
    'srcip','sport','dstip','dsport','proto','state','dur',
    'sbytes','dbytes','sttl','dttl','sloss','dloss','service',
    'Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb',
    'smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit',
    'Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat',
    'is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login',
    'ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ltm',
    'ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm',
    'attack_cat','Label'
]

UNSW_PATHS = [
    r"C:\Users\User\Downloads\Update of Project before EID\UNSW-NB15_1.csv",
    r"C:\Users\User\Downloads\Update of Project before EID\UNSW-NB15_2.csv",
    r"C:\Users\User\Downloads\Update of Project before EID\UNSW-NB15_3.csv",
    r"C:\Users\User\Downloads\Update of Project before EID\UNSW-NB15_4.csv",
]

parts = []
for path in UNSW_PATHS:
    part = pd.read_csv(path, header=None, names=unsw_cols,
                       usecols=['srcip','dstip','service',
                                'Sintpkt','Spkts','Dpkts',
                                'sbytes','dbytes','attack_cat','Label'],
                       low_memory=False)
    parts.append(part)

df_unsw_raw = pd.concat(parts, ignore_index=True)

# Drop garbage rows
df_unsw_raw = df_unsw_raw[
    pd.to_numeric(df_unsw_raw['sbytes'], errors='coerce').notna()]

# Convert numeric cols
for col in ['Sintpkt','Spkts','Dpkts','sbytes','dbytes']:
    df_unsw_raw[col] = pd.to_numeric(df_unsw_raw[col], errors='coerce').fillna(0)

df_unsw_raw['service'] = df_unsw_raw['service'].astype(str).str.strip()
df_unsw_raw['attack_cat'] = (df_unsw_raw['attack_cat']
                              .astype(str).str.strip().str.lower()
                              .replace('backdoors','backdoor'))
df_unsw_raw['binary_label'] = pd.to_numeric(
    df_unsw_raw['Label'], errors='coerce').fillna(0).astype(int)
df_unsw_raw.drop(columns=['Label'], inplace=True)

df_unsw_raw.reset_index(drop=True, inplace=True)
print(f"  Loaded UNSW raw : {df_unsw_raw.shape}")

df_unsw_iot, iot_cols = compute_iot_features(
    df_unsw_raw,
    src_ip_col   = 'srcip',
    dst_ip_col   = 'dstip',
    service_col  = 'service',
    sintpkt_col  = 'Sintpkt',
    spkts_col    = 'Spkts',
    dpkts_col    = 'Dpkts',
    sbytes_col   = 'sbytes',
    dbytes_col   = 'dbytes'
)

# Normalize the new IoT features
scaler_iot = MinMaxScaler()
df_unsw_iot[iot_cols] = scaler_iot.fit_transform(df_unsw_iot[iot_cols])

# Drop raw IP cols — not needed after feature computation
df_unsw_iot.drop(columns=['srcip','dstip','sport','dsport'],
                 errors='ignore', inplace=True)

# Save — these IoT feature columns will be merged into T0/T1/T2
df_unsw_iot[iot_cols + ['attack_cat','binary_label']].to_parquet(
    'unsw_iot_features.parquet')
print(f"  Saved unsw_iot_features.parquet — shape : "
      f"{df_unsw_iot[iot_cols].shape}\n")


# ================================================================
# ToN-IoT — compute IoT features
# ================================================================
print("─" * 60)
print("ToN-IoT IoT Features")
print("─" * 60)

df_ton_raw = pd.read_csv(r"train_test_network.csv", low_memory=False)
print(f"  Loaded ToN-IoT raw : {df_ton_raw.shape}")

# Convert numeric cols
for col in ['src_pkts','dst_pkts','src_bytes','dst_bytes','duration']:
    df_ton_raw[col] = pd.to_numeric(df_ton_raw[col], errors='coerce').fillna(0)

df_ton_raw['service'] = df_ton_raw['service'].astype(str).str.strip()
df_ton_raw['binary_label'] = df_ton_raw['label'].astype(int)

# Use duration as IAT proxy for ToN-IoT (no direct Sintpkt column)
df_ton_raw['iat_proxy'] = df_ton_raw['duration']

df_ton_iot, iot_cols_ton = compute_iot_features(
    df_ton_raw,
    src_ip_col   = 'src_ip',
    dst_ip_col   = 'dst_ip',
    service_col  = 'service',
    sintpkt_col  = 'iat_proxy',
    spkts_col    = 'src_pkts',
    dpkts_col    = 'dst_pkts',
    sbytes_col   = 'src_bytes',
    dbytes_col   = 'dst_bytes'
)

# Normalize
scaler_iot_ton = MinMaxScaler()
df_ton_iot[iot_cols_ton] = scaler_iot_ton.fit_transform(
    df_ton_iot[iot_cols_ton])

df_ton_iot[iot_cols_ton + ['label','type']].to_parquet(
    'ton_iot_features.parquet')
print(f"  Saved ton_iot_features.parquet — shape : "
      f"{df_ton_iot[iot_cols_ton].shape}\n")


# ================================================================
# MERGE IoT features back into existing parquet splits
# ================================================================
print("─" * 60)
print("Merging IoT features into preprocessed splits")
print("─" * 60)

# UNSW — merge into T0, T1, T2 by index alignment
for split_name in ['unsw_T0', 'unsw_T1', 'unsw_T2']:
    df_split = pd.read_parquet(f'{split_name}.parquet')
    n = len(df_split)

    if split_name == 'unsw_T0':
        iot_chunk = df_unsw_iot[iot_cols].iloc[:int(len(df_unsw_iot)*0.6)]\
                    .reset_index(drop=True)
    elif split_name == 'unsw_T1':
        start = int(len(df_unsw_iot)*0.6)
        end   = int(len(df_unsw_iot)*0.8)
        iot_chunk = df_unsw_iot[iot_cols].iloc[start:end]\
                    .reset_index(drop=True)
    else:
        iot_chunk = df_unsw_iot[iot_cols]\
                    .iloc[int(len(df_unsw_iot)*0.8):]\
                    .reset_index(drop=True)

    # Align lengths (may differ by a few rows due to garbage row drops)
    min_len = min(len(df_split), len(iot_chunk))
    df_merged = pd.concat(
        [df_split.iloc[:min_len].reset_index(drop=True),
         iot_chunk.iloc[:min_len].reset_index(drop=True)],
        axis=1)
    df_merged.to_parquet(f'{split_name}_with_iot.parquet')
    print(f"  {split_name}_with_iot.parquet  — shape : {df_merged.shape}  ✓")

# ToN-IoT — merge into train/test splits by index
df_ton_train = pd.read_parquet('ton_train.parquet')
df_ton_test  = pd.read_parquet('ton_test.parquet')

n_train = len(df_ton_train)
n_test  = len(df_ton_test)

iot_train_chunk = df_ton_iot[iot_cols_ton].iloc[:n_train].reset_index(drop=True)
iot_test_chunk  = df_ton_iot[iot_cols_ton].iloc[n_train:n_train+n_test]\
                  .reset_index(drop=True)

min_train = min(len(df_ton_train), len(iot_train_chunk))
min_test  = min(len(df_ton_test),  len(iot_test_chunk))

df_ton_train_iot = pd.concat(
    [df_ton_train.iloc[:min_train].reset_index(drop=True),
     iot_train_chunk.iloc[:min_train].reset_index(drop=True)], axis=1)

df_ton_test_iot = pd.concat(
    [df_ton_test.iloc[:min_test].reset_index(drop=True),
     iot_test_chunk.iloc[:min_test].reset_index(drop=True)], axis=1)

df_ton_train_iot.to_parquet('ton_train_with_iot.parquet')
df_ton_test_iot.to_parquet('ton_test_with_iot.parquet')
print(f"  ton_train_with_iot.parquet    — shape : {df_ton_train_iot.shape}  ✓")
print(f"  ton_test_with_iot.parquet     — shape : {df_ton_test_iot.shape}  ✓")

print("\n  PHASE 2 IoT FEATURE ENGINEERING COMPLETE")
print("  Paste output for verification")
