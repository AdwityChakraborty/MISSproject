import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

print("Regenerating unsw_iot_profile.parquet...")

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

# Load raw files
parts = []
for path in UNSW_PATHS:
    part = pd.read_csv(path, header=None, names=unsw_cols, low_memory=False)
    parts.append(part)
df_unsw_raw = pd.concat(parts, ignore_index=True)

# Drop garbage rows
df_unsw_raw = df_unsw_raw[
    pd.to_numeric(df_unsw_raw['sbytes'], errors='coerce').notna()]
df_unsw_raw.reset_index(drop=True, inplace=True)
print(f"  Loaded raw UNSW : {df_unsw_raw.shape}")

# Convert numeric cols needed for filtering
for col in ['sbytes','dbytes','Sintpkt']:
    df_unsw_raw[col] = pd.to_numeric(
        df_unsw_raw[col], errors='coerce').fillna(0)
df_unsw_raw['service'] = df_unsw_raw['service'].astype(str).str.strip()

# IoT profile filter
iot_mask = (
    (df_unsw_raw['sbytes'] + df_unsw_raw['dbytes'] <= 512) &
    (df_unsw_raw['Sintpkt'] <= 0.5) &
    (~df_unsw_raw['service'].isin(['-','unknown','nan']))
)
df_iot = df_unsw_raw[iot_mask].copy()
print(f"  IoT profile rows : {len(df_iot):,}")
print(f"  Attack categories:")
print(df_iot['attack_cat'].astype(str).str.strip().str.lower()
      .value_counts().to_string())

# Preprocess
df_iot['attack_cat']   = df_iot['attack_cat'].astype(str).str.strip().str.lower()
df_iot['attack_cat']   = df_iot['attack_cat'].replace('backdoors','backdoor')
df_iot['binary_label'] = pd.to_numeric(
    df_iot['Label'], errors='coerce').fillna(0).astype(int)

drop_cols = ['srcip','dstip','sport','dsport','Stime','Ltime','Label']
df_iot.drop(columns=[c for c in drop_cols if c in df_iot.columns],
            inplace=True)

# Fill nulls
for col in df_iot.columns:
    if df_iot[col].dtype == 'object':
        df_iot[col].fillna('unknown', inplace=True)
    else:
        df_iot[col].fillna(df_iot[col].median(), inplace=True)

# Clean ct_ftp_cmd
if 'ct_ftp_cmd' in df_iot.columns:
    df_iot['ct_ftp_cmd'] = df_iot['ct_ftp_cmd'].astype(str).str.strip()

# Encode
protected = ['attack_cat','binary_label']
cat_cols  = [c for c in df_iot.select_dtypes(include='object').columns
             if c not in protected]
df_iot = pd.get_dummies(df_iot, columns=cat_cols)
df_iot = df_iot.loc[:, ~df_iot.columns.duplicated(keep='first')]

# Normalize
feat_cols = [c for c in df_iot.select_dtypes(include=[np.number]).columns
             if c != 'binary_label']
df_iot[feat_cols] = MinMaxScaler().fit_transform(df_iot[feat_cols])

# Save
df_iot.to_parquet('unsw_iot_profile.parquet')
print(f"\n  Final shape : {df_iot.shape}")
print("  unsw_iot_profile.parquet  ✅")
