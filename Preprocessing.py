import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# RAW FILE PATHS — update these if needed
# ================================================================
KDD99_PATH     = r"C:\Users\User\Downloads\Update of Project before EID\kddcup99_csv.csv"
NSL_TRAIN_PATH = r"C:\Users\User\Downloads\Update of Project before EID\KDDTrain+.txt"
NSL_TEST_PATH  = r"C:\Users\User\Downloads\Update of Project before EID\KDDTest+.txt"
UNSW_PATHS     = [
    r"C:\Users\User\Downloads\Update of Project before EID\UNSW-NB15_1.csv",
    r"C:\Users\User\Downloads\Update of Project before EID\UNSW-NB15_2.csv",
    r"C:\Users\User\Downloads\Update of Project before EID\UNSW-NB15_3.csv",
    r"C:\Users\User\Downloads\Update of Project before EID\UNSW-NB15_4.csv",
]
TONIOT_PATH    = r"C:\Users\User\Downloads\Update of Project before EID\Train_Test_Network.csv"


# ================================================================
# STEP 1 — KDD'99
# ================================================================
print("=" * 60)
print("STEP 1 — KDD'99")
print("=" * 60)

kdd_cols = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files','num_outbound_cmds',
    'is_host_login','is_guest_login','count','srv_count','serror_rate',
    'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
    'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label'
]
df_kdd = pd.read_csv(KDD99_PATH, header=0)
df_kdd.columns = kdd_cols
df_kdd['label'] = df_kdd['label'].str.rstrip('.')
print(f"  Loaded : {df_kdd.shape}")

before = len(df_kdd)
df_kdd.drop_duplicates(inplace=True)
print(f"  Duplicates removed : {before - len(df_kdd):,}  ({len(df_kdd):,} remain)")

attack_map_kdd = {
    'normal':'normal',
    'back':'dos','land':'dos','neptune':'dos','pod':'dos',
    'smurf':'dos','teardrop':'dos',
    'ipsweep':'probe','nmap':'probe','portsweep':'probe','satan':'probe',
    'ftp_write':'r2l','guess_passwd':'r2l','imap':'r2l','multihop':'r2l',
    'phf':'r2l','spy':'r2l','warezclient':'r2l','warezmaster':'r2l',
    'buffer_overflow':'u2r','loadmodule':'u2r','perl':'u2r','rootkit':'u2r',
}
df_kdd['label_cat']    = df_kdd['label'].map(attack_map_kdd)
df_kdd['binary_label'] = (df_kdd['label'] != 'normal').astype(int)

df_kdd = pd.get_dummies(df_kdd,
             columns=[c for c in ['protocol_type','service','flag']
                      if c in df_kdd.columns])

num_cols_kdd = [c for c in df_kdd.select_dtypes(include=[np.number]).columns
                if c != 'binary_label']
df_kdd[num_cols_kdd] = MinMaxScaler().fit_transform(df_kdd[num_cols_kdd])

X_kdd = df_kdd.drop(columns=['label','label_cat','binary_label'])
y_kdd = df_kdd['label_cat']
X_kdd_train, X_kdd_test, y_kdd_train, y_kdd_test = train_test_split(
    X_kdd, y_kdd, test_size=0.3, random_state=42, stratify=y_kdd)

print(f"  Train : {X_kdd_train.shape}  |  Test : {X_kdd_test.shape}")
print(f"  Labels:\n{y_kdd_train.value_counts().to_string()}\n")


# ================================================================
# STEP 2 — NSL-KDD
# ================================================================
print("=" * 60)
print("STEP 2 — NSL-KDD")
print("=" * 60)

nsl_cols = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files','num_outbound_cmds',
    'is_host_login','is_guest_login','count','srv_count','serror_rate',
    'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
    'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]
df_nsl_train = pd.read_csv(NSL_TRAIN_PATH, header=None, names=nsl_cols)
df_nsl_test  = pd.read_csv(NSL_TEST_PATH,  header=None, names=nsl_cols)
print(f"  Loaded train : {df_nsl_train.shape} | test : {df_nsl_test.shape}")

df_nsl_train.drop(columns=['difficulty'], inplace=True)
df_nsl_test.drop(columns=['difficulty'],  inplace=True)

attack_map_nsl = {
    'normal':'normal',
    'back':'dos','land':'dos','neptune':'dos','pod':'dos','smurf':'dos',
    'teardrop':'dos','apache2':'dos','udpstorm':'dos','processtable':'dos',
    'mailbomb':'dos',
    'ipsweep':'probe','nmap':'probe','portsweep':'probe','satan':'probe',
    'saint':'probe','mscan':'probe',
    'ftp_write':'r2l','guess_passwd':'r2l','imap':'r2l','multihop':'r2l',
    'phf':'r2l','spy':'r2l','warezclient':'r2l','warezmaster':'r2l',
    'sendmail':'r2l','named':'r2l','snmpgetattack':'r2l','snmpguess':'r2l',
    'worm':'r2l','xlock':'r2l','xsnoop':'r2l','httptunnel':'r2l',
    'buffer_overflow':'u2r','loadmodule':'u2r','perl':'u2r','rootkit':'u2r',
    'ps':'u2r','sqlattack':'u2r','xterm':'u2r',
}
df_nsl_train['label_cat']    = df_nsl_train['label'].map(attack_map_nsl)
df_nsl_test['label_cat']     = df_nsl_test['label'].map(attack_map_nsl)
df_nsl_train['binary_label'] = (df_nsl_train['label'] != 'normal').astype(int)
df_nsl_test['binary_label']  = (df_nsl_test['label']  != 'normal').astype(int)

cat_cols_nsl = [c for c in ['protocol_type','service','flag']
                if c in df_nsl_train.columns]
df_nsl_train = pd.get_dummies(df_nsl_train, columns=cat_cols_nsl)
df_nsl_test  = pd.get_dummies(df_nsl_test,  columns=cat_cols_nsl)
df_nsl_train, df_nsl_test = df_nsl_train.align(
    df_nsl_test, join='left', axis=1, fill_value=0)

feat_nsl = [c for c in df_nsl_train.select_dtypes(include=[np.number]).columns
            if c != 'binary_label']
scaler_nsl = MinMaxScaler()
df_nsl_train[feat_nsl] = scaler_nsl.fit_transform(df_nsl_train[feat_nsl])
df_nsl_test[feat_nsl]  = scaler_nsl.transform(df_nsl_test[feat_nsl])

print(f"  Train : {df_nsl_train.shape}  |  Test : {df_nsl_test.shape}")
print(f"  Labels:\n{df_nsl_train['label_cat'].value_counts().to_string()}\n")


# ================================================================
# STEP 3 — UNSW-NB15
# ================================================================
print("=" * 60)
print("STEP 3 — UNSW-NB15")
print("=" * 60)

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

parts = []
for path in UNSW_PATHS:
    part = pd.read_csv(path, header=None, names=unsw_cols, low_memory=False)
    parts.append(part)
    print(f"  Loaded {path.split(chr(92))[-1]} — {part.shape[0]:,} rows")

df_unsw = pd.concat(parts, ignore_index=True)
print(f"  Combined : {df_unsw.shape}")

# Drop garbage rows where sbytes is not numeric
df_unsw = df_unsw[pd.to_numeric(df_unsw['sbytes'], errors='coerce').notna()]
df_unsw.reset_index(drop=True, inplace=True)
print(f"  After dropping garbage rows : {len(df_unsw):,}")

# Verify attack_cat is present
print(f"  attack_cat present : {'attack_cat' in df_unsw.columns}")
print(f"  attack_cat sample  : {df_unsw['attack_cat'].unique()[:10]}")

# Fix Backdoor/Backdoors typo
df_unsw['attack_cat'] = df_unsw['attack_cat'].astype(str).str.strip().str.lower()
df_unsw['attack_cat'] = df_unsw['attack_cat'].replace('backdoors', 'backdoor')
df_unsw['binary_label'] = pd.to_numeric(
    df_unsw['Label'], errors='coerce').fillna(0).astype(int)

# Handle nulls
null_pct = df_unsw.isnull().mean() * 100
high_null = null_pct[null_pct > 50].index.tolist()
print(f"  Dropping high-null cols (>50%) : {high_null}")
df_unsw.drop(columns=high_null, inplace=True)

for col in df_unsw.columns:
    if df_unsw[col].dtype == 'object':
        df_unsw[col].fillna('unknown', inplace=True)
    else:
        df_unsw[col].fillna(df_unsw[col].median(), inplace=True)
print(f"  Nulls remaining : {df_unsw.isnull().sum().sum()}")

# Drop IPs, ports, timestamps
drop_unsw = ['srcip','dstip','sport','dsport','Stime','Ltime','Label']
df_unsw.drop(columns=[c for c in drop_unsw if c in df_unsw.columns], inplace=True)

# ── KEY FIX: clean ct_ftp_cmd before encoding to prevent duplicate columns ──
if 'ct_ftp_cmd' in df_unsw.columns:
    df_unsw['ct_ftp_cmd'] = df_unsw['ct_ftp_cmd'].astype(str).str.strip()

# Encode categoricals — protect attack_cat and binary_label
protected = ['attack_cat', 'binary_label']
cat_unsw = [c for c in df_unsw.select_dtypes(include='object').columns
            if c not in protected]
print(f"  Encoding : {cat_unsw}")
df_unsw = pd.get_dummies(df_unsw, columns=cat_unsw)

# Remove any remaining duplicate columns
before_dedup = df_unsw.shape[1]
df_unsw = df_unsw.loc[:, ~df_unsw.columns.duplicated(keep='first')]
print(f"  Duplicate columns removed  : {before_dedup - df_unsw.shape[1]}")
print(f"  attack_cat present after encoding : {'attack_cat' in df_unsw.columns}")

# Temporal splits
n = len(df_unsw)
df_unsw_T0 = df_unsw.iloc[:int(n * 0.60)].copy()
df_unsw_T1 = df_unsw.iloc[int(n * 0.60):int(n * 0.80)].copy()
df_unsw_T2 = df_unsw.iloc[int(n * 0.80):].copy()

# Normalize — fit on T0 only, apply to T1 and T2
feat_unsw = [c for c in df_unsw_T0.select_dtypes(include=[np.number]).columns
             if c != 'binary_label']
scaler_unsw = MinMaxScaler()
df_unsw_T0[feat_unsw] = scaler_unsw.fit_transform(df_unsw_T0[feat_unsw])
df_unsw_T1[feat_unsw] = scaler_unsw.transform(df_unsw_T1[feat_unsw])
df_unsw_T2[feat_unsw] = scaler_unsw.transform(df_unsw_T2[feat_unsw])

print(f"  T0 : {df_unsw_T0.shape} | T1 : {df_unsw_T1.shape} | T2 : {df_unsw_T2.shape}")
print(f"  Attack categories:\n{df_unsw_T0['attack_cat'].value_counts().to_string()}\n")


# ================================================================
# STEP 4 — ToN-IoT
# ================================================================
print("=" * 60)
print("STEP 4 — ToN-IoT")
print("=" * 60)

df_ton = pd.read_csv(TONIOT_PATH, low_memory=False)
print(f"  Loaded : {df_ton.shape}")

drop_ton = ['src_ip','dst_ip','dns_query','ssl_subject','ssl_issuer',
            'http_uri','http_user_agent','http_orig_mime_types',
            'http_resp_mime_types','weird_name','weird_addl',
            'ssl_cipher','conn_state']
df_ton.drop(columns=[c for c in drop_ton if c in df_ton.columns], inplace=True)

for col in df_ton.columns:
    if df_ton[col].dtype == 'object':
        df_ton[col].fillna('-', inplace=True)
    else:
        df_ton[col].fillna(0, inplace=True)

cat_ton = [c for c in df_ton.select_dtypes(include='object').columns
           if c not in ['label', 'type']]
df_ton = pd.get_dummies(df_ton, columns=cat_ton)

# Remove duplicate columns just in case
df_ton = df_ton.loc[:, ~df_ton.columns.duplicated(keep='first')]

df_ton['binary_label'] = df_ton['label'].astype(int)

X_ton = df_ton.drop(columns=['label', 'type', 'binary_label'])
y_ton = df_ton['type']
X_ton_train, X_ton_test, y_ton_train, y_ton_test = train_test_split(
    X_ton, y_ton, test_size=0.3, random_state=42, stratify=y_ton)

scaler_ton = MinMaxScaler()
X_ton_train = pd.DataFrame(
    scaler_ton.fit_transform(X_ton_train), columns=X_ton_train.columns)
X_ton_test = pd.DataFrame(
    scaler_ton.transform(X_ton_test), columns=X_ton_test.columns)

print(f"  Train : {X_ton_train.shape}  |  Test : {X_ton_test.shape}")
print(f"  Labels:\n{y_ton_train.value_counts().to_string()}\n")


# ================================================================
# STEP 5 — SAVE
# ================================================================
print("=" * 60)
print("STEP 5 — SAVING")
print("=" * 60)

X_kdd_train.to_parquet('kdd99_train.parquet')
X_kdd_test.to_parquet('kdd99_test.parquet')
y_kdd_train.to_frame().to_parquet('kdd99_train_labels.parquet')
y_kdd_test.to_frame().to_parquet('kdd99_test_labels.parquet')

df_nsl_train.to_parquet('nsl_train.parquet')
df_nsl_test.to_parquet('nsl_test.parquet')

df_unsw_T0.to_parquet('unsw_T0.parquet')
df_unsw_T1.to_parquet('unsw_T1.parquet')
df_unsw_T2.to_parquet('unsw_T2.parquet')

X_ton_train.assign(label=y_ton_train.values).to_parquet('ton_train.parquet')
X_ton_test.assign(label=y_ton_test.values).to_parquet('ton_test.parquet')

print("  kdd99_train.parquet   ✓")
print("  kdd99_test.parquet    ✓")
print("  nsl_train.parquet     ✓")
print("  nsl_test.parquet      ✓")
print("  unsw_T0.parquet       ✓")
print("  unsw_T1.parquet       ✓")
print("  unsw_T2.parquet       ✓")
print("  ton_train.parquet     ✓")
print("  ton_test.parquet      ✓")
print("\n  ALL DONE — paste output for verification")
