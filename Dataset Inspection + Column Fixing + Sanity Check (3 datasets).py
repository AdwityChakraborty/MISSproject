import pandas as pd

# ================================================================
# FIX 1 — KDD'99: load full file + rename broken column names
# ================================================================

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

df_kdd = pd.read_csv(r"kddcup99_csv.csv", header=0)  # has header already
df_kdd.columns = kdd_cols   # force correct names, dropping the broken 'l' prefix

# Remove trailing dot from labels if present
df_kdd['label'] = df_kdd['label'].str.rstrip('.')

print("KDD'99 fixed:")
print(f"  Shape  : {df_kdd.shape}")
print(f"  Labels : {df_kdd['label'].value_counts().to_string()}\n")


# ================================================================
# FIX 2 — UNSW-NB15: assign correct 49 column names
# ================================================================

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
    part = pd.read_csv(path, header=None, names=unsw_cols, low_memory=False)
    parts.append(part)
    print(f"  Loaded {path.split(chr(92))[-1]} — {part.shape[0]:,} rows")

df_unsw = pd.concat(parts, ignore_index=True)

# Clean up label columns
df_unsw['attack_cat'] = df_unsw['attack_cat'].str.strip()
df_unsw['Label'] = pd.to_numeric(df_unsw['Label'], errors='coerce').fillna(0).astype(int)

print(f"\nUNSW-NB15 fixed:")
print(f"  Shape       : {df_unsw.shape}")
print(f"  Null values : {df_unsw.isnull().sum().sum():,}")
print(f"  Attack cats : \n{df_unsw['attack_cat'].value_counts().to_string()}")
print(f"  Label (0/1) : \n{df_unsw['Label'].value_counts().to_string()}\n")


# ================================================================
# FIX 3 — ToN-IoT: confirm 'type' column for multi-class labels
# ================================================================

df_ton = pd.read_csv(r"train_test_network.csv", low_memory=False)

print("ToN-IoT confirmed:")
print(f"  Shape      : {df_ton.shape}")
print(f"  Attack types (use 'type' column):")
print(f"{df_ton['type'].value_counts().to_string()}\n")
