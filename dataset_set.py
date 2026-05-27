import pandas as pd
import os

# ── Change these paths to wherever your files are saved ──────────────────────
KDD99_PATH     = r"C:\Users\User\Downloads\Update of Project before EID\kddcup99_csv.csv"
NSL_TRAIN_PATH = r"C:\Users\User\Downloads\Update of Project before EID\KDDTrain+.txt"
NSL_TEST_PATH  = r"C:\Users\User\Downloads\Update of Project before EID\KDDTest+.txt"
UNSW_PATHS     = [r"C:\Users\User\Downloads\Update of Project before EID\UNSW-NB15_1.csv", 
                  r"C:\Users\User\Downloads\Update of Project before EID\UNSW-NB15_2.csv", 
                  r"C:\Users\User\Downloads\Update of Project before EID\UNSW-NB15_3.csv", 
                  r"C:\Users\User\Downloads\Update of Project before EID\UNSW-NB15_4.csv"]
TONIOT_PATH    = r"C:\Users\User\Downloads\Update of Project before EID\Train_Test_Network.csv"
# ─────────────────────────────────────────────────────────────────────────────

def check_dataset(name, df):
    print("=" * 60)
    print(f"  DATASET : {name}")
    print("=" * 60)
    print(f"  Shape        : {df.shape[0]:,} rows  x  {df.shape[1]} columns")
    print(f"  Columns      : {df.columns.tolist()}")
    print(f"  Dtypes       :\n{df.dtypes.value_counts().to_string()}")
    print(f"  Null values  : {df.isnull().sum().sum():,} total")
    
    # Try to find label column
    possible_labels = ['label', 'Label', 'class', 'Class', 'attack', 
                       'attack_cat', 'type', 'Type']
    found = [c for c in df.columns if c in possible_labels]
    if found:
        col = found[0]
        print(f"\n  Label column : '{col}'")
        print(f"  Unique labels:\n{df[col].value_counts().to_string()}")
    else:
        print(f"\n  Label column : NOT FOUND — check column names above")
    
    print(f"\n  First 2 rows :\n{df.head(2).to_string()}")
    print()


# ── 1. KDD'99 ─────────────────────────────────────────────────────────────────
try:
    df_kdd = pd.read_csv(KDD99_PATH, nrows=5000)
    check_dataset("KDD'99 (Kaggle CSV)", df_kdd)
except Exception as e:
    print(f"[ERROR] KDD99 failed to load: {e}\n")


# ── 2. NSL-KDD Train ─────────────────────────────────────────────────────────
try:
    # NSL-KDD .txt files have no header — 41 features + label + difficulty
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
        'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
        'dst_host_srv_rerror_rate','label','difficulty'
    ]
    df_nsl_train = pd.read_csv(NSL_TRAIN_PATH, header=None, names=nsl_cols)
    check_dataset("NSL-KDD Train (KDDTrain+.txt)", df_nsl_train)
except Exception as e:
    print(f"[ERROR] NSL-KDD Train failed to load: {e}\n")


# ── 3. NSL-KDD Test ──────────────────────────────────────────────────────────
try:
    df_nsl_test = pd.read_csv(NSL_TEST_PATH, header=None, names=nsl_cols)
    check_dataset("NSL-KDD Test (KDDTest+.txt)", df_nsl_test)
except Exception as e:
    print(f"[ERROR] NSL-KDD Test failed to load: {e}\n")


# ── 4. UNSW-NB15 (merge all 4 CSVs) ─────────────────────────────────────────
try:
    parts = []
    for path in UNSW_PATHS:
        if os.path.exists(path):
            parts.append(pd.read_csv(path, low_memory=False))
            print(f"  [OK] Loaded {path} — {parts[-1].shape[0]:,} rows")
        else:
            print(f"  [MISSING] {path} not found")
    
    if parts:
        df_unsw = pd.concat(parts, ignore_index=True)
        check_dataset("UNSW-NB15 (all 4 files merged)", df_unsw)
    else:
        print("[ERROR] No UNSW-NB15 files loaded\n")
except Exception as e:
    print(f"[ERROR] UNSW-NB15 failed to load: {e}\n")


# ── 5. ToN-IoT ───────────────────────────────────────────────────────────────
try:
    df_ton = pd.read_csv(TONIOT_PATH, low_memory=False)
    check_dataset("ToN-IoT (train_test_network.csv)", df_ton)
except Exception as e:
    print(f"[ERROR] ToN-IoT failed to load: {e}\n")
