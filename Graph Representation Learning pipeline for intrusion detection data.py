import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Check if node2vec is installed
try:
    from node2vec import Node2Vec
    import networkx as nx
    print("node2vec and networkx ready")
except ImportError:
    import subprocess, sys
    print("Installing node2vec...")
    subprocess.check_call([sys.executable, "-m", "pip",
                           "install", "node2vec", "-q"])
    from node2vec import Node2Vec
    import networkx as nx
    print("Installed and imported successfully")

print("=" * 60)
print("PHASE 2 — Graph View (Node2Vec Embeddings)")
print("=" * 60)

# ================================================================
# CONCEPT
# Build a bipartite host-service graph per dataset:
#   Nodes  : source IPs  +  services
#   Edges  : src_ip → service  (weighted by flow count)
# Node2Vec walks the graph and produces a 32-dim embedding
# per node. Each flow's embedding = concat(src_ip_emb, svc_emb)
# giving a 64-dim graph-based feature vector per flow.
# ================================================================

EMB_DIM    = 32    # embedding dimensions per node
WALK_LEN   = 10   # random walk length
NUM_WALKS  = 20   # walks per node
WORKERS    = 2    # parallel workers


def build_graph_embeddings(src_ips, services, dataset_name,
                            emb_dim=32, walk_len=10,
                            num_walks=20, workers=2):
    """
    Build bipartite host-service graph and compute Node2Vec embeddings.
    Returns a (n_flows, emb_dim*2) numpy array.
    """
    print(f"  Building graph for {dataset_name}...")
    print(f"    Unique src IPs  : {src_ips.nunique():,}")
    print(f"    Unique services : {services.nunique():,}")

    # Build edge list with weights
    edges = pd.DataFrame({
        'src': 'ip_' + src_ips.astype(str),
        'svc': 'svc_' + services.astype(str)
    })
    edge_weights = (edges.groupby(['src','svc'])
                         .size()
                         .reset_index(name='weight'))

    # Build networkx graph
    G = nx.Graph()
    for _, row in edge_weights.iterrows():
        G.add_edge(row['src'], row['svc'], weight=row['weight'])

    print(f"    Graph nodes : {G.number_of_nodes():,}  "
          f"edges : {G.number_of_edges():,}")

    # Node2Vec
    n2v = Node2Vec(G, dimensions=emb_dim, walk_length=walk_len,
                   num_walks=num_walks, workers=workers, quiet=True)
    model = n2v.fit(window=5, min_count=1, batch_words=4)

    # Map each flow to its embedding
    src_col  = 'ip_' + src_ips.astype(str)
    svc_col  = 'svc_' + services.astype(str)

    def get_emb(node):
        try:
            return model.wv[node]
        except KeyError:
            return np.zeros(emb_dim, dtype=np.float32)

    src_embs = np.array([get_emb(n) for n in src_col], dtype=np.float32)
    svc_embs = np.array([get_emb(n) for n in svc_col], dtype=np.float32)

    # Concatenate src + service embeddings → 64-dim per flow
    flow_embs = np.concatenate([src_embs, svc_embs], axis=1)
    print(f"    Flow embeddings shape : {flow_embs.shape}\n")
    return flow_embs


# ================================================================
# NSL-KDD — use protocol_type as proxy for service
# (no IP columns — use index-based src proxy)
# ================================================================
print("─" * 60)
print("NSL-KDD")
print("─" * 60)

df_nsl_tr = pd.read_parquet("nsl_train.parquet")
df_nsl_te = pd.read_parquet("nsl_test.parquet")

# NSL-KDD has no IP — use row-index bucketed into 100 groups as
# src proxy and service column for service node
src_proxy_tr = (df_nsl_tr.index % 100).astype(str)
src_proxy_te = (df_nsl_te.index % 100).astype(str)

# Find service column (one-hot — reverse to get service name)
svc_cols_tr = [c for c in df_nsl_tr.columns if c.startswith('service_')]
if svc_cols_tr:
    svc_tr = df_nsl_tr[svc_cols_tr].idxmax(axis=1).str.replace('service_','')
    svc_te = df_nsl_te[svc_cols_tr].idxmax(axis=1).str.replace('service_','')
else:
    svc_tr = pd.Series(['unknown'] * len(df_nsl_tr))
    svc_te = pd.Series(['unknown'] * len(df_nsl_te))

emb_nsl_tr = build_graph_embeddings(
    src_proxy_tr, svc_tr, "NSL-KDD train",
    EMB_DIM, WALK_LEN, NUM_WALKS, WORKERS)
emb_nsl_te = build_graph_embeddings(
    src_proxy_te, svc_te, "NSL-KDD test",
    EMB_DIM, WALK_LEN, NUM_WALKS, WORKERS)

np.save("graph_nsl_X_train.npy", emb_nsl_tr)
np.save("graph_nsl_X_test.npy",  emb_nsl_te)
np.save("graph_nsl_y_train.npy", df_nsl_tr["binary_label"].values)
np.save("graph_nsl_y_test.npy",  df_nsl_te["binary_label"].values)
print("  Saved graph_nsl_X/y_train/test.npy  ✅\n")


# ================================================================
# UNSW-NB15 — has real srcip and service columns
# ================================================================
print("─" * 60)
print("UNSW-NB15")
print("─" * 60)

# Reload raw to get srcip and service before they were dropped
unsw_cols_needed = ['srcip','service','sbytes','Label']
unsw_cols_all = [
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
    part = pd.read_csv(path, header=None, names=unsw_cols_all,
                       usecols=['srcip','service','Label'],
                       low_memory=False)
    parts.append(part)
df_unsw_graph = pd.concat(parts, ignore_index=True)
df_unsw_graph = df_unsw_graph[
    df_unsw_graph['Label'].astype(str).str.strip()
    .str.replace('.','',regex=False).str.isnumeric()]
df_unsw_graph['service'] = (df_unsw_graph['service']
                              .astype(str).str.strip()
                              .replace('-','other'))
df_unsw_graph['Label'] = pd.to_numeric(
    df_unsw_graph['Label'], errors='coerce').fillna(0).astype(int)
df_unsw_graph.reset_index(drop=True, inplace=True)

# Use T0/T2 index splits
n = len(df_unsw_graph)
t0_end = int(n * 0.60)
t2_start = int(n * 0.80)

# Sample for speed
idx_tr = np.random.RandomState(42).choice(t0_end,
                                           size=60_000, replace=False)
idx_te = np.random.RandomState(42).choice(
    np.arange(t2_start, n), size=20_000, replace=False)

src_unsw_tr  = df_unsw_graph['srcip'].iloc[idx_tr]
svc_unsw_tr  = df_unsw_graph['service'].iloc[idx_tr]
src_unsw_te  = df_unsw_graph['srcip'].iloc[idx_te]
svc_unsw_te  = df_unsw_graph['service'].iloc[idx_te]
y_unsw_gr_tr = df_unsw_graph['Label'].iloc[idx_tr].values
y_unsw_gr_te = df_unsw_graph['Label'].iloc[idx_te].values

emb_unsw_tr = build_graph_embeddings(
    src_unsw_tr, svc_unsw_tr, "UNSW-NB15 T0 (sampled)",
    EMB_DIM, WALK_LEN, NUM_WALKS, WORKERS)
emb_unsw_te = build_graph_embeddings(
    src_unsw_te, svc_unsw_te, "UNSW-NB15 T2 (sampled)",
    EMB_DIM, WALK_LEN, NUM_WALKS, WORKERS)

np.save("graph_unsw_X_train.npy", emb_unsw_tr)
np.save("graph_unsw_X_test.npy",  emb_unsw_te)
np.save("graph_unsw_y_train.npy", y_unsw_gr_tr)
np.save("graph_unsw_y_test.npy",  y_unsw_gr_te)
print("  Saved graph_unsw_X/y_train/test.npy  ✅\n")


# ================================================================
# ToN-IoT — has src_ip and service columns
# ================================================================
print("─" * 60)
print("ToN-IoT")
print("─" * 60)

df_ton_raw = pd.read_csv(r"train_test_network.csv",
                          usecols=['src_ip','service','label'],
                          low_memory=False)
df_ton_raw['service'] = (df_ton_raw['service']
                          .astype(str).str.strip()
                          .replace('-','other'))
df_ton_raw['binary_label'] = (
    df_ton_raw['label'].astype(str).str.lower() != 'normal').astype(int)

from sklearn.model_selection import train_test_split
tr_idx, te_idx = train_test_split(
    df_ton_raw.index, test_size=0.3,
    random_state=42,
    stratify=df_ton_raw['binary_label'])

emb_ton_tr = build_graph_embeddings(
    df_ton_raw['src_ip'].iloc[tr_idx],
    df_ton_raw['service'].iloc[tr_idx],
    "ToN-IoT train",
    EMB_DIM, WALK_LEN, NUM_WALKS, WORKERS)
emb_ton_te = build_graph_embeddings(
    df_ton_raw['src_ip'].iloc[te_idx],
    df_ton_raw['service'].iloc[te_idx],
    "ToN-IoT test",
    EMB_DIM, WALK_LEN, NUM_WALKS, WORKERS)

np.save("graph_ton_X_train.npy", emb_ton_tr)
np.save("graph_ton_X_test.npy",  emb_ton_te)
np.save("graph_ton_y_train.npy",
        df_ton_raw['binary_label'].iloc[tr_idx].values)
np.save("graph_ton_y_test.npy",
        df_ton_raw['binary_label'].iloc[te_idx].values)
print("  Saved graph_ton_X/y_train/test.npy  ✅\n")


# ================================================================
# VERIFY all graph files
# ================================================================
import os
print("=" * 60)
print("GRAPH VIEW FILE INVENTORY")
print("=" * 60)
graph_files = [
    "graph_nsl_X_train.npy",  "graph_nsl_y_train.npy",
    "graph_nsl_X_test.npy",   "graph_nsl_y_test.npy",
    "graph_unsw_X_train.npy", "graph_unsw_y_train.npy",
    "graph_unsw_X_test.npy",  "graph_unsw_y_test.npy",
    "graph_ton_X_train.npy",  "graph_ton_y_train.npy",
    "graph_ton_X_test.npy",   "graph_ton_y_test.npy",
]
for f in graph_files:
    if os.path.exists(f):
        arr = np.load(f)
        print(f"  ✅  {f:35s}  shape: {arr.shape}")
    else:
        print(f"  ❌  {f:35s}  MISSING")

print(f"\n  Embedding dim per node : {EMB_DIM}")
print(f"  Flow embedding dim     : {EMB_DIM * 2} "
      f"(src_emb + service_emb concatenated)")
print("\n  GRAPH VIEW COMPLETE — paste output for verification")
