import os

files = [
    "H_kdd_train.parquet",  "H_kdd_test.parquet",
    "H_nsl_train.parquet",  "H_nsl_test.parquet",
    "H_unsw_T0.parquet",    "H_unsw_T1.parquet",   "H_unsw_T2.parquet",
    "H_ton_train.parquet",  "H_ton_test.parquet",
    "kdd99_train_smote.parquet",
    "nsl_train_smote.parquet",
    "unsw_T0_with_iot.parquet",
    "ton_train_with_iot.parquet",
    "unsw_iot_profile.parquet",
]

print("File inventory check:")
all_ok = True
for f in files:
    exists = os.path.exists(f)
    size   = f"{os.path.getsize(f)/1024/1024:.1f} MB" if exists else "MISSING"
    status = "✅" if exists else "❌"
    print(f"  {status}  {f:45s}  {size}")
    if not exists:
        all_ok = False

print(f"\n  {'ALL FILES PRESENT — ready for Phase 3' if all_ok else 'SOME FILES MISSING — check above'}")
