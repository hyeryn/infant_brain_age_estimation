import os
import shutil
import pandas as pd

csv_path = "t1t2_target.csv"
src_dir = "$PATH/external_cohort/dataset_nifty"
dst_dir = "$PATH/external_cohort/dataset_T1T2"

df = pd.read_csv(csv_path)

os.makedirs(dst_dir, exist_ok=True)

for folder, group in df.groupby("Folder"):
    t1_row = group[group["SeriesDescription"].str.contains("T1", case=False, na=False)]
    t2_row = group[group["SeriesDescription"].str.contains("T2", case=False, na=False)]

    if t1_row.empty and t2_row.empty:
        print(f"{folder}: no T1/T2 series, skipping")
        continue

    folder_out = os.path.join(dst_dir, str(folder))
    os.makedirs(folder_out, exist_ok=True)

    if not t1_row.empty:
        t1_idx = t1_row.index[0]
        t1_series_num = df.loc[t1_idx, df.columns[-1]]
        t1_src = os.path.join(src_dir, f"{folder}_series{t1_series_num}.nii.gz")
        t1_dst = os.path.join(folder_out, "t1.nii.gz")
        if os.path.exists(t1_src):
            shutil.copy2(t1_src, t1_dst)
            print(f"✅ {folder}: T1 → {t1_dst}")
        else:
            print(f"{folder}: T1 file not found → {t1_src}")

    if not t2_row.empty:
        t2_idx = t2_row.index[0]
        t2_series_num = df.loc[t2_idx, df.columns[-1]]
        t2_src = os.path.join(src_dir, f"{folder}_series{t2_series_num}.nii.gz")
        t2_dst = os.path.join(folder_out, "t2.nii.gz")
        if os.path.exists(t2_src):
            shutil.copy2(t2_src, t2_dst)
            print(f"✅ {folder}: T2 → {t2_dst}")
        else:
            print(f"{folder}: T2 file not found → {t2_src}")

print("Copy finished.")
