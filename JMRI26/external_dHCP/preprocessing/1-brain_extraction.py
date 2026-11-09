import subprocess
import os
from multiprocessing import Pool, cpu_count

base_dir = "$PATH/DHCP/pair"

t1w_in_dir = os.path.join(base_dir, "t1w")
t2w_in_dir = os.path.join(base_dir, "t2w_reg2t1w")
t1w_out_dir = os.path.join(base_dir, "t1w_brain")
t2w_out_dir = os.path.join(base_dir, "t2w_brain")

os.makedirs(t1w_out_dir, exist_ok=True)
os.makedirs(t2w_out_dir, exist_ok=True)


def brain_extract(subject_id):
    t1w_in = os.path.join(t1w_in_dir, subject_id + "_T1w.nii.gz")
    t2w_in = os.path.join(t2w_in_dir, subject_id + "_T2w_reg2T1w.nii.gz")
    t1w_out = os.path.join(t1w_out_dir, subject_id + "_T1w_brain")
    t2w_out = os.path.join(t2w_out_dir, subject_id + "_T2w_brain.nii.gz")

    if not os.path.exists(t2w_in):
        return f"SKIP {subject_id}: T2w_reg2T1w not found"

    try:
        # BET on T1w -> brain + mask
        subprocess.run(
            ["bet", t1w_in, t1w_out, "-m"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        # Apply T1w mask to T2w
        mask_path = t1w_out + "_mask.nii.gz"
        subprocess.run(
            ["fslmaths", t2w_in, "-mas", mask_path, t2w_out],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        return f"DONE {subject_id}"
    except subprocess.CalledProcessError as e:
        return f"ERROR {subject_id}: {e}"


if __name__ == "__main__":
    subject_ids = sorted(
        f.replace("_T1w.nii.gz", "")
        for f in os.listdir(t1w_in_dir)
        if f.endswith(".nii.gz")
    )

    n_workers = cpu_count()
    print(f"Total: {len(subject_ids)} subjects | Workers: {n_workers}")

    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(brain_extract, subject_ids):
            print(result)

    print("All done.")
