import subprocess
import os
from multiprocessing import Pool, cpu_count

t1w_dir = "$PATH/DHCP/pair/t1w"
t2w_dir = "$PATH/DHCP/pair/t2w"
out_dir = "$PATH/DHCP/pair/t2w_reg2t1w"

os.makedirs(out_dir, exist_ok=True)


def register(t2w_file):
    subject_id = t2w_file.replace("_T2w.nii.gz", "")
    t1w_file = subject_id + "_T1w.nii.gz"

    moving = os.path.join(t2w_dir, t2w_file)
    fixed = os.path.join(t1w_dir, t1w_file)
    out = os.path.join(out_dir, subject_id + "_T2w_reg2T1w.nii.gz")
    omat = os.path.join(out_dir, subject_id + "_T2w_reg2T1w.mat")

    if not os.path.exists(fixed):
        return f"SKIP {subject_id}: matching T1w not found"

    cmd = [
        "flirt",
        "-in", moving,
        "-ref", fixed,
        "-out", out,
        "-omat", omat,
        "-dof", "6"
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return f"DONE {subject_id}"
    except subprocess.CalledProcessError as e:
        return f"ERROR {subject_id}: {e}"


if __name__ == "__main__":
    t2w_files = sorted([f for f in os.listdir(t2w_dir) if f.endswith(".nii.gz")])

    n_workers = cpu_count()  # 48
    print(f"Total: {len(t2w_files)} pairs | Workers: {n_workers}")

    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(register, t2w_files):
            print(result)

    print("All done.")
