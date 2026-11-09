import subprocess
import os
import shutil
import tempfile
from multiprocessing import Pool, cpu_count

base_dir = "$PATH/DHCP/pair"

t1w_in_dir = os.path.join(base_dir, "t1w_brain")
t2w_in_dir = os.path.join(base_dir, "t2w_brain")
t1w_out_dir = os.path.join(base_dir, "t1w_biascorr")
t2w_out_dir = os.path.join(base_dir, "t2w_biascorr")

os.makedirs(t1w_out_dir, exist_ok=True)
os.makedirs(t2w_out_dir, exist_ok=True)


def bias_correct(args):
    input_path, output_path = args
    basename = os.path.basename(output_path)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "fast")
            cmd = ["fast", "-B", "-o", prefix, input_path]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            restore = prefix + "_restore.nii.gz"
            if os.path.exists(restore):
                shutil.copy2(restore, output_path)
            else:
                return f"ERROR {basename}: _restore.nii.gz not created"
        return f"DONE {basename}"
    except subprocess.CalledProcessError as e:
        return f"ERROR {basename}: {e}"


if __name__ == "__main__":
    tasks = []

    # T1w bias correction
    for f in sorted(os.listdir(t1w_in_dir)):
        if not f.endswith(".nii.gz") or "_mask" in f:
            continue
        subject_id = f.replace("_T1w_brain.nii.gz", "")
        inp = os.path.join(t1w_in_dir, f)
        out = os.path.join(t1w_out_dir, subject_id + "_T1w_biascorr.nii.gz")
        tasks.append((inp, out))

    # T2w bias correction
    for f in sorted(os.listdir(t2w_in_dir)):
        if not f.endswith(".nii.gz"):
            continue
        subject_id = f.replace("_T2w_brain.nii.gz", "")
        inp = os.path.join(t2w_in_dir, f)
        out = os.path.join(t2w_out_dir, subject_id + "_T2w_biascorr.nii.gz")
        tasks.append((inp, out))

    n_workers = cpu_count()
    print(f"Total: {len(tasks)} images | Workers: {n_workers}")

    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(bias_correct, tasks):
            print(result)

    print("All done.")
