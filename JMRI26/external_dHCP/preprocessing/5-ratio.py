import nibabel as nib
import numpy as np
import os
from multiprocessing import Pool, cpu_count

base_dir = "$PATH/DHCP/pair"

t1w_in_dir = os.path.join(base_dir, "t1w_template")
t2w_in_dir = os.path.join(base_dir, "t2w_template")
out_dir = os.path.join(base_dir, "ratio")

os.makedirs(out_dir, exist_ok=True)


def compute_ratio(args):
    subject_id, t1w_path, t2w_path = args
    try:
        t1_img = nib.load(t1w_path)
        t2_img = nib.load(t2w_path)
        t1_data = t1_img.get_fdata()
        t2_data = t2_img.get_fdata()

        valid_mask = (t1_data != 0) & (t2_data != 0)
        ratio = t1_data[valid_mask] / t2_data[valid_mask]

        # Raw ratio
        result = np.zeros_like(t1_data)
        result[valid_mask] = ratio
        nib.save(
            nib.Nifti1Image(result, affine=t1_img.affine),
            os.path.join(out_dir, subject_id + "_ratio.nii.gz"),
        )

        clip_90 = np.percentile(ratio, 90)
        result_90 = np.zeros_like(t1_data)
        result_90[valid_mask] = np.clip(ratio, None, clip_90)
        nib.save(
            nib.Nifti1Image(result_90, affine=t1_img.affine),
            os.path.join(out_dir, subject_id + "_ratio_clipped_90.nii.gz"),
        )

        clip_95 = np.percentile(ratio, 95)
        result_95 = np.zeros_like(t1_data)
        result_95[valid_mask] = np.clip(ratio, None, clip_95)
        nib.save(
            nib.Nifti1Image(result_95, affine=t1_img.affine),
            os.path.join(out_dir, subject_id + "_ratio_clipped_95.nii.gz"),
        )

        return f"DONE {subject_id} (clip90={clip_90:.4f}, clip95={clip_95:.4f})"
    except Exception as e:
        return f"ERROR {subject_id}: {e}"


if __name__ == "__main__":
    tasks = []

    # Match T1w and T2w by subject_id
    t1w_files = {
        f.replace("_T1w_template.nii.gz", ""): f
        for f in os.listdir(t1w_in_dir)
        if f.endswith(".nii.gz")
    }
    t2w_files = {
        f.replace("_T2w_template.nii.gz", ""): f
        for f in os.listdir(t2w_in_dir)
        if f.endswith(".nii.gz")
    }

    common_ids = sorted(set(t1w_files.keys()) & set(t2w_files.keys()))

    for subject_id in common_ids:
        t1w_path = os.path.join(t1w_in_dir, t1w_files[subject_id])
        t2w_path = os.path.join(t2w_in_dir, t2w_files[subject_id])
        tasks.append((subject_id, t1w_path, t2w_path))

    n_workers = cpu_count()
    print(f"Total: {len(tasks)} pairs | Workers: {n_workers}")

    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(compute_ratio, tasks):
            print(result)

    print("All done.")
