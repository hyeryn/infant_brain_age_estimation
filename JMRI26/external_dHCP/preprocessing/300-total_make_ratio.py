import subprocess
import os
import shutil
import tempfile
import time
from multiprocessing import Pool, cpu_count
import concurrent.futures

import nibabel as nib
import numpy as np

os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "4"
import ants

# ============================================================
# Paths
# ============================================================
BASE_DIRS = [
    "$PATH/DHCP/pair2",
    "$PATH/DHCP/pair1",
]
atlas_path = "$PATH/datasets/infancy_brain_MRI/atlas/t1_reference_subject_15mm.nii"

N_WORKERS = cpu_count() // 2  # 24
N_WORKERS_ANTS = 6             # 4 threads x 6 = 24


def run_step(name, func, args, n_workers):
    if not args:
        print(f"\n>>> Step {name} — nothing to do (all done), skipping.")
        return
    print(f"\n{'='*60}")
    print(f"  Step {name}")
    print(f"  Workers: {n_workers}  |  Tasks: {len(args)}")
    print(f"{'='*60}")
    start = time.time()

    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(func, args):
            print(result)

    elapsed = time.time() - start
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    print(f"\n>>> Step {name} finished  |  {h:02d}h {m:02d}m {s:02d}s")
    print(f"{'='*60}\n")


def run_step_thread(name, func, args, n_workers):
    """Thread-based runner for ANTs (avoids fork issues)."""
    if not args:
        print(f"\n>>> Step {name} — nothing to do (all done), skipping.")
        return
    print(f"\n{'='*60}")
    print(f"  Step {name}")
    print(f"  Workers: {n_workers}  |  Tasks: {len(args)}")
    print(f"{'='*60}")
    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(func, a): a for a in args}
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

    elapsed = time.time() - start
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    print(f"\n>>> Step {name} finished  |  {h:02d}h {m:02d}m {s:02d}s")
    print(f"{'='*60}\n")


# ============================================================
# ============================================================
#
#


# ============================================================
# Step 1 - Brain Extraction (BET on T1w, apply mask to T2w)
# ============================================================
#


# ============================================================
# ============================================================


# ============================================================
# Step 3 - ANTs SyN Registration to Atlas (T1w, apply to T2w)
#   Replaces old Step 3 (FLIRT) + Step 4 (ANTs) in one step.
#   ANTs SyN does affine + nonlinear internally.
# ============================================================
def register_to_atlas_ants(args):
    base_dir, subject_id = args
    t1w_in = os.path.join(base_dir, "t1w_biascorr", subject_id + "_T1w_biascorr.nii.gz")
    t2w_in = os.path.join(base_dir, "t2w_biascorr", subject_id + "_T2w_biascorr.nii.gz")
    t1w_out = os.path.join(base_dir, "t1w_reg2atlas", subject_id + "_T1w_reg2atlas.nii.gz")
    t2w_out = os.path.join(base_dir, "t2w_reg2atlas", subject_id + "_T2w_reg2atlas.nii.gz")

    mask_path = os.path.join(base_dir, "t1w_brain", subject_id + "_T1w_brain_mask.nii.gz")

    if os.path.exists(t1w_out) and os.path.exists(t2w_out):
        return f"SKIP {subject_id}: already done"
    if not os.path.exists(t2w_in):
        return f"SKIP {subject_id}: T2w_biascorr not found"
    try:
        template = ants.image_read(atlas_path)

        t1w_moving = ants.image_read(t1w_in)
        moving_mask = ants.image_read(mask_path) if os.path.exists(mask_path) else None
        reg = ants.registration(
            fixed=template,
            moving=t1w_moving,
            type_of_transform="Affine",
        )
        ants.image_write(reg["warpedmovout"], t1w_out)

        # Apply same transform to T2w
        t2w_moving = ants.image_read(t2w_in)
        t2w_warped = ants.apply_transforms(
            fixed=template,
            moving=t2w_moving,
            transformlist=reg["fwdtransforms"],
        )
        ants.image_write(t2w_warped, t2w_out)

        return f"DONE {subject_id}"
    except Exception as e:
        return f"ERROR {subject_id}: {e}"


# ============================================================
#   Old Step 4 did ANTs nonlinear on top of FLIRT result.
#   Now Step 3 does ANTs SyN directly, so this is no longer needed.
# ============================================================
#
#


# ============================================================
# ============================================================
def compute_ratio(args):
    subject_id, t1w_path, t2w_path, out_dir = args
    out_file = os.path.join(out_dir, subject_id + "_ratio.nii.gz")
    if os.path.exists(out_file):
        return f"SKIP {subject_id}: already done"
    try:
        t1_img = nib.load(t1w_path)
        t2_img = nib.load(t2w_path)
        t1_data = t1_img.get_fdata()
        t2_data = t2_img.get_fdata()

        valid_mask = (t1_data != 0) & (t2_data != 0)
        ratio = t1_data[valid_mask] / t2_data[valid_mask]

        result = np.zeros_like(t1_data)
        result[valid_mask] = ratio
        nib.save(
            nib.Nifti1Image(result, affine=t1_img.affine),
            out_file,
        )


        clip_98 = np.percentile(ratio, 98)
        result_98 = np.zeros_like(t1_data)
        result_98[valid_mask] = np.clip(ratio, None, clip_98)
        nib.save(
            nib.Nifti1Image(result_98, affine=t1_img.affine),
            os.path.join(out_dir, subject_id + "_ratio_clipped_98.nii.gz"),
        )

        return f"DONE {subject_id} (clip98={clip_98:.4f})"
    except Exception as e:
        return f"ERROR {subject_id}: {e}"


# ============================================================
# Main - Process all BASE_DIRS
#   Steps 0-2 are already completed, start from Step 3.
# ============================================================
if __name__ == "__main__":
    total_start = time.time()

    for base_dir in BASE_DIRS:
        print(f"\n{'#'*60}")
        print(f"  Processing: {base_dir}")
        print(f"{'#'*60}")
        dir_start = time.time()

        subdirs = [
            "t1w_reg2atlas", "t2w_reg2atlas", "ratio",
        ]
        for sd in subdirs:
            os.makedirs(os.path.join(base_dir, sd), exist_ok=True)

        # --- Steps 0-2: already completed, skipping ---

        # --- Step 3: ANTs SyN Registration to Atlas ---
        t1bc_dir = os.path.join(base_dir, "t1w_biascorr")
        if os.path.isdir(t1bc_dir):
            tasks_3 = [
                (base_dir, f.replace("_T1w_biascorr.nii.gz", ""))
                for f in sorted(os.listdir(t1bc_dir)) if f.endswith(".nii.gz")
            ]
            run_step_thread("3 - ANTs SyN Registration to Atlas", register_to_atlas_ants, tasks_3, N_WORKERS_ANTS)

        # --- Step 5: T1w/T2w Ratio (from reg2atlas) ---
        t1r_dir = os.path.join(base_dir, "t1w_reg2atlas")
        t2r_dir = os.path.join(base_dir, "t2w_reg2atlas")
        ratio_dir = os.path.join(base_dir, "ratio")
        if os.path.isdir(t1r_dir) and os.path.isdir(t2r_dir):
            t1w_files = {
                f.replace("_T1w_reg2atlas.nii.gz", ""): f
                for f in os.listdir(t1r_dir) if f.endswith(".nii.gz")
            }
            t2w_files = {
                f.replace("_T2w_reg2atlas.nii.gz", ""): f
                for f in os.listdir(t2r_dir) if f.endswith(".nii.gz")
            }
            common_ids = sorted(set(t1w_files.keys()) & set(t2w_files.keys()))
            tasks_5 = [
                (sid, os.path.join(t1r_dir, t1w_files[sid]), os.path.join(t2r_dir, t2w_files[sid]), ratio_dir)
                for sid in common_ids
            ]
            run_step("5 - T1w/T2w Ratio", compute_ratio, tasks_5, N_WORKERS)

        dir_elapsed = time.time() - dir_start
        dm, ds = divmod(int(dir_elapsed), 60)
        dh, dm = divmod(dm, 60)
        print(f"\n>>> {base_dir} finished  |  {dh:02d}h {dm:02d}m {ds:02d}s")

    # --- Total ---
    total_elapsed = time.time() - total_start
    m, s = divmod(int(total_elapsed), 60)
    h, m = divmod(m, 60)
    print(f"\n{'#'*60}")
    print(f"  ALL DONE  |  Total: {h:02d}h {m:02d}m {s:02d}s")
    print(f"{'#'*60}")
