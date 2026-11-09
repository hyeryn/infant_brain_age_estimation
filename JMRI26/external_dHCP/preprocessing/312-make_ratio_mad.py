"""
Generate adaptive ratio images using MAD-based outlier removal.

For each subject:
1. Compute raw T1w/T2w ratio
2. Calculate median and MAD (Median Absolute Deviation) of ratio
3. Remove voxels where ratio > median + 3 * 1.4826 * MAD (≈ 3σ equivalent)
4. Clip remaining at p98

This is adaptive per subject — clean subjects lose <1% of voxels,
severe outlier subjects lose more where needed.

Output: {pair}/ratio_mad/ with *_ratio_mad.nii.gz
"""
import os
from pathlib import Path
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from scipy import stats as sp_stats

DHCP_ROOT = Path("$PATH/DHCP")
PAIRS = ["pair1", "pair2"]
MAD_K = 3
MAD_SCALE = 1.4826  # scale factor to convert MAD to std equivalent


def compute_ratio_mad(args):
    subject_id, t1w_path, t2w_path, out_dir = args
    out_file = os.path.join(out_dir, subject_id + "_ratio_mad.nii.gz")
    if os.path.exists(out_file):
        return f"SKIP {subject_id}"
    try:
        t1_img = nib.load(t1w_path)
        t2_img = nib.load(t2w_path)
        t1_data = t1_img.get_fdata()
        t2_data = t2_img.get_fdata()

        # Base mask
        base_mask = (t1_data != 0) & (t2_data != 0)
        n_orig = base_mask.sum()

        # Compute raw ratio
        raw_ratio = np.zeros_like(t1_data)
        raw_ratio[base_mask] = t1_data[base_mask] / t2_data[base_mask]

        # MAD-based adaptive threshold
        ratio_vals = raw_ratio[base_mask]
        med = np.median(ratio_vals)
        mad = np.median(np.abs(ratio_vals - med))
        upper_thresh = med + MAD_K * MAD_SCALE * mad

        # Adaptive mask: keep only voxels within threshold
        valid_mask = base_mask & (raw_ratio <= upper_thresh) & (raw_ratio > 0)
        n_valid = valid_mask.sum()
        removed_pct = (1 - n_valid / n_orig) * 100

        # Clip at p98
        ratio_clean = raw_ratio[valid_mask]
        clip_98 = np.percentile(ratio_clean, 98)
        result = np.zeros_like(t1_data)
        result[valid_mask] = np.clip(ratio_clean, None, clip_98)

        nib.save(
            nib.Nifti1Image(result.astype(np.float32), affine=t1_img.affine),
            out_file,
        )

        brain = result[valid_mask]
        skew = float(sp_stats.skew(brain))
        cov = brain.std() / brain.mean()
        return (f"DONE {subject_id}: n={n_valid} removed={removed_pct:.1f}% "
                f"thresh={upper_thresh:.2f} mean={brain.mean():.3f} CoV={cov:.3f} "
                f"skew={skew:.3f} p98={clip_98:.2f}")
    except Exception as e:
        return f"ERROR {subject_id}: {e}"


def main():
    for pair in PAIRS:
        pair_path = DHCP_ROOT / pair
        t1_dir = pair_path / "t1w_reg2atlas"
        t2_dir = pair_path / "t2w_reg2atlas"
        out_dir = pair_path / "ratio_mad"
        out_dir.mkdir(exist_ok=True)

        t1_files = sorted(t1_dir.glob("*_T1w_reg2atlas.nii.gz"))
        tasks = []
        for t1f in t1_files:
            subject_id = t1f.name.replace("_T1w_reg2atlas.nii.gz", "")
            t2f = t2_dir / f"{subject_id}_T2w_reg2atlas.nii.gz"
            if not t2f.exists():
                continue
            tasks.append((subject_id, str(t1f), str(t2f), str(out_dir)))

        print(f"\n{'='*60}")
        print(f"  {pair}: {len(tasks)} subjects -> {out_dir}")
        print(f"  Method: MAD k={MAD_K} (adaptive per subject)")
        print(f"{'='*60}")

        with Pool(processes=8) as pool:
            for result in pool.imap_unordered(compute_ratio_mad, tasks):
                print(f"  {result}")

    print("\nDone.")


if __name__ == "__main__":
    main()
