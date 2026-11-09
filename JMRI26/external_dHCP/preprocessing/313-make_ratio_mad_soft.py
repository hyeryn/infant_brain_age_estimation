"""
Generate adaptive ratio images using MAD-based soft masking.

For each subject:
1. Compute raw T1w/T2w ratio
2. Calculate MAD threshold (median + 3 * 1.4826 * MAD)
3. Create binary mask (1=keep, 0=outlier)
4. Gaussian blur the mask → smooth transition at boundaries
5. Multiply ratio by soft mask → outlier regions fade smoothly to 0
6. Clip remaining at p98

Output: {pair}/ratio_mad_soft/ with *_ratio_mad_soft.nii.gz
"""
import os
from pathlib import Path
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from scipy import stats as sp_stats
from scipy.ndimage import gaussian_filter

DHCP_ROOT = Path("$PATH/DHCP")
PAIRS = ["pair1", "pair2"]
MAD_K = 3
MAD_SCALE = 1.4826
SIGMA = 1.5  # Gaussian blur sigma (in voxels)


def compute_ratio_mad_soft(args):
    subject_id, t1w_path, t2w_path, out_dir = args
    out_file = os.path.join(out_dir, subject_id + "_ratio_mad_soft.nii.gz")
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

        # Binary mask: 1 = within threshold, 0 = outlier
        binary_mask = np.zeros_like(t1_data)
        binary_mask[base_mask & (raw_ratio <= upper_thresh) & (raw_ratio > 0)] = 1.0

        # Gaussian blur the mask → smooth transition [0, 1]
        soft_mask = gaussian_filter(binary_mask, sigma=SIGMA)

        # Normalize: ensure max is 1.0 within brain region
        brain_max = soft_mask[base_mask].max()
        if brain_max > 0:
            soft_mask = soft_mask / brain_max
        soft_mask = np.clip(soft_mask, 0, 1)

        # Apply soft mask to ratio
        soft_ratio = raw_ratio * soft_mask

        # Zero out non-brain regions
        soft_ratio[~base_mask] = 0

        nonzero_mask = soft_ratio > 0
        if nonzero_mask.sum() > 0:
            clip_98 = np.percentile(soft_ratio[nonzero_mask], 98)
            soft_ratio[nonzero_mask] = np.clip(soft_ratio[nonzero_mask], None, clip_98)

        nib.save(
            nib.Nifti1Image(soft_ratio.astype(np.float32), affine=t1_img.affine),
            out_file,
        )

        # Stats on result
        brain = soft_ratio[nonzero_mask]
        n_nonzero = nonzero_mask.sum()
        removed_pct = (1 - n_nonzero / n_orig) * 100
        skew = float(sp_stats.skew(brain))
        cov = brain.std() / brain.mean()

        hard_removed = (base_mask & (binary_mask == 0)).sum()
        soft_region = (base_mask & (soft_mask > 0) & (soft_mask < 0.99)).sum()

        return (f"DONE {subject_id}: n={n_nonzero} hard_cut={hard_removed}({hard_removed/n_orig*100:.1f}%) "
                f"soft_transition={soft_region} mean={brain.mean():.3f} CoV={cov:.3f} "
                f"skew={skew:.3f} p98={clip_98:.2f}")
    except Exception as e:
        return f"ERROR {subject_id}: {e}"


def main():
    for pair in PAIRS:
        pair_path = DHCP_ROOT / pair
        t1_dir = pair_path / "t1w_reg2atlas"
        t2_dir = pair_path / "t2w_reg2atlas"
        out_dir = pair_path / "ratio_mad_soft"
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
        print(f"  Method: MAD k={MAD_K} soft mask (sigma={SIGMA})")
        print(f"{'='*60}")

        with Pool(processes=8) as pool:
            for result in pool.imap_unordered(compute_ratio_mad_soft, tasks):
                print(f"  {result}")

    print("\nDone.")


if __name__ == "__main__":
    main()
