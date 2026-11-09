"""
Generate improved ratio images for dHCP dataset.

Key change: Exclude T2w low-value boundary voxels (below p5 of brain T2w)
before computing T1w/T2w ratio. This removes near-zero T2w voxels at tissue
boundaries that cause ratio explosion, making dHCP ratio distribution
closer to Infancy training data.

Output: {pair}/ratio_improved/  with *_ratio_improved.nii.gz files
"""
import os
import sys
from pathlib import Path
from multiprocessing import Pool

import nibabel as nib
import numpy as np

DHCP_ROOT = Path("$PATH/DHCP")
PAIRS = ["pair1", "pair2"]


def compute_ratio_improved(args):
    subject_id, t1w_path, t2w_path, out_dir = args
    out_file = os.path.join(out_dir, subject_id + "_ratio_improved.nii.gz")
    if os.path.exists(out_file):
        return f"SKIP {subject_id}: already done"
    try:
        t1_img = nib.load(t1w_path)
        t2_img = nib.load(t2w_path)
        t1_data = t1_img.get_fdata()
        t2_data = t2_img.get_fdata()

        # Original mask: both non-zero
        base_mask = (t1_data != 0) & (t2_data != 0)

        # Improved: exclude T2w voxels below p5 of brain region
        t2_brain = t2_data[base_mask]
        t2_p5 = np.percentile(t2_brain, 5)
        valid_mask = base_mask & (t2_data > t2_p5)

        # Compute ratio
        ratio = t1_data[valid_mask] / t2_data[valid_mask]

        # Clip at p98
        clip_98 = np.percentile(ratio, 98)
        result = np.zeros_like(t1_data)
        result[valid_mask] = np.clip(ratio, None, clip_98)

        nib.save(
            nib.Nifti1Image(result.astype(np.float32), affine=t1_img.affine),
            out_file,
        )
        brain = result[valid_mask]
        return (f"DONE {subject_id}: n_vox={valid_mask.sum()}, "
                f"mean={brain.mean():.3f}, std={brain.std():.3f}, "
                f"CoV={brain.std()/brain.mean():.3f}, p98={clip_98:.2f}")
    except Exception as e:
        return f"ERROR {subject_id}: {e}"


def main():
    for pair in PAIRS:
        pair_path = DHCP_ROOT / pair
        t1_dir = pair_path / "t1w_reg2atlas"
        t2_dir = pair_path / "t2w_reg2atlas"
        out_dir = pair_path / "ratio_improved"
        out_dir.mkdir(exist_ok=True)

        # Collect subject list from T1w files
        t1_files = sorted(t1_dir.glob("*_T1w_reg2atlas.nii.gz"))
        tasks = []
        for t1f in t1_files:
            subject_id = t1f.name.replace("_T1w_reg2atlas.nii.gz", "")
            t2f = t2_dir / f"{subject_id}_T2w_reg2atlas.nii.gz"
            if not t2f.exists():
                print(f"  SKIP {subject_id}: T2w not found")
                continue
            tasks.append((subject_id, str(t1f), str(t2f), str(out_dir)))

        print(f"\n{'='*60}")
        print(f"  {pair}: {len(tasks)} subjects")
        print(f"  Output: {out_dir}")
        print(f"{'='*60}")

        with Pool(processes=8) as pool:
            for result in pool.imap_unordered(compute_ratio_improved, tasks):
                print(f"  {result}")

    print("\nDone.")


if __name__ == "__main__":
    main()
