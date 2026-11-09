"""
Generate improved ratio images with T2w > p10 masking.
More aggressive than p5 — removes bottom 10% of T2w brain voxels.
Output: {pair}/ratio_improved_p10/ with *_ratio_improved_p10.nii.gz
"""
import os
from pathlib import Path
from multiprocessing import Pool

import nibabel as nib
import numpy as np

DHCP_ROOT = Path("$PATH/DHCP")
PAIRS = ["pair1", "pair2"]


def compute_ratio(args):
    subject_id, t1w_path, t2w_path, out_dir = args
    out_file = os.path.join(out_dir, subject_id + "_ratio_improved_p10.nii.gz")
    if os.path.exists(out_file):
        return f"SKIP {subject_id}"
    try:
        t1_img = nib.load(t1w_path)
        t2_img = nib.load(t2w_path)
        t1_data = t1_img.get_fdata()
        t2_data = t2_img.get_fdata()

        base_mask = (t1_data != 0) & (t2_data != 0)
        t2_brain = t2_data[base_mask]
        t2_p10 = np.percentile(t2_brain, 10)
        valid_mask = base_mask & (t2_data > t2_p10)

        ratio = t1_data[valid_mask] / t2_data[valid_mask]
        clip_98 = np.percentile(ratio, 98)
        result = np.zeros_like(t1_data)
        result[valid_mask] = np.clip(ratio, None, clip_98)

        nib.save(
            nib.Nifti1Image(result.astype(np.float32), affine=t1_img.affine),
            out_file,
        )
        brain = result[valid_mask]
        return (f"DONE {subject_id}: n_vox={valid_mask.sum()}, "
                f"mean={brain.mean():.3f}, CoV={brain.std()/brain.mean():.3f}")
    except Exception as e:
        return f"ERROR {subject_id}: {e}"


def main():
    for pair in PAIRS:
        pair_path = DHCP_ROOT / pair
        t1_dir = pair_path / "t1w_reg2atlas"
        t2_dir = pair_path / "t2w_reg2atlas"
        out_dir = pair_path / "ratio_improved_p10"
        out_dir.mkdir(exist_ok=True)

        t1_files = sorted(t1_dir.glob("*_T1w_reg2atlas.nii.gz"))
        tasks = []
        for t1f in t1_files:
            subject_id = t1f.name.replace("_T1w_reg2atlas.nii.gz", "")
            t2f = t2_dir / f"{subject_id}_T2w_reg2atlas.nii.gz"
            if not t2f.exists():
                continue
            tasks.append((subject_id, str(t1f), str(t2f), str(out_dir)))

        print(f"{pair}: {len(tasks)} subjects -> {out_dir}")
        with Pool(processes=8) as pool:
            for result in pool.imap_unordered(compute_ratio, tasks):
                print(f"  {result}")

    print("\nDone.")


if __name__ == "__main__":
    main()
