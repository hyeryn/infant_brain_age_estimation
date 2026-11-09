import os
from multiprocessing import Pool

os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "4"
import ants

base_dir = "$PATH/DHCP/pair"
atlas = "$PATH/datasets/infancy_brain_MRI/atlas/t1_reference_subject_15mm.nii"

t1w_in_dir = os.path.join(base_dir, "t1w_reg2atlas")
t2w_in_dir = os.path.join(base_dir, "t2w_reg2atlas")
t1w_out_dir = os.path.join(base_dir, "t1w_template")
t2w_out_dir = os.path.join(base_dir, "t2w_template")

os.makedirs(t1w_out_dir, exist_ok=True)
os.makedirs(t2w_out_dir, exist_ok=True)


def register_to_template(subject_id):
    t1w_in = os.path.join(t1w_in_dir, subject_id + "_T1w_reg2atlas.nii.gz")
    t2w_in = os.path.join(t2w_in_dir, subject_id + "_T2w_reg2atlas.nii.gz")
    t1w_out = os.path.join(t1w_out_dir, subject_id + "_T1w_template.nii.gz")
    t2w_out = os.path.join(t2w_out_dir, subject_id + "_T2w_template.nii.gz")

    if not os.path.exists(t2w_in):
        return f"SKIP {subject_id}: T2w_reg2atlas not found"

    try:
        template = ants.image_read(atlas)

        # ANTs nonlinear registration on T1w
        t1w_moving = ants.image_read(t1w_in)
        reg = ants.registration(fixed=template, moving=t1w_moving)
        ants.image_write(reg["warpedmovout"], t1w_out)

        # Apply same transforms to T2w
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


if __name__ == "__main__":
    subject_ids = sorted(
        f.replace("_T1w_reg2atlas.nii.gz", "")
        for f in os.listdir(t1w_in_dir)
        if f.endswith(".nii.gz") and ".mat" not in f
    )

    # ANTs uses 4 threads per process -> 48 cores / 4 = 12 workers
    n_workers = 12
    print(f"Total: {len(subject_ids)} subjects | Workers: {n_workers} (4 threads each)")

    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(register_to_template, subject_ids):
            print(result)

    print("All done.")
