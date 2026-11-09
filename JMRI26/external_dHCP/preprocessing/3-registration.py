import ants
import os
import concurrent.futures

base_dir = "$PATH/DHCP/pair"
atlas_path = "$PATH/datasets/infancy_brain_MRI/atlas/t1_reference_subject_15mm.nii"

t1w_in_dir = os.path.join(base_dir, "t1w_biascorr")
t2w_in_dir = os.path.join(base_dir, "t2w_biascorr")
t1w_out_dir = os.path.join(base_dir, "t1w_reg2atlas")
t2w_out_dir = os.path.join(base_dir, "t2w_reg2atlas")

os.makedirs(t1w_out_dir, exist_ok=True)
os.makedirs(t2w_out_dir, exist_ok=True)

template = ants.image_read(atlas_path)


def register_to_atlas(subject_id):
    t1w_in = os.path.join(t1w_in_dir, subject_id + "_T1w_biascorr.nii.gz")
    t2w_in = os.path.join(t2w_in_dir, subject_id + "_T2w_biascorr.nii.gz")
    t1w_out = os.path.join(t1w_out_dir, subject_id + "_T1w_reg2atlas.nii.gz")
    t2w_out = os.path.join(t2w_out_dir, subject_id + "_T2w_reg2atlas.nii.gz")

    if not os.path.exists(t2w_in):
        return f"SKIP {subject_id}: T2w_biascorr not found"

    try:
        t1w_moving = ants.image_read(t1w_in)
        registration = ants.registration(fixed=template, moving=t1w_moving)
        ants.image_write(registration['warpedmovout'], t1w_out)

        t2w_moving = ants.image_read(t2w_in)
        t2w_warped = ants.apply_transforms(
            fixed=template,
            moving=t2w_moving,
            transformlist=registration['fwdtransforms'],
        )
        ants.image_write(t2w_warped, t2w_out)

        return f"DONE {subject_id}"
    except Exception as e:
        return f"ERROR {subject_id}: {e}"


if __name__ == "__main__":
    subject_ids = sorted(
        f.replace("_T1w_biascorr.nii.gz", "")
        for f in os.listdir(t1w_in_dir)
        if f.endswith(".nii.gz")
    )

    max_workers = 40
    print(f"Total: {len(subject_ids)} subjects | Workers: {max_workers}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(register_to_atlas, sid): sid
            for sid in subject_ids
        }
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

    print("All done.")
