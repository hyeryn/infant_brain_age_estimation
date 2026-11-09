import os
import shutil
import nibabel as nib
import numpy as np

input_dir = "$PATH/external_cohort/dataset_T1T2"
crop_size_mm = 170.0

for root, dirs, files in os.walk(input_dir):
    for file_name in files:
        if file_name.endswith(".nii.gz"):
            input_path = os.path.join(root, file_name)
            output_name = file_name.replace(".nii.gz", "_crop.nii.gz")
            output_path = os.path.join(root, output_name)

            try:
                nii = nib.load(input_path)
                data = nii.get_fdata()
                affine = nii.affine.copy()

                z_spacing = np.abs(affine[2, 2])
                num_slices = data.shape[2]

                crop_slices = int(np.round(crop_size_mm / z_spacing))

                if crop_slices >= num_slices:
                    shutil.copy(input_path, output_path)
                    print(f"ℹ️ Copied (no crop needed): {input_path} -> {output_path}")
                else:
                    end = num_slices
                    start = max(end - crop_slices, 0)
                    cropped_data = data[:, :, start:end]

                    new_affine = affine.copy()
                    new_affine[:3, 3] = affine[:3, 3] + affine[:3, 2] * start

                    cropped_nii = nib.Nifti1Image(cropped_data, new_affine)
                    nib.save(cropped_nii, output_path)
                    print(f"✅ Cropped {input_path} -> {output_path} | slices: {start}-{end-1}")

            except Exception as e:
                print(f"Failed: {input_path} | {e}")
