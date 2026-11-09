import nibabel as nib
import numpy as np
from pathlib import Path

# -----------------------------
# -----------------------------
file_path = "$PATH/external_cohort/dataset_T1T2/case_0001/t1.nii.gz"
threshold = 50  # intensity threshold
output_path = "$PATH/external_cohort/dataset_T1T2/case_0001/t1_cropped.nii.gz"

# -----------------------------
# -----------------------------
img = nib.load(file_path)
data = img.get_fdata()
affine = img.affine

# -----------------------------
# -----------------------------
mask = data > threshold
masked_data = data * mask

# -----------------------------
# Bounding box crop
# -----------------------------
coords = np.array(np.nonzero(mask))
x_min, y_min, z_min = coords.min(axis=1)
x_max, y_max, z_max = coords.max(axis=1) + 1

cropped_data = masked_data[x_min:x_max, y_min:y_max, z_min:z_max]

# -----------------------------
# -----------------------------
new_affine = affine.copy()
new_affine[:3, 3] += np.dot(affine[:3, :3], [x_min, y_min, z_min])

# -----------------------------
# -----------------------------
cropped_img = nib.Nifti1Image(cropped_data, new_affine)
nib.save(cropped_img, output_path)
print(f"✅ Saved cropped MRI: {output_path}")
