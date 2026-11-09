import nibabel as nib
import numpy as np
import os

import nibabel as nib
import numpy as np
import os

def process_patient(patient):
    
    t1_path = f'$PATH/external_cohort/{patient}/T1_template.nii.gz'
    t2_path = f'$PATH/external_cohort/{patient}/T2_template.nii.gz'
    
    if not os.path.exists(t1_path) or not os.path.exists(t2_path):
        print(f"Skipping {patient} due to missing files")
        return
    
    t1_img = nib.load(t1_path)
    t2_img = nib.load(t2_path)
    t1_data = t1_img.get_fdata()
    t2_data = t2_img.get_fdata()

    valid_mask = (t1_data != 0) & (t2_data != 0)
    
    result = np.zeros_like(t1_data)
    result[valid_mask] = (t1_data[valid_mask]) / (t2_data[valid_mask])
    
    output_dir = f'$PATH/external_cohort/{patient}/'
    os.makedirs(output_dir, exist_ok=True)
    
    mask_nifti = nib.Nifti1Image(result, affine=t1_img.affine)
    nib.save(mask_nifti, os.path.join(output_dir, 'ratio.nii.gz'))
    
    print(f"Processed {patient}")

    result = np.zeros_like(t1_data)
    ratio = t1_data[valid_mask] / t2_data[valid_mask]
    
    percentile_clip=90
    clip_val = np.percentile(ratio, percentile_clip)
    ratio_clipped = np.clip(ratio, None, clip_val)
    
    result[valid_mask] = ratio_clipped
    
    output_dir = f'$PATH/external_cohort/{patient}/'
    os.makedirs(output_dir, exist_ok=True)
    
    mask_nifti = nib.Nifti1Image(result, affine=t1_img.affine)
    nib.save(mask_nifti, os.path.join(output_dir, f'ratio_clipped_{percentile_clip}.nii.gz'))
    
    print(f"Processed {patient} (clip {percentile_clip}th percentile: {clip_val:.4f})")

    percentile_clip=95
    clip_val = np.percentile(ratio, percentile_clip)
    ratio_clipped = np.clip(ratio, None, clip_val)
    
    result[valid_mask] = ratio_clipped
    
    output_dir = f'$PATH/external_cohort/{patient}/'
    os.makedirs(output_dir, exist_ok=True)
    
    mask_nifti = nib.Nifti1Image(result, affine=t1_img.affine)
    nib.save(mask_nifti, os.path.join(output_dir, f'ratio_clipped_{percentile_clip}.nii.gz'))
    
    print(f"Processed {patient} (clip {percentile_clip}th percentile: {clip_val:.4f})")


model_list = [
    'case_0001',]

for model in model_list:
    print(model)
    process_patient(model)
