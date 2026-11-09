import ants
import os
import numpy as np

def apply_registration_to_subject(i, template):

    subject_images = {
        "T1": f"$PATH/external_cohort/{i}/T1_brain_biascorr_restore.nii.gz",
        "T2": f"$PATH/external_cohort/{i}/T2_brain_biascorr_restore.nii.gz"
    }
    
    registered_images = {}
    
    for key, image_path in subject_images.items():
        if not os.path.exists(image_path):
            print(f"Image not found for subject {i}, {key} image: {image_path}")
            continue
        
        moving_image = ants.image_read(image_path)
        
        registration = ants.registration(fixed=template, moving=moving_image)
        
        registered_images[key] = registration['warpedmovout']
    
    if registered_images:
        output_dir = f"$PATH/external_cohort/{i}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if 'T1' in registered_images:
            t1_registered_path = os.path.join(output_dir, 'T1_template.nii.gz')
            ants.image_write(registered_images['T1'], t1_registered_path)
        
        if 'T2' in registered_images:
            t2_registered_path = os.path.join(output_dir, 'T2_template.nii.gz')
            ants.image_write(registered_images['T2'], t2_registered_path)
        
        print(f"Registration completed for subject {i}")
    else:
        print(f"Skipping subject {i} due to missing images")


neonatal_template_path = '$PATH/external_cohort/t1_reference_subject_15mm.nii.gz'
neonatal_template = ants.image_read(neonatal_template_path)


model_list = [
    'case_0001',]

for model in model_list:
    print(model)
    apply_registration_to_subject(model, neonatal_template)
