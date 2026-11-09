import os
import subprocess

input_dir = "$PATH/external_cohort/dataset_MAGiC"

target_files = ("t1.nii.gz", "t2.nii.gz")

# ----------------------------------------
# ----------------------------------------
for root, dirs, files in os.walk(input_dir):
    for file_name in files:
        if file_name.endswith(target_files):
            input_path = os.path.join(root, file_name)
            
            if file_name.endswith("t1.nii.gz"):
                output_name = file_name.replace("t1.nii.gz", "t1_brain.nii.gz")
            elif file_name.endswith("t2.nii.gz"):
                output_name = file_name.replace("t2.nii.gz", "t2_brain.nii.gz")
            else:
                continue
            
            output_path = os.path.join(root, output_name)
            
            cmd = [
                "bet",
                input_path,
                output_path,
            ]
            
            try:
                subprocess.run(cmd, check=True)
                print(f"✅ Done: {input_path} -> {output_path}")
            except subprocess.CalledProcessError as e:
                print(f"Failed: {input_path} | {e}")


# # ----------------------------------------
# # ----------------------------------------


#                 "bet",
#                 input_path,
#                 output_path,
