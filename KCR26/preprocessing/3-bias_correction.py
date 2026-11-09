import subprocess
import os

model_list = [
    'case_0001',]

base_dir = "$PATH/external_cohort"

for model in model_list:
    t1_path = os.path.join(base_dir, model, "T1_brain.nii.gz")
    out_prefix = os.path.join(base_dir, model, "T1_brain_biascorr")

    cmd = [
        "fast",
        "-B",               # bias field correction
        "-o", out_prefix,
        t1_path
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Bias-corrected image saved:", out_prefix + "_restore.nii.gz")

    t2_path = os.path.join(base_dir, model, "T2_reg2T1_flirt.nii.gz")
    out_prefix = os.path.join(base_dir, model, "T2_brain_biascorr")

    cmd = [
        "fast",
        "-B",               # bias field correction
        "-o", out_prefix,
        t2_path
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Bias-corrected image saved:", out_prefix + "_restore.nii.gz")
