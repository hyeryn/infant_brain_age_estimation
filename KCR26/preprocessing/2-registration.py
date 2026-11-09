import subprocess
import os

model_list = [
    'case_0001',]

base_dir = "$PATH/external_cohort"

for model in model_list:
    moving = os.path.join(base_dir, model, "T2_brain.nii.gz")
    fixed = os.path.join(base_dir, model, "T1_brain.nii.gz")
    out = os.path.join(base_dir, model, "T2_reg2T1_flirt.nii.gz")
    omat = os.path.join(base_dir, model, "T2_reg2T1_flirt.mat")

    cmd = [
        "flirt",
        "-in", moving,
        "-ref", fixed,
        "-out", out,
        "-omat", omat,
        "-dof", "6"   # 6 = rigid, 12 = affine
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Saved registered image:", out)


#         "flirt",
#         "-in", moving,
#         "-ref", fixed,
#         "-out", out,
#         "-omat", omat,
#         "-dof", "6"   # 6 = rigid, 12 = affine
