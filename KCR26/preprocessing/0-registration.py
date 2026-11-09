import subprocess
import os

input_dir = "$PATH/external_cohort/dataset_T1T2"

for model in sorted(os.listdir(input_dir)):
    model_path = os.path.join(input_dir, model)
    if not os.path.isdir(model_path):
        continue

    moving = os.path.join(model_path, "t2_crop.nii.gz")
    fixed = os.path.join(model_path, "t1_crop.nii.gz")
    out = os.path.join(model_path, "t2_reg2t1_flirt.nii.gz")
    omat = os.path.join(model_path, "t2_reg2t1_flirt.mat")

    if os.path.exists(moving) and os.path.exists(fixed):
        cmd = [
            "flirt",
            "-in", moving,
            "-ref", fixed,
            "-out", out,
            "-omat", omat,
            "-dof", "6"  # 6 = rigid registration
        ]

        print(f"\n🚀 Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Saved registered image: {out}")
        except subprocess.CalledProcessError as e:
            print(f"Error in {model}: {e}")
    else:
        print(f"⚠️ Skipping {model}: missing t1_crop.nii.gz or t2_crop.nii.gz")
