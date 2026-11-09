


import pandas as pd

volume_df = pd.read_csv("$PATH/datasets/infancy_brain_MRI/brain/report/sienax_full_report.csv")
meta_df = pd.read_csv("$PATH/datasets/infancy_brain_MRI/infancyMRI_info_under_24.csv")

meta_df = meta_df[['image_id', 'doctor_predicted_age']]

merged_df = pd.merge(meta_df, volume_df, left_on='image_id', right_on='SubjectID')

final_df = merged_df[['image_id', 'doctor_predicted_age', 'WHITE_ICV', 'BRAIN_ICV', 'WM/ICV_ratio']]

final_df.to_csv("sienax_output.csv", index=False)

print("Saved merged CSV to 'merged_output.csv'.")
