# Infant brain age estimation from T1w/T2w ratio imaging

Code for training, inference and evaluation of 3D convolutional networks that estimate
brain age from infant MRI, using T1-weighted (T1w), T2-weighted (T2w) and T1w/T2w ratio
image (RI) inputs.

Patient images, identifiers, split lists and trained weights are not part of this repository.
Sample names such as `sample_0001` and `case_0001` are placeholders.

## Repository layout

| Directory | Scope |
|---|---|
| `JMRI26/` | Comparison of 3D CNN architectures across input types, multitask training, statistics, cross-validation stability, external inference on dHCP |
| `ESNR26/` | ResNet with FiLM conditioning on white matter and intracranial volume |
| `RSNA26/` | Training at 0.6 mm, Grad-CAM, statistics for ResNet18 and ChenNet |
| `KCR26/` | Preprocessing of and inference on an external cohort. Uses the training and Grad-CAM code in `RSNA26/` |
| `THIRD_PARTY_LICENSES/` | License texts of third-party code |

Each training directory has `training/classifier/` (trainer, Lightning module, models,
transforms), `training/libs/` (utilities) and `training/dataset_config/` (dataset configs).

## Pipeline

| Step | Code | Notes |
|---|---|---|
| 1. Preprocessing | `KCR26/preprocessing/`, `JMRI26/external_dHCP/preprocessing/` | Registration, brain extraction, bias field correction, template registration |
| 2. Ratio image | `5-ratio.py`, `300-total_make_ratio.py`, `31x-make_ratio_*.py` | Voxel-wise T1w/T2w with outlier handling (percentile clipping, MAD-based masking) |
| 3. Dataset config | `training/dataset_config/*.json` | Image file names, crop size, label column |
| 4. Training | `training/classifier/lit_trainer*.py` | 5-fold cross-validation |
| 5. Inference | `training/classifier/run_inference.py`, `JMRI26/external_dHCP/inference_and_analysis/run_inference*.py`, `KCR26/analysis/external_inference.ipynb` | Per-subject predictions |
| 6. Metrics and statistics | See [Evaluation](#evaluation) | |
| 7. Visualization | `*/analysis/attention/gradcam_example.ipynb` | Grad-CAM example for a single sample |

## Setup

Main Python dependencies: PyTorch, PyTorch Lightning, MONAI, NumPy, SciPy, scikit-learn,
pandas, nibabel, ANTsPy, p_tqdm, pytorch-grad-cam, matplotlib, seaborn. FSL is required for
the preprocessing scripts.

File locations in the code use the placeholder `$PATH` (for example
`$PATH/datasets/infancy_brain_MRI`). It is a literal string and is not expanded from the
environment, so replace it with your own root directory before running:

```bash
grep -rl '\$PATH' --include='*.py' --include='*.ipynb' . | xargs sed -i 's#\$PATH#/your/root#g'
```

## Data

Each subject has a directory of registered NIfTI volumes. The files used for an experiment
are listed under `img_files` in the dataset config. A metadata CSV provides the subject
identifier and target age (`subject_col`, `label_col_reg`). The trainer loads
`<data_path>/<exp_name>.json`, so place the configs in the directory set as `data_path`.

| Input | Channels |
|---|---|
| T1w | 1 |
| T2w | 1 |
| T1w+T2w | 2 |
| RI | 1 |
| T1w+T2w+RI | 3 |

## Training

```bash
cd JMRI26/training/classifier

# single run
python lit_trainer.py --model resnet18 --exp_name ratio --fold 0 --gpu 0

# all architectures, inputs and folds
bash b_basic.sh

# multitask model (age regression + segmentation + reconstruction)
python lit_trainer_seg.py --model resnet_seg --exp_name ratio_seg --gpu 0
```

Other configurations:

```bash
# FiLM conditioning on WM and ICV
cd ESNR26/training/classifier
python lit_trainer.py --model resnet_film --exp_name ratio_origin_15 --fold 0 \
    -add_option both -data_option freesurfer --gpu 0
bash run_training.sh 0       # argument is the GPU id

# 0.6 mm
cd RSNA26/training/classifier
bash run_training.sh 0
```

Available architectures in `JMRI26`: `agenet`, `basiccnn`, `cbr_tiny`, `chen_net`, `densenet`,
`efficientnet`, `resnet18`, `resnet34`, `resnet50`, `sfcn`, `unet`.

| Setting | `JMRI26` | `ESNR26` | `RSNA26` |
|---|---|---|---|
| Trainer | `lit_trainer.py` | `lit_trainer.py` | `lit_trainer2.py` |
| Voxel size | 1.5 mm | 1.5 mm | 0.6 mm |
| 3D crop | 86 x 94 x 84 | 86 x 94 x 84 | 215 x 234 x 42 |
| Optimizer | Adam, lr 1e-3 | Adam, lr 1e-3 | Adam, lr 1e-3 |
| Scheduler | Cosine annealing | Cosine annealing | Cosine annealing |
| Batch size | 16 | 16 | 16 |
| Maximum epochs | 150 | 150 | 100 |
| Early stopping | Patience 10 | None | None |
| Loss | MSE (KLD for SFCN) | MSE | MSE (KLD for SFCN) |

Each fold uses a 60/20/20 train/validation/test split at the subject level. The checkpoint
with the lowest validation loss is kept and used for testing. Training uses 16-bit mixed
precision. Augmentation is applied with probability 0.3 per transform: random zoom, contrast
adjustment, Gaussian smoothing and flip, followed by random cropping. Further details are in
`JMRI26/docs/Supplementary_Methods_Training_Details.md`, and model sizes in
`JMRI26/docs/model_parameter_counts.md`.

## Inference on external data

Trained checkpoints can be applied to external cohorts without fine-tuning. The external data
must first go through the preprocessing and ratio steps above.

```bash
cd JMRI26/external_dHCP/inference_and_analysis
bash run_all.sh 0      # argument is the GPU id
```

`run_inference*.py` variants differ in how the input ratio image is prepared. The scripts
iterate over architectures, input types and folds, and write predictions and metrics per
combination. `p5_preprocessing/analyze_p5_full.py` and `analyze_p5_results.py` aggregate
those outputs across models and folds.

## Evaluation

With `error = predicted age - true age`:

| Metric | Definition |
|---|---|
| MAE | `mean(abs(error))` |
| RMSE | `sqrt(mean(error^2))` |
| Bias | `mean(error)` |
| Bias-corrected MAE | `mean(abs(predicted - bias - true))` |
| R² | `1 - SS_res / SS_tot` |
| Pearson r, Spearman ρ | Between true and predicted age |

Group comparisons use metric values per run as the unit of analysis, with independent runs
from cross-validation folds and random seeds:

| Analysis | Method |
|---|---|
| Across input types | One-way ANOVA, Kruskal-Wallis |
| Pairwise | Welch's t-test, Mann-Whitney U (two-sided) |
| Effect size | Cohen's d |

```bash
# descriptive statistics, omnibus and pairwise tests
python RSNA26/analysis/statistics/RSNA26_stats.py

# fold and seed variability
python JMRI26/cv_robustness/build_cv_analysis.py
python JMRI26/cv_robustness/visualize_cv.py
```

The notebook for the full architecture comparison is
`JMRI26/analysis/statistics/effect_size_and_stats.ipynb`.

## Notes

Notebooks are provided without outputs. Results depend on data, splits and random seeds, and
reruns may not be bit-identical to earlier runs.

## Third-party code

The training framework is derived from [MyelinAge](https://github.com/wasserth/MyelinAge)
and has been modified for this work. MyelinAge is licensed under the Apache License 2.0, see
`THIRD_PARTY_LICENSES/MyelinAge-Apache-2.0.txt`.
