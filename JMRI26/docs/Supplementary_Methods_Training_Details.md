# Supplementary Methods: Model Training and Optimization Details

---

## Methods


### Training Framework and General Pipeline

All models were implemented using PyTorch (via PyTorch Lightning) with MONAI for medical image transformations. Training was performed on NVIDIA GPUs with 16-bit mixed-precision training enabled through PyTorch Lightning's native automatic mixed precision (AMP). TensorBoard was used for logging baseline experiments; Weights & Biases (wandb) was used for logging multitask experiments.

### Data Splitting and Cross-Validation

A five-fold cross-validation scheme was employed. Data were split into training (60%), validation (20%), and test (20%) sets using a `KFold` split with `shuffle=True`. Stratification by class was not applied (i.e., `skip_stratification=True`), as the primary task was regression. The fold assignment was pre-computed and saved to a JSON file to ensure reproducibility across models. For baseline models, a global random seed of 2025 was used; for multitask models, a seed of 1234 was used. In both cases, Python, NumPy, PyTorch, and CUDA random states were fixed, and `torch.backends.cudnn.deterministic` was set to `True` with `torch.backends.cudnn.benchmark` set to `False`.

### Input Data and Preprocessing

All models operated on 3D volumetric MRI inputs. The 3D crop size was fixed at 86 × 94 × 84 voxels across all experiments and modality configurations. Five input modality configurations were evaluated: T1-weighted only, T2-weighted only, combined T1 + T2 (2-channel), T1/T2 ratio map only, and combined T1 + T2 + ratio (3-channel). Channel-wise z-score normalization (zero mean, unit variance) was applied to all inputs. No intensity clipping was used.

### Data Augmentation

During training, the following augmentations were applied stochastically with a per-transform probability of 0.3:

- **Random zoom**: scale factor sampled uniformly from [0.7, 1.2] with constant padding.
- **Random contrast adjustment**: gamma sampled from [0.5, 1.2].
- **Random Gaussian smoothing**: sigma sampled from [0.1, 0.8] independently per axis.
- **Random flip**: random axis flipping.
- **Random spatial crop**: images were first padded/cropped to 1.05× the target size, then randomly cropped to the final 86 × 94 × 84 size.

No augmentation was applied during validation or testing; images were center-cropped (via `ResizeWithPadOrCrop`) and normalized identically.

### Optimization

Unless otherwise noted, all models were trained with the Adam optimizer (default PyTorch betas of β₁ = 0.9, β₂ = 0.999) with an initial learning rate of 1 × 10⁻³ and no weight decay. A cosine annealing learning rate schedule (`CosineAnnealingLR`) was used with `T_max` set equal to the maximum number of epochs. Training was conducted for a maximum of 150 epochs with early stopping (patience = 10 epochs) monitoring the validation loss. The batch size was 16 for all 3D experiments. No gradient clipping was applied.

### Loss Functions

For the primary age regression task, mean squared error (MSE) loss was used for all models except SFCN. SFCN used a KL divergence (KLD) loss following the soft-label approach, in which continuous age labels were converted to probability distributions over a discretized age range of [−1, 36] months with bin width 1 and sigma 1 (via `num2vect`), and `torch.nn.KLDivLoss(reduction='sum')` was applied, normalized by batch size.

### Model Checkpoint Selection

Model selection was based on the lowest validation loss (`loss/val`, mode `min`). Only the single best checkpoint (`save_top_k=1`) was retained per fold.

### Baseline Models

Eleven baseline architectures were evaluated: AgeNet, BasicCNN, CBR-Tiny, ChenNet, DenseNet121, EfficientNet-B0, ResNet-18, ResNet-34, ResNet-50, SFCN, and U-Net (encoder only). All models were implemented as 3D convolutional networks with an adaptive average pooling layer followed by a fully connected regression head outputting a single scalar (predicted age in months). All models were trained from random initialization (no ImageNet pretraining). Dropout rates varied by architecture (see Supplementary Table D).

### FiLM-Conditioned Models

For each baseline architecture, a Feature-wise Linear Modulation (FiLM) variant was also evaluated. FiLM layers received two scalar conditioning inputs — FreeSurfer-derived white matter (WM) volume and intracranial volume (ICV) — which were min-max normalized across the dataset. FiLM blocks were inserted at intermediate network layers (typically after the 2nd and 3rd encoding stages) and applied learned affine transformations (scale and shift) to feature maps conditioned on the anatomical priors.

### Biologically Informed Multitask Model

The multitask model extended either a ResNet-18 encoder or a U-Net encoder with three output heads: (1) a white matter segmentation head (binary cross-entropy + Dice loss), (2) an image reconstruction head (MSE loss), and (3) an age regression head with deep supervision from multiple encoder levels. The total loss was a weighted combination:

**L_total = λ_seg × L_seg + λ_recon × L_recon + λ_age × L_age**

where L_seg = 0.5 × BCE + 0.5 × Dice. Default loss weights were λ_seg = 0.0, λ_recon = 0.0, and λ_age = 1.0 (age-only baseline). When deep supervision was enabled, per-level age loss weights of [1.0, 0.6, 0.3, 0.1] were applied from the deepest to shallowest encoder level.

### Hyperparameter Search for Multitask Models

Hyperparameter tuning for the multitask model was conducted by varying the loss component weights. The following values were searched (as documented in shell scripts):

- **Segmentation loss weight (λ_seg)**: {0.0, 0.1, 0.5, 1.0, 5.0, 10.0}
- **Reconstruction loss weight (λ_recon)**: {0.0, 0.1, 0.5, 0.7, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0}
- **Age loss weight (λ_age)**: {1.0, 10.0}
- **Per-level age loss weights**: {[1.0, 0.0, 0.0, 0.0], [1.0, 0.6, 0.3, 0.1]}

The search was script-driven (shell scripts launching multiple runs in parallel). Final settings were selected based on the best validation loss. The search was primarily conducted on fold 0 for efficiency.

### Hardware

Training was distributed across up to 5 GPUs (GPU IDs 0–4) in parallel at the experiment level (one model per GPU), with each individual training run using a single GPU. DataLoader workers were set to 10 with pinned memory enabled.

---

## Table: General Training Configuration

| Component | Setting | Value | Source |
|-----------|---------|-------|--------|
| Framework | Deep learning library | PyTorch + PyTorch Lightning | `lit_trainer.py:26–28` |
| Framework | Medical image transforms | MONAI | `transformations.py:13–19` |
| Framework | Logging (baseline) | TensorBoard | `lit_trainer.py:403` |
| Framework | Logging (multitask) | Weights & Biases | `lit_trainer_seg.py:438` |
| Data split | Cross-validation | 5-fold KFold (shuffle=True) | `utils.py:157` |
| Data split | Train / Val / Test ratio | 60% / 20% / 20% | `utils.py:176–179` |
| Data split | Stratification | Disabled (skip_stratification=True) | `lit_trainer.py:236` |
| Reproducibility | Random seed (baseline) | 2025 | `lit_trainer.py:447` |
| Reproducibility | Random seed (multitask) | 1234 | `lit_trainer_seg.py:458` |
| Reproducibility | Deterministic cuDNN | True | `lit_trainer.py:460–461` |
| Reproducibility | Deterministic algorithms | True (warn_only=True) | `lit_trainer.py:462` |
| Input | 3D crop size | 86 × 94 × 84 | `dataset/*.json` |
| Input | Normalization | Channel-wise z-score | `lit_trainer.py:287–288` |
| Input | Intensity clipping | Disabled | `lit_trainer.py:289` |
| Input | Zoom factor | 1.0 × 1.0 × 1.0 (no resize) | `lit_trainer.py:72` |
| Augmentation | Overall probability | 0.3 per transform | `lit_trainer.py:94` |
| Augmentation | Random zoom | [0.7, 1.2] | `transformations.py:88` |
| Augmentation | Random contrast | gamma [0.5, 1.2] | `transformations.py:95` |
| Augmentation | Random Gaussian smooth | sigma [0.1, 0.8] per axis | `transformations.py:100` |
| Augmentation | Random flip | Enabled | `transformations.py:105` |
| Augmentation | Spatial crop padding | 1.05× then random crop | `transformations.py:168–170` |
| Augmentation | Elastic deformation | Disabled | `lit_trainer.py:96` |
| Augmentation | Gaussian noise | Disabled | `lit_trainer.py:100` |
| Optimizer | Type | Adam | `lit_classifier.py:335–337` |
| Optimizer | Learning rate | 1 × 10⁻³ | `lit_trainer.py:77` |
| Optimizer | Betas (β₁, β₂) | PyTorch default (0.9, 0.999) | `lit_classifier.py:336` |
| Optimizer | Weight decay | 0.0 | `lit_trainer.py:67` |
| Scheduler | Type | CosineAnnealingLR | `lit_classifier.py:351–352` |
| Scheduler | T_max | max_epochs (150) | `lit_classifier.py:352` |
| Training | Batch size | 16 | `lit_trainer.py:65` |
| Training | Max epochs | 150 | `lit_trainer.py:73` |
| Training | Precision | 16-bit (AMP) | `lit_trainer.py:129` |
| Training | Gradient clipping | Not applied | Not found in code |
| Training | DataLoader workers | 10 | `lit_trainer.py:348` |
| Early stopping | Patience | 10 epochs | `lit_trainer.py:417` |
| Early stopping | Monitor | loss/val (min) | `lit_trainer.py:417` |
| Checkpoint | Selection metric | loss/val (min) | `lit_trainer.py:136–137` |
| Checkpoint | Save top-k | 1 | `lit_trainer.py:409` |
| Loss (regression) | Type | MSE (all models except SFCN) | `lit_classifier.py:117` |
| Loss (SFCN) | Type | KLDivLoss(reduction='sum') / batch_size | `lit_classifier.py:121` |
| Loss (SFCN) | Label encoding | Soft labels, range [−1, 36], bin=1, σ=1 | `lit_classifier.py:157` |
| FiLM conditioning | Auxiliary data source | FreeSurfer | `b_basic.sh:56` |
| FiLM conditioning | Conditioning inputs | WM volume + ICV (min-max normalized) | `lit_trainer.py:199–210` |
| Multitask loss | Segmentation | 0.5 × BCE + 0.5 × Dice | `lit_classifier_seg.py:67–68` |
| Multitask loss | Reconstruction | MSE | `lit_classifier_seg.py:112` |
| Multitask loss | Age (deep supervision) | MSE per encoder level | `lit_classifier_seg.py:123–126` |
| Multitask loss | Default weights | λ_seg=0.0, λ_recon=0.0, λ_age=1.0 | `lit_trainer_seg.py:69–71` |
| Multitask loss | Deep supervision weights | [1.0, 0.0, 0.0, 0.0] (default) or [1.0, 0.6, 0.3, 0.1] | `lit_classifier_seg.py:118` |

---

## Model-specific table

| Model | Input Channels | Backbone / Architecture Notes | Dropout | Channel Sizes | Regression Head | Loss | FiLM Variant | Source |
|-------|---------------|------------------------------|---------|---------------|-----------------|------|-------------|--------|
| AgeNet | 1–3 | Inception + Fire modules (6 stages); stem: 32→64→192; max 832 channels | None specified | 32→64→192→...→832→512 | FC: 512→512→256→128→1 | MSE | AgeNet_film (2 FiLM blocks after fire4, fire5) | `models/agenet.py` |
| BasicCNN | 1–3 | 4 conv blocks + BN + ReLU; AdaptiveAvgPool([8,8,8]) | 0.4 | 8→16→32→64 | FC: (64×512)→128→1 | MSE | BasicCNN_film (2 FiLM blocks after conv2, conv3) | `models/basic_cnn.py` |
| CBR-Tiny | 1–3 | 4 conv blocks + BN + ReLU; nr_filt=8 | None | 8→16→32→64 | FC: (64×64)→1 | MSE | CbrTiny_film (2 FiLM blocks after conv2, conv3) | `models/cbr_tiny.py` |
| ChenNet | 1–3 | 6 conv blocks (nr_filt=16); 1×1 conv final reduction; based on Chen et al. (Radiology 2022) | 0.5 | 16→32→64→128→256→64 | FC: (64×8)→512→1 | MSE | ChenNet_film (4 FiLM blocks after conv2–5) | `models/chen_net.py` |
| DenseNet121 | 1–3 | DenseNet-121 (3D); block_config=(6,12,24,16); growth_rate=32 | 0.3 (Dropout3d) | 64→...→1024 | Linear: features→1 | MSE | DenseNet121_film (3 FiLM blocks after transitions) | `models/densenet.py` |
| EfficientNet-B0 | 1–3 | EfficientNet-B0 (3D); via `efficientnet_pytorch_3d` | Inherent | Inherent to B0 | Custom FC→1 | MSE | efficient_net_film | `models/efficient_net.py` |
| ResNet-18 | 1–3 | 3D ResNet-18; BasicBlock [2,2,2,2]; channels [64,64,128,256,512] | Not enabled | 64→64→128→256→512 | Linear: 512→1 | MSE | resnet18_film (2 FiLM blocks on layer2, layer3) | `models/resnet3d.py` |
| ResNet-34 | 1–3 | 3D ResNet-34; BasicBlock [3,4,6,3]; channels [64,64,128,256,512] | Not enabled | 64→64→128→256→512 | Linear: 512→1 | MSE | resnet34_film | `models/resnet3d.py` |
| ResNet-50 | 1–3 | 3D ResNet-50; Bottleneck [3,4,6,3]; channels [64,64,128,256,512] | Not enabled | 64→256→512→1024→2048 | Linear: 2048→1 | MSE | resnet50_film | `models/resnet3d.py` |
| SFCN | 1–3 | 6 conv layers; channels [32,64,128,256,256,64]; fully convolutional | 0.5 | 32→64→128→256→256→64 | Conv1×1→37 (log-softmax) | KLD | SFCN_film (FiLM on layers 1–4) | `models/sfcn.py` |
| U-Net (encoder) | 1–3 | 5-level encoder (32→64→128→256→512); MaxPool between levels | 0.5 | 32→64→128→256→512 | FC: 512→128→1 | MSE | Not available | `models/unet.py` |
| UNet_AgeSegRecon | 1–3 | 4-level encoder (32→64→128→256) + bottleneck (512); decoder with skip connections | None specified | 32→64→128→256→512 | 4 deep-supervision regressors (per encoder level) | Multitask | Not available | `models/unet.py` |
| ResNet_AgeSegRecon | 1–3 | ResNet-18 encoder; U-Net-style decoder with skip connections; base_ch=64 | None specified | 64→64→128→256→512 | 4 deep-supervision regressors (per encoder level) | Multitask | Not available | `models/resnet3d_seg.py` |

**Notes:**
- "Input Channels" depends on the modality configuration: T1=1, T2=1, T1+T2=2, ratio=1, T1+T2+ratio=3.
- All models use 3D convolutions with BatchNorm3d and ReLU activations throughout.
- Weight initialization: Kaiming normal for Conv3d layers; constant (1, 0) for BatchNorm3d (explicitly in ResNet, ChenNet; PyTorch defaults elsewhere).
- All baseline models share the same optimizer (Adam), learning rate (1e-3), batch size (16), max epochs (150), and scheduler (CosineAnnealingLR) — only the loss function differs for SFCN.
