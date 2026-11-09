import sys
from pathlib import Path
p_dir = str(Path(__file__).absolute().parents[1])
if p_dir not in sys.path: sys.path.insert(0, p_dir)

from argparse import ArgumentParser
import json

import os
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import pytorch_lightning as pl
import torchmetrics
import pandas as pd
import numpy as np
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
import random

from libs.dp_utils import num2vect_torch  
from libs.utils import scale_to_range  

from classifier.models.agenet import AgeNet, AgeNet_film
from classifier.models.basic_cnn import BasicCNN, BasicCNN_att, BasicCNN_film
from classifier.models.cbr_tiny import CbrTiny, CbrTiny_att, CbrTiny_film
from classifier.models.chen_net import ChenNet, ChenNet_att, ChenNet_film
from classifier.models.efficient_net import efficient_net, efficient_net_film
from classifier.models.resnet3d import resnet18, resnet34, resnet50, resnet101, resnet152
from classifier.models.resnet3d import resnet18_film, resnet34_film, resnet50_film, resnet101_film, resnet152_film
from classifier.models.sfcn import SFCN, SFCN_film
from classifier.models.densenet import DenseNet121, DenseNet121_film
from classifier.models.unet import UNet, UNet_AgeSegRecon
from classifier.models.resnet3d_seg import ResNet_AgeSegRecon


def dice_loss(pred, target, smooth=1e-6):
    """
    pred: logits or probabilities, shape (B, C, D, H, W)
    target: ground truth (same shape)
    """
    pred = torch.sigmoid(pred)
    
    # flatten
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    
    dice = (2 * intersection + smooth) / (union + smooth)
    loss = 1 - dice
    return loss.mean()

def loss_fn(seg_out, seg_target, recon_out, recon_target, age_preds, age_target,
            age_loss_weights=None,
            seg_bce_weight=0.5,
            seg_dice_weight=0.5,
            seg_loss_weight=1.0,
            recon_loss_weight=1.0,
            age_loss_weight=1.0):
    """
    Args:
        seg_out: segmentation logits (B, C, D, H, W)
        seg_target: segmentation GT (B, C, D, H, W)
        recon_out: reconstruction output (B, C, D, H, W)
        recon_target: input image or target for reconstruction (B, C, D, H, W)
        age_preds: list of age preds tensors, each (B,) from encoder levels
        age_target: ground truth age (B,)
        age_loss_weights: list of weights per encoder level (same length as age_preds)
        seg_bce_weight: weight for BCE loss
        seg_dice_weight: weight for Dice loss
        seg_loss_weight: overall weight for segmentation loss
        recon_loss_weight: weight for reconstruction loss
        age_loss_weight: overall weight for age prediction loss

    Returns:
        total_loss, dict_of_individual_losses
    """
    losses = {}

    # 1) Segmentation losses: BCE + Dice combined
    bce_loss = F.binary_cross_entropy_with_logits(seg_out, seg_target)
    d_loss = dice_loss(seg_out, seg_target)


    seg_loss = seg_bce_weight * bce_loss + seg_dice_weight * d_loss
    losses['seg_bce_loss'] = bce_loss
    losses['seg_dice_loss'] = d_loss
    losses['seg_loss'] = seg_loss

    recon_loss = F.mse_loss(recon_out, recon_target)
    losses['recon_loss'] = recon_loss

    if age_loss_weights is None:
        age_loss_weights = [1.0, 0.0, 0.0, 0.0]

    age_losses = []
    for pred, w in zip(age_preds, age_loss_weights):
        l = F.mse_loss(pred, age_target)
        age_losses.append(w * l)
    age_loss = sum(age_losses)
    losses['age_loss'] = age_loss

    total_loss = seg_loss_weight * seg_loss + recon_loss_weight * recon_loss + age_loss_weight * age_loss
    losses['total_loss'] = total_loss

    return total_loss, losses


def save_image(tensor, path, title="", slice_idx=None):
    """
    Save a single 2D slice (middle slice if not specified) from a 3D tensor as a PNG.
    tensor: shape (B, C, D, H, W)
    """
    tensor = tensor.detach().cpu()
    B, C, D, H, W = tensor.shape

    img = tensor[0, 0]  # shape: (D, H, W)

    if slice_idx is None:
        slice_idx = W // 2  # middle slice

    img_slice = img[:, :, slice_idx]  # shape: (H, W)

    plt.figure()
    plt.imshow(img_slice, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig(path)
    plt.close()


class LitClassifier(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # will save all kwargs to hparams.yml + store into self.hparams
        nr_channels = self.hparams.nr_channels
        # bce: one less than classes, because all 0 is first class
        nr_classes = self.hparams.nr_classes-1 if self.hparams.loss == "bce" else self.hparams.nr_classes
        spatial_dims = self.hparams.dim
        self.epochs = 0

        if self.hparams.model == 'agenet':
            self.backbone = AgeNet(in_chans=nr_channels)
        elif self.hparams.model == 'agenet_film':
            self.backbone = AgeNet_film(in_chans=nr_channels)
        elif self.hparams.model == "basiccnn":
            self.backbone = BasicCNN(in_chans=nr_channels, dropout=self.hparams.dropout)
        elif self.hparams.model == "basiccnn_att":
            self.backbone = BasicCNN_att(in_chans=nr_channels, dropout=self.hparams.dropout)
        elif self.hparams.model == "basiccnn_film":
            self.backbone = BasicCNN_film(in_chans=nr_channels, dropout=self.hparams.dropout)
        elif self.hparams.model == "cbr_tiny":
            self.backbone = CbrTiny(in_chans=nr_channels, nr_filt=self.hparams.nr_filt)
        elif self.hparams.model == "cbr_tiny_att":
            self.backbone = CbrTiny_att(in_chans=nr_channels, nr_filt=self.hparams.nr_filt)
        elif self.hparams.model == "cbr_tiny_film":
            self.backbone = CbrTiny_film(in_chans=nr_channels, nr_filt=self.hparams.nr_filt)
        elif self.hparams.model == "chen_net":
            self.backbone = ChenNet(in_chans=nr_channels, dropout=self.hparams.dropout, nr_filt=self.hparams.nr_filt)
        elif self.hparams.model == "chen_net_att":
            self.backbone = ChenNet_att(in_chans=nr_channels, dropout=self.hparams.dropout, nr_filt=self.hparams.nr_filt)
        elif self.hparams.model == "chen_net_film":
            self.backbone = ChenNet_film(in_chans=nr_channels, dropout=self.hparams.dropout, nr_filt=self.hparams.nr_filt)
        elif self.hparams.model == "densenet":
            self.backbone = DenseNet121(in_channels=nr_channels)
        elif self.hparams.model == "densenet_film":
            self.backbone = DenseNet121_film(in_channels=nr_channels)
        elif self.hparams.model == "efficientnet":
            self.backbone = efficient_net("efficientnet-b0", in_channels=nr_channels, out_channels=nr_classes)
        elif self.hparams.model == "efficientnet_film":
            self.backbone = efficient_net_film("efficientnet-b0", in_channels=nr_channels, out_channels=nr_classes)

        elif self.hparams.model == 'resnet18':
            self.backbone = resnet18(in_chans=nr_channels)
        elif self.hparams.model == 'resnet34':
            self.backbone = resnet34(in_chans=nr_channels)
        elif self.hparams.model == 'resnet50':
            self.backbone = resnet50(in_chans=nr_channels)
        elif self.hparams.model == 'resnet101':
            self.backbone = resnet101(in_chans=nr_channels)
        elif self.hparams.model == 'resnet152':
            self.backbone = resnet152(in_chans=nr_channels)
        elif self.hparams.model == 'resnet18_film':
            self.backbone = resnet18_film(in_chans=nr_channels, option=2)
        elif self.hparams.model == 'resnet34_film':
            self.backbone = resnet34_film(in_chans=nr_channels, option=2)
        elif self.hparams.model == 'resnet50_film':
            self.backbone = resnet50_film(in_chans=nr_channels, option=2)
        elif self.hparams.model == 'resnet101_film':
            self.backbone = resnet101_film(in_chans=nr_channels, option=2)
        elif self.hparams.model == 'resnet152_film':
            self.backbone = resnet152_film(in_chans=nr_channels, option=2)

        elif self.hparams.model == 'sfcn':
            self.backbone = SFCN(in_chans=nr_channels)
        elif self.hparams.model == 'sfcn_film':
            self.backbone = SFCN_film(in_chans=nr_channels)
        elif self.hparams.model == 'unet':
            self.backbone = UNet(in_channels=nr_channels)
        elif self.hparams.model == 'unet_seg':
            self.backbone = UNet_AgeSegRecon(in_channels=nr_channels)
        elif self.hparams.model == 'resnet_seg':
            self.backbone = ResNet_AgeSegRecon(in_channels=nr_channels)
                
        else:
            import timm
            self.backbone = timm.create_model(self.hparams.model, pretrained=self.hparams.pretrain,
                                              num_classes=nr_classes, in_chans=nr_channels)

        if self.hparams.loss.startswith("mse"):
            self.loss_func = torch.nn.MSELoss()
        elif self.hparams.loss.startswith("mae"):
            self.loss_func = torch.nn.L1Loss()
        elif self.hparams.loss == "kld":
            self.loss_func = torch.nn.KLDivLoss(reduction='sum')

        self.max_f1 = 0

    def forward(self, x):
        return self.backbone(x)  # use forward for inference/predictions


    #     # log each component loss


    def training_step(self, batch, batch_idx):
        recon_target, age_target, wm, icv, ratio, seg_target = batch

        if self.current_epoch == 0 and batch_idx == 0:
            print(f"x.shape: {recon_target.shape}")

        seg_out, recon_out, age_preds = self.backbone(recon_target) 
        total_loss, losses = loss_fn(seg_out, seg_target, recon_out, recon_target, age_preds, age_target,
                                        self.hparams.age_loss_weights,
                                        self.hparams.seg_bce_weight,
                                        self.hparams.seg_dice_weight,
                                        self.hparams.seg_loss_weight,
                                        self.hparams.recon_loss_weight,
                                        self.hparams.age_loss_weight)

        self.log('loss/train', total_loss, on_step=True, on_epoch=True)
        for k, v in losses.items():
            self.log(f'loss/{k}', v, on_step=True, on_epoch=True)

        # === Save images for debugging ===


        return total_loss


    def validation_step(self, batch, batch_idx):
        self.epochs += 1
        recon_target, age_target, wm, icv, ratio, seg_target = batch

        seg_out, recon_out, age_preds = self.backbone(recon_target) 
        total_loss, losses = loss_fn(seg_out, seg_target, recon_out, recon_target, age_preds, age_target,
                                        self.hparams.age_loss_weights,
                                        self.hparams.seg_bce_weight,
                                        self.hparams.seg_dice_weight,
                                        self.hparams.seg_loss_weight,
                                        self.hparams.recon_loss_weight,
                                        self.hparams.age_loss_weight)

        y_hat = age_preds[0].detach()  # last feature

        self.log('loss/val', total_loss, on_step=False, on_epoch=True)
            
        return {"y": age_target, "y_hat": y_hat}

    def validation_epoch_end(self, outputs):
        current_lr = torch.tensor(self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log("loss/current_lr", current_lr)


    def test_step(self, batch, batch_idx):
        
        recon_target, age_target, wm, icv, ratio, seg_target = batch

        seg_out, recon_out, age_preds = self.backbone(recon_target) 
        y_hat = age_preds[0].detach()  # last feature


        return {"y_raw": age_target, "y_hat_raw": y_hat}

    def test_epoch_end(self, outputs):

        y_hat_raw = torch.cat([step['y_hat_raw'] for step in outputs])
        y_raw = torch.cat([step['y_raw'] for step in outputs])
        
        mse = torchmetrics.functional.mean_squared_error(y_hat_raw, y_raw).item()
        mae = torchmetrics.functional.mean_absolute_error(y_hat_raw, y_raw).item()  # identical to: torch.abs(y_hat_raw-y_raw).mean()
        pearsonr = torchmetrics.functional.pearson_corrcoef(y_hat_raw, y_raw).item()
        spearmanr = torchmetrics.functional.spearman_corrcoef(y_hat_raw, y_raw).item()
        r2 = torchmetrics.functional.r2_score(y_hat_raw, y_raw).item()


        self.logger.experiment.log({
            "MSE": mse,
            "MAE": mae,
            "RMSE": np.sqrt(mse),
            "PearsonR": pearsonr,
            "SpearmanR": spearmanr,
            "R2": r2,
        }, step=self.current_epoch)

        # update hp_metric
        val = self.trainer.checkpoint_callback.best_model_score
        self.log('hp_metric', val)  # mse
        self.log('metric_mae', mae)  
        self.log('metric_rmse', np.sqrt(mse)) 
        self.log('metric_pearsonr', pearsonr) 
        self.log('metric_spearmanr', spearmanr) 
        self.log('metric_r2', r2) 

        ### SAVE RESULTS !!!
        output_path = Path("$PATH/RI_anatomical_prior/result_0701_seg")

        model_dir = output_path / self.hparams.model 
        model_dir.mkdir(parents=True, exist_ok=True)


        fn = model_dir / f"{self.hparams.fold}_{self.hparams.exp_name}_{self.hparams.age_loss_weights}_{self.hparams.seg_bce_weight}_{self.hparams.seg_dice_weight}_{self.hparams.seg_loss_weight}_{self.hparams.recon_loss_weight}_{self.hparams.age_loss_weight}_results.csv"

        gts = y_raw.detach().cpu().numpy()
        preds = y_hat_raw.detach().cpu().numpy()
        df = pd.DataFrame(data=np.array([gts, preds]).T, columns=["age", "age_predicted"])
        df.to_csv(fn, index=False, float_format="%.2f")

        metrics = {
            "mse": mse,
            "mae": mae,
            "rmse": np.sqrt(mse),
            "pearsonr": pearsonr,
            "spearmanr": spearmanr,
            "r2": r2
        }

        metrics_rounded = {k: round(v, 3) if not isinstance(v, list) else [round(e, 3) for e in v] for k, v in metrics.items()}

        json_fn = model_dir / f"{self.hparams.fold}_{self.hparams.exp_name}_{self.hparams.age_loss_weights}_{self.hparams.seg_bce_weight}_{self.hparams.seg_dice_weight}_{self.hparams.seg_loss_weight}_{self.hparams.recon_loss_weight}_{self.hparams.age_loss_weight}_metrics.json"
        
        with open(json_fn, "w") as json_file:
            json.dump(metrics_rounded, json_file, indent=4)

        print(f"Results saved as {fn} and metrics saved as {json_fn}")


    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate,
                                        weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate,
                                        momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.weight_decay)

        if self.hparams.scheduler == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5,
                                          cooldown=0, min_lr=1e-8)
            scheduler_config = {
                'scheduler': scheduler,
                'reduce_on_plateau': True,  # For ReduceLROnPlateau scheduler
                'monitor': 'loss/val',  # Metric for ReduceLROnPlateau to monitor
            }
        elif self.hparams.scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, self.hparams.max_epochs)
            scheduler_config = {'scheduler': scheduler}
            
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler_config,
            'monitor': 'loss/val'
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--my_test', type=float, default=1e-4)
        return parser
