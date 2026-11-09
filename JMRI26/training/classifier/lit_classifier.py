import sys
from pathlib import Path
p_dir = str(Path(__file__).absolute().parents[1])
if p_dir not in sys.path: sys.path.insert(0, p_dir)

from argparse import ArgumentParser
import json

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
from classifier.models.unet import UNet


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

    def training_step(self, batch, batch_idx):
        x, y, wm, icv, ratio = batch
        if self.current_epoch == 0 and batch_idx == 0:
            print(f"x.shape: {x.shape}")

        if self.hparams.add_option == 'wm':
            y_hat = self.backbone(x, wm)
        elif self.hparams.add_option == 'icv':
            y_hat = self.backbone(x, icv)
        elif self.hparams.add_option == 'ratio':
            y_hat = self.backbone(x, ratio)
        elif self.hparams.add_option == 'both':
            y_hat = self.backbone(x, (wm, icv))
        else:
            y_hat = self.backbone(x) 

        if self.hparams.loss.startswith("mse"):
            y_hat = y_hat[:, 0] 
        elif self.hparams.loss.startswith("mae"):
            y_hat = y_hat[:, 0] 
        else:
            y_hat

        ### CUSTOM
        if self.hparams.loss != "kld":
            loss = self.loss_func(y_hat, y)
        else:
            y_label, bc = num2vect_torch(y, [-1, 36], 1, 1)
            y_label += 1e-16
            n = y_label.shape[0]
            loss = self.loss_func(y_hat, y_label) / n

        y_hat = y_hat.detach()

        if self.hparams.loss == "kld":
            y_hat = (torch.exp(y_hat).to(torch.float32)) @ (bc.to(torch.float32))

        self.log('loss/train', loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        self.epochs += 1
        x, y, wm, icv, ratio = batch

        if self.hparams.add_option == 'wm':
            y_hat_prob = self.backbone(x, wm)
        elif self.hparams.add_option == 'icv':
            y_hat_prob = self.backbone(x, icv)
        elif self.hparams.add_option == 'ratio':
            y_hat_prob = self.backbone(x, ratio)
        elif self.hparams.add_option == 'both':
            y_hat_prob = self.backbone(x, (wm, icv))
        else:
            y_hat_prob = self.backbone(x)  # [bs, nr_classes]

        if self.hparams.loss.startswith("mse"):
            y_hat_prob = y_hat_prob[:, 0] 
        elif self.hparams.loss.startswith("mae"):
            y_hat_prob = y_hat_prob[:, 0] 
        else:
            y_hat_prob
        
        ### CUSTOM
        if self.hparams.loss != "kld":
            loss = self.loss_func(y_hat_prob, y)
        else:
            y_label, bc = num2vect_torch(y, [-1, 36], 1, 1)
            y_label += 1e-16
            n = y_label.shape[0]
            loss = self.loss_func(y_hat_prob, y_label) / n

        y_hat_prob = y_hat_prob.detach()
        y_hat = y_hat_prob

        if self.hparams.loss == "kld":
            y_hat = (torch.exp(y_hat).to(torch.float32)) @ (bc.to(torch.float32))

        self.log('loss/val', loss, on_step=False, on_epoch=True)
            
        return {"y": y, "y_hat": y_hat}

    def validation_epoch_end(self, outputs):
        current_lr = torch.tensor(self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log("loss/current_lr", current_lr)


    def test_step(self, batch, batch_idx):
        
        x, y_raw, wm, icv, ratio = batch

        if self.hparams.add_option == 'wm':
            y_hat_raw = self.backbone(x, wm)
        elif self.hparams.add_option == 'icv':
            y_hat_raw = self.backbone(x, icv)
        elif self.hparams.add_option == 'ratio':
            y_hat_raw = self.backbone(x, ratio)
        elif self.hparams.add_option == 'both':
            y_hat_raw = self.backbone(x, (wm, icv))
        else:
            y_hat_raw = self.backbone(x)  # [bs, nr_classes]


        if self.hparams.loss.startswith("mse"):
            y_hat_raw = y_hat_raw[:, 0] 
        elif self.hparams.loss.startswith("mae"):
            y_hat_raw = y_hat_raw[:, 0] 
        else:
            y_hat_raw

        y = y_raw

        ### CUSTOM to discrete number


        y_hat_raw = y_hat_raw.detach()
        y_hat = y_hat_raw

        if self.hparams.loss == "kld":
            y_label, bc = num2vect_torch(y, [-1, 36], 1, 1)
            y_hat_raw = (torch.exp(y_hat_raw).to(torch.float32)) @ (bc.to(torch.float32))
            y_hat = (torch.exp(y_hat).to(torch.float32)) @ (bc.to(torch.float32))

        return {"y_raw": y_raw, "y_hat_raw": y_hat_raw, "y": y, "y_hat": y_hat}

    def test_epoch_end(self, outputs):

        y_hat_raw = torch.cat([step['y_hat_raw'] for step in outputs])
        y_raw = torch.cat([step['y_raw'] for step in outputs])
        y_hat = torch.cat([step['y_hat'] for step in outputs])
        y = torch.cat([step['y'] for step in outputs])
        
        mse = torchmetrics.functional.mean_squared_error(y_hat_raw, y_raw).item()
        # This is very close to run_inference.py -> mae, but not completely identical -> why?
        mae = torchmetrics.functional.mean_absolute_error(y_hat_raw, y_raw).item()  # identical to: torch.abs(y_hat_raw-y_raw).mean()
        pearsonr = torchmetrics.functional.pearson_corrcoef(y_hat_raw, y_raw).item()
        spearmanr = torchmetrics.functional.spearman_corrcoef(y_hat_raw, y_raw).item()
        r2 = torchmetrics.functional.r2_score(y_hat_raw, y_raw).item()

        self.logger.experiment.add_text("result", f"MSE: {mse:.4f}, MAE: {mae:.4f}, PEAR: {pearsonr:.4f}, SPEAR: {spearmanr:.4f}, R2: {r2:.4f}", self.current_epoch)

        # update hp_metric
        val = self.trainer.checkpoint_callback.best_model_score
        self.log('hp_metric', val)  # mse
        if self.hparams.loss.startswith("mse"):
            self.log('metric_mae', mae)  
            self.log('metric_rmse', np.sqrt(mse)) 
            self.log('metric_pearsonr', pearsonr) 
            self.log('metric_spearmanr', spearmanr) 
            self.log('metric_r2', r2) 
        elif self.hparams.loss.startswith("mae"):
            self.log('metric_mae', mae)  
            self.log('metric_rmse', np.sqrt(mse)) 
            self.log('metric_pearsonr', pearsonr) 
            self.log('metric_spearmanr', spearmanr) 
            self.log('metric_r2', r2) 
        elif self.hparams.loss == "kld":
            self.log('metric_mae', mae)  
            self.log('metric_rmse', np.sqrt(mse)) 
            self.log('metric_pearsonr', pearsonr) 
            self.log('metric_spearmanr', spearmanr) 
            self.log('metric_r2', r2) 
        else:
            self.log('hp_metric')

        ### SAVE RESULTS !!!
        output_path = Path("$PATH/RI_anatomical_prior/result_0619")

        model_dir = output_path / self.hparams.model 
        model_dir.mkdir(parents=True, exist_ok=True)

        fn = model_dir / f"{self.hparams.fold}_{self.hparams.exp_name}_{self.hparams.data_option}_{self.hparams.add_option}_results.csv"

        gts = y.detach().cpu().numpy()
        preds = y_hat.detach().cpu().numpy()
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

        json_fn = model_dir / f"{self.hparams.fold}_{self.hparams.exp_name}_{self.hparams.data_option}_{self.hparams.add_option}_metrics.json"
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
