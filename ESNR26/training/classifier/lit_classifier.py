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

from libs.dp_utils import num2vect_torch  
from libs.utils import scale_to_range  

from classifier.models.global_local_attention_3d import GlobalLocalBrainAge
from classifier.models.resnet_info_add import ResNet, ResNet_add, FCNet, ResNet_add_FiLM, ResNet_add_FiLM2, ResNet_add_FiLM3, ResNet_add_FiLM4
from classifier.models.resnet_info_add import ResNet_CrossAttention, ResNet_CrossAttention2
from classifier.models.dual_path import DualPathBrainAge, DualPathBrainAge_info, DualPathBrainAge_origin, DualPathBrainAge_origin_info
from classifier.models.sfcn import SFCN
from classifier.utils import confusion_matrix_to_figure

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True  # CuDNN deterministic
torch.backends.cudnn.benchmark = False  # CuDNN benchmark off
torch.random.manual_seed(seed)
torch.use_deterministic_algorithms(True)
pl.seed_everything(seed)

class LitClassifier(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # will save all kwargs to hparams.yml + store into self.hparams
        nr_channels = self.hparams.nr_channels
        # bce: one less than classes, because all 0 is first class
        nr_classes = self.hparams.nr_classes-1 if self.hparams.loss == "bce" else self.hparams.nr_classes
        spatial_dims = self.hparams.dim
        self.epochs = 0

        if self.hparams.model == 'glo_loc':
            if "15" in self.hparams.exp_name:
                self.backbone = GlobalLocalBrainAge(nr_channels,
                            patch_size=32,
                            patch_size_z=32,
                            step=16,
                            nblock=6,
                            backbone='vgg8')
            elif "06" in self.hparams.exp_name:
                self.backbone = GlobalLocalBrainAge(nr_channels,
                            patch_size=64,
                            patch_size_z=16,
                            step=32,
                            nblock=6,
                            backbone='vgg8')
        elif self.hparams.model == 'resnet':
            self.backbone = ResNet(in_chans=nr_channels)
        elif self.hparams.model == 'resnet_add':
            if self.hparams.add_option == 'both':
                self.backbone = ResNet_add(in_chans=nr_channels, option=2)
            else:
                self.backbone = ResNet_add(in_chans=nr_channels, option=1)

        elif self.hparams.model == 'resnet_film':
            if self.hparams.add_option == 'both':
                self.backbone = ResNet_add_FiLM(in_chans=nr_channels, option=2)
            else:
                self.backbone = ResNet_add_FiLM(in_chans=nr_channels, option=1)
        elif self.hparams.model == 'resnet_film2':      ## both
            self.backbone = ResNet_add_FiLM2(in_chans=nr_channels)
        elif self.hparams.model == 'resnet_film3':      ## both
            self.backbone = ResNet_add_FiLM3(in_chans=nr_channels)
        elif self.hparams.model == 'resnet_film4':      ## both
            self.backbone = ResNet_add_FiLM4(in_chans=nr_channels)
        elif self.hparams.model == 'resnet_att':      
            if self.hparams.add_option == 'both':
                self.backbone = ResNet_CrossAttention(in_chans=nr_channels, option=2)
            else:
                self.backbone = ResNet_CrossAttention(in_chans=nr_channels, option=1)
        elif self.hparams.model == 'resnet_att2':      
            self.backbone = ResNet_CrossAttention2(in_chans=nr_channels, option=2)  ## both only

        #### ---- ####

        elif self.hparams.model == 'dual_origin':      
            self.backbone = DualPathBrainAge_origin()  
        elif self.hparams.model == 'dual_cse_origin':      
            self.backbone = DualPathBrainAge_origin(use_se_local='cse')
        elif self.hparams.model == 'dual_sse':      
            self.backbone = DualPathBrainAge_origin(use_se_local='sse')
        elif self.hparams.model == 'dual_csse':      
            self.backbone = DualPathBrainAge_origin(use_se_local='csse')
        elif self.hparams.model == 'dual_pse':      
            self.backbone = DualPathBrainAge_origin(use_se_local='pse')
        elif self.hparams.model == 'dual_se':      
            self.backbone = DualPathBrainAge_origin(use_se_local='se')

        elif self.hparams.model == 'dual_info':      
            if self.hparams.add_option == 'both':
                self.backbone = DualPathBrainAge_origin_info(use_aux_info=2)  ## both only
            else:
                self.backbone = DualPathBrainAge_origin_info(use_aux_info=1)
        elif self.hparams.model == 'dual_cse_info':      
            if self.hparams.add_option == 'both':
                self.backbone = DualPathBrainAge_origin_info(use_aux_info=2, use_se_local='cse') 
            else:
                self.backbone = DualPathBrainAge_origin_info(use_aux_info=1, use_se_local='cse')
        elif self.hparams.model == 'dual_sse_info':      
            if self.hparams.add_option == 'both':
                self.backbone = DualPathBrainAge_origin_info(use_aux_info=2, use_se_local='sse') 
            else:
                self.backbone = DualPathBrainAge_origin_info(use_aux_info=1, use_se_local='sse')
        elif self.hparams.model == 'dual_csse_info':      
            if self.hparams.add_option == 'both':
                self.backbone = DualPathBrainAge_origin_info(use_aux_info=2, use_se_local='csse') 
            else:
                self.backbone = DualPathBrainAge_origin_info(use_aux_info=1, use_se_local='csse')
        elif self.hparams.model == 'dual_pse_info':      
            if self.hparams.add_option == 'both':
                self.backbone = DualPathBrainAge_origin_info(use_aux_info=2, use_se_local='pse') 
            else:
                self.backbone = DualPathBrainAge_origin_info(use_aux_info=1, use_se_local='pse')
        elif self.hparams.model == 'dual_se_info':      
            if self.hparams.add_option == 'both':
                self.backbone = DualPathBrainAge_origin_info(use_aux_info=2, use_se_local='se') 
            else:
                self.backbone = DualPathBrainAge_origin_info(use_aux_info=1, use_se_local='se')

        #### ---- ####

        elif self.hparams.model == 'dual':      
            self.backbone = DualPathBrainAge()  
        elif self.hparams.model == 'dual_cse':      
            self.backbone = DualPathBrainAge(use_se_local='cse')
        elif self.hparams.model == 'dual_sse':      
            self.backbone = DualPathBrainAge(use_se_local='sse')
        elif self.hparams.model == 'dual_csse':      
            self.backbone = DualPathBrainAge(use_se_local='csse')
        elif self.hparams.model == 'dual_pse':      
            self.backbone = DualPathBrainAge(use_se_local='pse')
        elif self.hparams.model == 'dual_se':      
            self.backbone = DualPathBrainAge(use_se_local='se')

        elif self.hparams.model == 'dual_info':      
            if self.hparams.add_option == 'both':
                self.backbone = DualPathBrainAge_info(use_aux_info=2)  ## both only
            else:
                self.backbone = DualPathBrainAge_info(use_aux_info=1)
        elif self.hparams.model == 'dual_cse_info':      
            if self.hparams.add_option == 'both':
                self.backbone = DualPathBrainAge_info(use_aux_info=2, use_se_local='cse') 
            else:
                self.backbone = DualPathBrainAge_info(use_aux_info=1, use_se_local='cse')
        elif self.hparams.model == 'dual_sse_info':      
            if self.hparams.add_option == 'both':
                self.backbone = DualPathBrainAge_info(use_aux_info=2, use_se_local='sse') 
            else:
                self.backbone = DualPathBrainAge_info(use_aux_info=1, use_se_local='sse')
        elif self.hparams.model == 'dual_csse_info':      
            if self.hparams.add_option == 'both':
                self.backbone = DualPathBrainAge_info(use_aux_info=2, use_se_local='csse') 
            else:
                self.backbone = DualPathBrainAge_info(use_aux_info=1, use_se_local='csse')
        elif self.hparams.model == 'dual_pse_info':      
            if self.hparams.add_option == 'both':
                self.backbone = DualPathBrainAge_info(use_aux_info=2, use_se_local='pse') 
            else:
                self.backbone = DualPathBrainAge_info(use_aux_info=1, use_se_local='pse')
        elif self.hparams.model == 'dual_se_info':      
            if self.hparams.add_option == 'both':
                self.backbone = DualPathBrainAge_info(use_aux_info=2, use_se_local='se') 
            else:
                self.backbone = DualPathBrainAge_info(use_aux_info=1, use_se_local='se')


        elif self.hparams.model == 'fcnet':
            if self.hparams.add_option == 'both':
                self.backbone = FCNet(option=2)
            else:
                self.backbone = FCNet(option=1)
        
        else:
            import timm
            self.backbone = timm.create_model(self.hparams.model, pretrained=self.hparams.pretrain,
                                              num_classes=nr_classes, in_chans=nr_channels)

        if self.hparams.loss == "ce":
            weights = None
            self.loss_func = torch.nn.CrossEntropyLoss(weight=weights)
        elif self.hparams.loss == "bce":
            self.loss_func = torch.nn.BCEWithLogitsLoss()  # min: 0, max: +inf
        elif self.hparams.loss.startswith("mse"):
            self.loss_func = torch.nn.MSELoss()
        elif self.hparams.loss.startswith("mae"):
            self.loss_func = torch.nn.L1Loss()
        elif self.hparams.loss == "kld":
            self.loss_func = torch.nn.KLDivLoss(reduction='sum')

        #     # different results depending on num_classes:
        #     # num_classes=1: like sklearn f1_score binary
        #     # num_classes=2: like sklearn f1_score micro   
        #     # (for Recall&Precision + lightning>=1.2.0 only num_classes=2 is working)

        self.max_f1 = 0

    def forward(self, x):
        return self.backbone(x)  # use forward for inference/predictions


    def _decode_y(self, y):
        if self.hparams.loss == "ce":
            label = y
        elif self.hparams.loss == "bce":
            label = y.sum(1).int()  # works for ordinal classes as well as binary
        elif self.hparams.loss == "mse":
            label = y
        elif self.hparams.loss == "mae":
            label = y
        elif self.hparams.loss == "kld":
            label = y
        return label

    def _y_hat_to_prob(self, y):
        if self.hparams.loss == "ce":
            label = y.detach().softmax(dim=1)
        elif self.hparams.loss == "bce":
            label = y.detach().sigmoid()
        elif self.hparams.loss == "mse":
            label = y.detach()  # [bs], float16
        elif self.hparams.loss == "mae":
            label = y.detach()  # [bs], float16
        elif self.hparams.loss == "mse_cat":
            label = y.detach()
        elif self.hparams.loss == "kld":
            label = y.detach()
        return label

    def _decode_y_hat(self, y):
        if self.hparams.loss == "ce":
            # Add softmax before argmax? Only relevant for loss and there it is added automatically; 
            # for argmax results is identical (?)
            label = torch.argmax(y.detach(), dim=1)
        elif self.hparams.loss == "bce":
            label = y.detach().sum(1).round().int()  # works for ordinal classes as well as binary
        elif self.hparams.loss == "mse":
            label = y.detach()  # [bs], float16
        elif self.hparams.loss == "mae":
            label = y.detach()  # [bs], float16
        elif self.hparams.loss == "kld":
            label = y.detach() 
        return label

    def training_step(self, batch, batch_idx):
        x, y, wm, icv, ratio = batch
        if self.current_epoch == 0 and batch_idx == 0:
            print(f"x.shape: {x.shape}")

        # if self.hparams.model 
        if self.hparams.model == 'resnet_deep':
            y_hat, aux_out2, aux_out3, aux_out4 = self.backbone(x)
            main_loss = self.loss_func(y_hat[:, 0], y)
            aux2_loss = self.loss_func(aux_out2[:, 0], y)
            aux3_loss = self.loss_func(aux_out3[:, 0], y)
            aux4_loss = self.loss_func(aux_out4[:, 0], y)
            loss = (
                1.0 * main_loss +
                0.1 * aux2_loss +
                0.2 * aux3_loss +
                0.3 * aux4_loss
            )
        elif self.hparams.model == 'resnet_att_cross_deep':
            y_hat, aux_out1, aux_out2, aux_out3, aux_out4 = self.backbone(x)
            main_loss = self.loss_func(y_hat[:, 0], y)
            aux1_loss = self.loss_func(aux_out1[:, 0], y)
            aux2_loss = self.loss_func(aux_out2[:, 0], y)
            aux3_loss = self.loss_func(aux_out3[:, 0], y)
            aux4_loss = self.loss_func(aux_out4[:, 0], y)
            loss = (
                1.0 * main_loss +
                0.1 * aux1_loss + 
                0.2 * aux2_loss +
                0.3 * aux3_loss +
                0.4 * aux4_loss
            )

        else:
            if self.hparams.add_option == 'wm':
                y_hat = self.backbone(x, wm)
            elif self.hparams.add_option == 'icv':
                y_hat = self.backbone(x, icv)
            elif self.hparams.add_option == 'ratio':
                y_hat = self.backbone(x, ratio)
            elif self.hparams.add_option == 'both':
                y_hat = self.backbone(x, (wm, icv))
            else:
                y_hat = self.backbone(x)  # [bs, nr_classes]


            if self.hparams.model == 'glo_loc':
                outlist = y_hat
                target = y
                    
                target = target.view(-1, 1)  # ensure target shape matches predictions
                
                # Global loss
                global_pred = outlist[0]
                loss_global = torch.mean(torch.abs(global_pred - target))

                # Local losses
                loss_local = 0.0
                for local_pred in outlist[1:]:
                    loss_local += torch.mean(torch.abs(local_pred - target)) 
                # Total loss = global + local sum
                loss = 5 * loss_global + loss_local
                
                ## for not error debugging
                y_hat = y_hat[0]


            else:
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

        y = self._decode_y(y)
        y_hat = self._y_hat_to_prob(y_hat)
        y_hat = self._decode_y_hat(y_hat)

        if self.hparams.loss == "kld":
            y_hat = (torch.exp(y_hat).to(torch.float32)) @ (bc.to(torch.float32))

        self.log('loss/train', loss, on_step=False, on_epoch=True)
        if self.hparams.loss == "ce":
            self.log('f1/train', self.f1(y_hat, y), on_step=False, on_epoch=True)

        if self.hparams.log_images:
            print("WARNING: Logging images")
            nr_imgs = 8  # nr of imgs from batch
            if x.shape[2] > 200:
                x = F.interpolate(x, scale_factor=0.5, mode="nearest")  # reduce img size to make event file smaller
            # remove the rescaling to properly display binary masks
            img = 255 - scale_to_range(x[:nr_imgs], (0, 254))  # [nr_imgs, channels, x, y, (z)]
            if self.hparams.dim == 2:
                # Show channels underneath each other
                img = img.reshape([img.shape[0], 1, img.shape[1]*img.shape[2], img.shape[3]])
            else:
                img = img[:, :, :, :, img.shape[-1] // 2]  # select middle slice in z
                # Show channels underneath each other
                img = img.reshape([img.shape[0], 1, img.shape[1]*img.shape[2], img.shape[3]])
            self.logger.experiment.add_images('data/train', img, dataformats='NCHW')

        return loss

    def validation_step(self, batch, batch_idx):
        self.epochs += 1
        x, y, wm, icv, ratio = batch
        # if self.hparams.model 
        if self.hparams.model == 'resnet_deep':
            y_hat_prob, aux_out2, aux_out3, aux_out4 = self.backbone(x)
            y_hat_prob = y_hat_prob[:, 0]
        elif self.hparams.model == 'resnet_att_cross_deep':
            y_hat_prob, _, _, _, _ = self.backbone(x)
            y_hat_prob = y_hat_prob[:, 0]
        
        else:
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

            if self.hparams.model == 'glo_loc':
                outlist = y_hat_prob
                target = y
                    
                target = target.view(-1, 1)  # ensure target shape matches predictions
                
                # Global loss
                global_pred = outlist[0]
                loss_global = torch.mean(torch.abs(global_pred - target))

                # Local losses
                loss_local = sum(torch.mean(torch.abs(pred - target)) for pred in outlist[1:]) / (len(outlist) - 1)

                # Total loss = global + local sum
                loss = loss_global + loss_local
                
                ## for not error debugging
                y_hat_prob = y_hat_prob[0]
            else:
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

        y = self._decode_y(y)
        y_hat_prob = self._y_hat_to_prob(y_hat_prob)
        y_hat = self._decode_y_hat(y_hat_prob)

        if self.hparams.loss == "kld":
            y_hat = (torch.exp(y_hat).to(torch.float32)) @ (bc.to(torch.float32))

        self.log('loss/val', loss, on_step=False, on_epoch=True)
        if self.hparams.loss == "ce":
            # This is identical results to first aggregating all y and y_hat in validation_epoch_end and
            # then calculating f1. Not sure how it works. Should give different result when calculating f1
            # per batch and then mean?
            self.log('f1/val', self.f1(y_hat, y), on_step=False, on_epoch=True)
            self.log('misc_val/accuracy', self.accuracy_m(y_hat, y), on_step=False, on_epoch=True)
            self.log('misc_val/sensitivity', self.recall_m(y_hat, y), on_step=False, on_epoch=True)
            self.log('misc_val/precision', self.precision_m(y_hat, y), on_step=False, on_epoch=True)
            
        return {"y": y, "y_hat": y_hat}

    def validation_epoch_end(self, outputs):
        current_lr = torch.tensor(self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log("loss/current_lr", current_lr)

        if self.hparams.loss == "ce":
            # Important note: this does not correctly work for Multi-GPU training
            preds = torch.cat([step['y_hat'] for step in outputs])
            targets = torch.cat([step['y'] for step in outputs])
            f1 = torchmetrics.functional.f1_score(preds, targets, num_classes=self.hparams.nr_classes_config).item()
            if self.current_epoch > 0:  # skip validation sanity check (and first epoch)
                self.max_f1 = max(self.max_f1, f1)
            self.log("f1/val_max", self.max_f1)

    def test_step(self, batch, batch_idx):
        
        x, y_raw, wm, icv, ratio = batch
        if self.hparams.model == 'resnet_deep':
            y_hat_raw, aux_out2, aux_out3, aux_out4 = self.backbone(x)
        elif self.hparams.model == 'resnet_att_cross_deep':
            y_hat_raw, _, _, _, _ = self.backbone(x)
        
        else:

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
        
            if self.hparams.model == 'glo_loc':
                
                y = self._decode_y(y_raw)
                y_hat_raw = torch.stack(y_hat_raw, dim=0)  # shape: (N_patches, B, 1)
                y_hat_raw = y_hat_raw.mean(dim=0)[:, 0]
                y_hat = y_hat_raw

            else:
                if self.hparams.loss != "kld":
                    y_hat_raw = torch.round(y_hat_raw)

                if self.hparams.loss.startswith("mse"):
                    y_hat_raw = y_hat_raw[:, 0] 
                elif self.hparams.loss.startswith("mae"):
                    y_hat_raw = y_hat_raw[:, 0] 
                else:
                    y_hat_raw

                y = self._decode_y(y_raw)

                ### CUSTOM to discrete number


                y_hat_raw = self._y_hat_to_prob(y_hat_raw)  ## -> performance evaluation
                y_hat = self._decode_y_hat(y_hat_raw)

                if self.hparams.loss == "kld":
                    y_label, bc = num2vect_torch(y, [-1, 36], 1, 1)
                    y_hat_raw = (torch.exp(y_hat_raw).to(torch.float32)) @ (bc.to(torch.float32))
                    y_hat = (torch.exp(y_hat).to(torch.float32)) @ (bc.to(torch.float32))
                    y_hat_raw = torch.round(y_hat_raw)
                    y_hat = torch.round(y_hat)


        if self.hparams.log_images:
            print("WARNING: Logging images")
            x = x[y_hat != y]  # select wrongly classified samples
            print(f"Found {x.shape[0]} wrong classified images in batch.")
            nr_imgs = 5  # nr of imgs from batch
            x = F.interpolate(x, scale_factor=0.5, mode="nearest")  # reduce img size to make event file smaller
            img = 255 - scale_to_range(x[:nr_imgs], (0, 255))
            if self.hparams.dim == 2:
                # Show channels underneath each other
                img = img.reshape([img.shape[0], 1, img.shape[1]*img.shape[2], img.shape[3]])
            else:
                img = img[:, :1, :, :, img.shape[-1] // 2] 
            self.logger.experiment.add_images('wrongly_classified', img, dataformats='NCHW', global_step=batch_idx)

        return {"y_raw": y_raw, "y_hat_raw": y_hat_raw, "y": y, "y_hat": y_hat}

    def test_epoch_end(self, outputs):
        # All the evaluations we want to do on the final model. This is also run on the validation set.
        # 
        # Important note: This only works properly for single GPU training. For Multi-GPU we would have
        # to aggregate results accross GPUs. (e.g. try to use the metrics class API)

        y_hat_raw = torch.cat([step['y_hat_raw'] for step in outputs])
        y_raw = torch.cat([step['y_raw'] for step in outputs])
        y_hat = torch.cat([step['y_hat'] for step in outputs])
        y = torch.cat([step['y'] for step in outputs])

        if self.hparams.loss == "ce":
            # Plot confusion matrix
            confusion_matrix = torchmetrics.functional.confusion_matrix(y_hat, y, num_classes=self.hparams.nr_classes_config)
            fig_ = confusion_matrix_to_figure(confusion_matrix.detach().cpu().numpy(), self.hparams.nr_classes_config)
            self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

            # Print f1 per class
            s = ""
            f1_all = torchmetrics.functional.f1_score(y_hat, y, num_classes=self.hparams.nr_classes_config).item()
            s += f"All: {f1_all:.3f}\n  "
            for class_idx in range(self.hparams.nr_classes_config):
                # different results depending on num_classes:
                f1 = torchmetrics.functional.f1_score((y_hat == class_idx).int(),
                                                (y == class_idx).int(), num_classes=2).item()
                s += f"{class_idx}: {f1:.3f}\n  "
            self.logger.experiment.add_text("F1 per class", s, self.current_epoch)
        
        if self.hparams.loss.startswith("mse"):
            mse = torchmetrics.functional.mean_squared_error(y_hat_raw, y_raw).item()
            # This is very close to run_inference.py -> mae, but not completely identical -> why?
            mae = torchmetrics.functional.mean_absolute_error(y_hat_raw, y_raw).item()  # identical to: torch.abs(y_hat_raw-y_raw).mean()
            pearsonr = torchmetrics.functional.pearson_corrcoef(y_hat_raw, y_raw).item()
            spearmanr = torchmetrics.functional.spearman_corrcoef(y_hat_raw, y_raw).item()
            r2 = torchmetrics.functional.r2_score(y_hat_raw, y_raw).item()

            self.logger.experiment.add_text("MSE", f"MSE: {mse:.4f}, MAE: {mae:.4f}, PEAR: {pearsonr:.4f}, SPEAR: {spearmanr:.4f}, R2: {r2:.4f}", self.current_epoch)

        elif self.hparams.loss.startswith("mae"):
            mse = torchmetrics.functional.mean_squared_error(y_hat_raw, y_raw).item()
            # This is very close to run_inference.py -> mae, but not completely identical -> why?
            mae = torchmetrics.functional.mean_absolute_error(y_hat_raw, y_raw).item()  # identical to: torch.abs(y_hat_raw-y_raw).mean()
            pearsonr = torchmetrics.functional.pearson_corrcoef(y_hat_raw, y_raw).item()
            spearmanr = torchmetrics.functional.spearman_corrcoef(y_hat_raw, y_raw).item()
            r2 = torchmetrics.functional.r2_score(y_hat_raw, y_raw).item()

            self.logger.experiment.add_text("MAE", f"MSE: {mse:.4f}, MAE: {mae:.4f}, PEAR: {pearsonr:.4f}, SPEAR: {spearmanr:.4f}, R2: {r2:.4f}", self.current_epoch)

        elif self.hparams.loss == "kld":
            mse = torchmetrics.functional.mean_squared_error(y_hat_raw, y_raw).item()
            # This is very close to run_inference.py -> mae, but not completely identical -> why?
            mae = torchmetrics.functional.mean_absolute_error(y_hat_raw, y_raw).item()  # identical to: torch.abs(y_hat_raw-y_raw).mean()
            pearsonr = torchmetrics.functional.pearson_corrcoef(y_hat_raw, y_raw).item()
            spearmanr = torchmetrics.functional.spearman_corrcoef(y_hat_raw, y_raw).item()
            r2 = torchmetrics.functional.r2_score(y_hat_raw, y_raw).item()

            self.logger.experiment.add_text("KLD", f"MSE: {mse:.4f}, MAE: {mae:.4f}, PEAR: {pearsonr:.4f}, SPEAR: {spearmanr:.4f}, R2: {r2:.4f}", self.current_epoch)

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
            self.log('hp_metric', f1_all)

        ### SAVE RESULTS !!!
        output_path = Path("$PATH/WM_ICV/result_0521")

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

        # ### SAVE RESULTS !!!


        #     "mse": mse,
        #     "mae": mae,
        #     "pearsonr": pearsonr,
        #     "spearmanr": spearmanr,
        #     "r2": r2


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
