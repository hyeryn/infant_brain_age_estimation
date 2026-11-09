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
from classifier.models.alexnet import alexnet
from classifier.models.torchvision_overwrite import TorchVisionOverwrite
from classifier.models.basic_cnn import BasicCNN
from classifier.models.cnn_light import CNN_light
from classifier.models.basic_cnn_attention import BasicCNN_att
from classifier.models.basic_cnn_custom import BasicCNN_att_cat_global, BasicCNN_att_cat_spatial_1, BasicCNN_att_cat_spatial_2, BasicCNN_att_cat_spatial_3
from classifier.models.basic_cnn_custom import BasicCNN_att_over_global, BasicCNN_att_over_spatial_1, BasicCNN_att_over_spatial_2, BasicCNN_att_over_spatial_3
from classifier.models.basic_cnn_custom import BasicCNN_cat_global, BasicCNN_att_cat_spatial_11, BasicCNN_att_cat_spatial_33, BasicCNN_att_weighted_cat_spatial
from classifier.models.basic_cnn_custom import BasicCNN_over_global, BasicCNN_att_over_spatial_11, BasicCNN_att_over_spatial_33, BasicCNN_att_weighted_over_spatial
from classifier.models.chen_net import ChenNet
from classifier.models.chen_net_attention import ChenNet_att
from classifier.models.chen_net_custom import ChenNet_att_cat_global, ChenNet_att_cat_spatial_1, ChenNet_att_cat_spatial_2, ChenNet_att_cat_spatial_3
from classifier.models.chen_net_custom import ChenNet_att_over_global, ChenNet_att_over_spatial_1, ChenNet_att_over_spatial_2, ChenNet_att_over_spatial_3
from classifier.models.chen_net_custom import ChenNet_cat_global, ChenNet_att_cat_spatial_11, ChenNet_att_cat_spatial_33, ChenNet_att_weighted_cat_spatial
from classifier.models.chen_net_custom import ChenNet_over_global, ChenNet_att_over_spatial_11, ChenNet_att_over_spatial_33, ChenNet_att_weighted_over_spatial

from classifier.models.chen_light_custom import ChenLight_att_cat_spatial_1, ChenLight_att_cat_spatial_2, ChenLight_att_cat_spatial_3
from classifier.models.chen_light_custom import ChenLight_att_over_spatial_1, ChenLight_att_over_spatial_2, ChenLight_att_over_spatial_3
from classifier.models.chen_light_custom import ChenLight_cat_global, ChenLight_att_cat_spatial_11, ChenLight_att_cat_spatial_33
from classifier.models.chen_light_custom import ChenLight_over_global, ChenLight_att_over_spatial_11, ChenLight_att_over_spatial_33
from classifier.models.chen_light_custom import ChenLight_att_weighted_cat_spatial, ChenLight_att_weighted_over_spatial, ChenLight_att_weighted_cat_spatial_, ChenLight_att_weighted_over_spatial_

from classifier.models.aggregate_custom import Agg_att_cat_spatial_1, Agg_att_cat_spatial_2, Agg_att_cat_spatial_3
from classifier.models.aggregate_custom import Agg_att_over_spatial_1, Agg_att_over_spatial_2, Agg_att_over_spatial_3
from classifier.models.aggregate_custom import Agg_cat_global, Agg_att_cat_spatial_11, Agg_att_cat_spatial_33
from classifier.models.aggregate_custom import Agg_over_global, Agg_att_over_spatial_11, Agg_att_over_spatial_33
from classifier.models.aggregate_custom import Agg_att_weighted_cat_spatial, Agg_att_weighted_over_spatial, Agg_att_weighted_cat_spatial_, Agg_att_weighted_over_spatial_

from classifier.models.aggregate_sample import Agg_sample
from classifier.models.unet_plus import UNetPlus
from classifier.models.unet_plusplus import UNetPlusPlus
from classifier.models.unet_plusplus_upsampling import UNetPlusPlus_upsampling
from classifier.models.resnet_custom import ResNet, ResNet_deep, ResNet_attention, ResNet_sigmoid, ResNet_crossattention, ResNet_crossattention_dp
from classifier.models.resnet_custom import ResNet_attention2, ResNet_attention_final, ResNet_attention_weight, ResNet_attention_weight_final, ResNet_attention_noweight_final
from classifier.models.resnet_custom import ResNet_indep_final, ResNet_indep_final2, ResNet_indep, ResNet_indep2
from classifier.models.resnet_attention import MultiModalAttentionResNet, MultiModalAttentionResNet2
from classifier.models.resnet_attention import MultiModalAttentionResNet_reverse, MultiModalAttentionResNet_reverse2

from classifier.models.cnn_rnn import CNNRNN
from classifier.models.cbr_tiny import CbrTiny
from classifier.models.cnn_transformer import CNNTransformer
from classifier.models.enet_hydra import enet_hydra
from classifier.models.cbr_tiny_hydra import cbr_tiny_hydra
from classifier.models.sfcn import SFCN
from classifier.models.agenet import AgeNet
from classifier.models.resnet3d import resnet18, resnet34, resnet50, resnet101, resnet152
from classifier.utils import confusion_matrix_to_figure
import classifier.dataset_utils

class LitClassifier(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # will save all kwargs to hparams.yml + store into self.hparams
        nr_channels = self.hparams.nr_channels
        # bce: one less than classes, because all 0 is first class
        nr_classes = self.hparams.nr_classes-1 if self.hparams.loss == "bce" else self.hparams.nr_classes
        spatial_dims = self.hparams.dim
        self.epochs = 0

        #                                                         "cbr_tiny", "cnn_rnn", "cnn_transformer",
        #                                                         "cnn_att_cat_g", "cnn_att_cat_s1", "cnn_att_cat_s2", "cnn_att_cat_s3", 
        #                                                         "cnn_att_over_g", "cnn_att_over_s1", "cnn_att_over_s2", "cnn_att_over_s3",
        #                                                         "chen_att_cat_g", "chen_att_cat_s1", "chen_att_cat_s2", "chen_att_cat_s3", 
        #                                                         "chen_att_over_g", "chen_att_over_s1", "chen_att_over_s2", "chen_att_over_s3",
        #                                                         "cbr_tiny_hydra", "chen_net", "chen_net_attention", "sfcn", "agenet",
        #                                                         "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        if self.hparams.model == "densenet":
            from monai.networks.nets.densenet import DenseNet121
            self.backbone = DenseNet121(spatial_dims=spatial_dims, in_channels=nr_channels,
                                        out_channels=nr_classes, dropout_prob=self.hparams.dropout,
                                        pretrained=self.hparams.pretrain)
        elif self.hparams.model == "efficientnet":
            from classifier.models.efficient_net import enet
            self.backbone = enet("efficientnet-b0", in_channels=nr_channels,
                                 out_channels=nr_classes, dropout=self.hparams.dropout)
        elif self.hparams.model == "efficientnet_3D":
            from classifier.models.efficient_net_3d import efficient_net_3d
            self.backbone = efficient_net_3d("efficientnet-b0", in_channels=nr_channels,
                                             out_channels=nr_classes, pretrained=self.hparams.pretrain)
        elif self.hparams.model == "resnext":
            from classifier.models.resnext import ResNext
            self.backbone = ResNext("resnext50_32x4d_ssl", 2)
        elif self.hparams.model == "vgg16":
            self.backbone = TorchVisionOverwrite("vgg16", pretrained=self.hparams.pretrain,
                                                 num_classes=nr_classes, in_chans=nr_channels)
        elif self.hparams.model == "alexnet":
            self.backbone = alexnet(
                num_classes=nr_classes, in_chans=nr_channels)
        elif self.hparams.model == "cnn_light":
            self.backbone = CNN_light(in_chans=nr_channels, dropout=self.hparams.dropout)
        elif self.hparams.model == "basiccnn":
            self.backbone = BasicCNN(self.hparams.crop_size, dim=spatial_dims, num_classes=nr_classes,
                                     in_chans=nr_channels, dropout=self.hparams.dropout)
        elif self.hparams.model == "basiccnn_att":
            self.backbone = BasicCNN_att(self.hparams.crop_size, dim=spatial_dims, num_classes=nr_classes,
                                     in_chans=nr_channels, dropout=self.hparams.dropout)
        elif self.hparams.model == "chen_net":
            self.backbone = ChenNet(self.hparams.crop_size, dim=spatial_dims, num_classes=nr_classes,
                                     in_chans=nr_channels, dropout=self.hparams.dropout, nr_filt=self.hparams.nr_filt)
        elif self.hparams.model == "chen_net_attention":
            self.backbone = ChenNet_att(self.hparams.crop_size, dim=spatial_dims, num_classes=nr_classes,
                                     in_chans=nr_channels, dropout=self.hparams.dropout, nr_filt=self.hparams.nr_filt)
        elif self.hparams.model == "cnn_rnn":
            self.backbone = CNNRNN(self.hparams.crop_size, pretrained=self.hparams.pretrain,
                                   num_classes=nr_classes, in_chans=nr_channels,
                                   dropout=self.hparams.dropout, hparams=self.hparams)
        elif self.hparams.model == "cnn_transformer":
            self.backbone = CNNTransformer(self.hparams.crop_size, pretrained=self.hparams.pretrain,
                                           num_classes=nr_classes, in_chans=nr_channels, 
                                           dropout=self.hparams.dropout, hparams=self.hparams)
        elif self.hparams.model == "cbr_tiny":
            self.backbone = CbrTiny(self.hparams.crop_size, dim=spatial_dims, num_classes=nr_classes,
                                    in_chans=nr_channels, nr_filt=self.hparams.nr_filt)
        elif self.hparams.model == "enet_hydra":
            self.backbone = enet_hydra("tf_efficientnet_b0_ns", in_channels=nr_channels,
                                       out_channels=nr_classes, dropout=self.hparams.dropout)
        elif self.hparams.model == "cbr_tiny_hydra":
            self.backbone = cbr_tiny_hydra(self.hparams.crop_size, dim=spatial_dims, num_classes=nr_classes,
                                           in_chans=nr_channels, nr_filt=self.hparams.nr_filt)

        elif self.hparams.model == 'sfcn':
            self.backbone = SFCN(in_chans=nr_channels, type=self.hparams.exp_name)
        elif self.hparams.model == 'agenet':
            self.backbone = AgeNet(in_chans=nr_channels)
        elif self.hparams.model == 'resnet18':
            self.backbone = resnet18(in_chans=nr_channels)
            print(self.backbone)
        elif self.hparams.model == 'resnet34':
            self.backbone = resnet34(in_chans=nr_channels)
        elif self.hparams.model == 'resnet50':
            self.backbone = resnet50(in_chans=nr_channels)
        elif self.hparams.model == 'resnet101':
            self.backbone = resnet101(in_chans=nr_channels)
        elif self.hparams.model == 'resnet152':
            self.backbone = resnet152(in_chans=nr_channels)

        elif self.hparams.model == 'cnn_att_cat_g':
            self.backbone = BasicCNN_att_cat_global(self.hparams)
        elif self.hparams.model == 'cnn_att_cat_s1':
            self.backbone = BasicCNN_att_cat_spatial_1(self.hparams)
        elif self.hparams.model == 'cnn_att_cat_s2':
            self.backbone = BasicCNN_att_cat_spatial_2(self.hparams)
        elif self.hparams.model == 'cnn_att_cat_s3':
            self.backbone = BasicCNN_att_cat_spatial_3(self.hparams)
        elif self.hparams.model == 'cnn_att_over_g':
            self.backbone = BasicCNN_att_over_global(self.hparams)
        elif self.hparams.model == 'cnn_att_over_s1':
            self.backbone = BasicCNN_att_over_spatial_1(self.hparams)
        elif self.hparams.model == 'cnn_att_over_s2':
            self.backbone = BasicCNN_att_over_spatial_2(self.hparams)
        elif self.hparams.model == 'cnn_att_over_s3':
            self.backbone = BasicCNN_att_over_spatial_3(self.hparams)

        elif self.hparams.model == 'cnn_cat_g':
            self.backbone = BasicCNN_cat_global(self.hparams)
        elif self.hparams.model == 'cnn_att_cat_s11':
            self.backbone = BasicCNN_att_cat_spatial_11(self.hparams)
        elif self.hparams.model == 'cnn_att_cat_s33':
            self.backbone = BasicCNN_att_cat_spatial_33(self.hparams)
        elif self.hparams.model == 'cnn_over_g':
            self.backbone = BasicCNN_over_global(self.hparams)
        elif self.hparams.model == 'cnn_att_over_s11':
            self.backbone = BasicCNN_att_over_spatial_11(self.hparams)
        elif self.hparams.model == 'cnn_att_over_s33':
            self.backbone = BasicCNN_att_over_spatial_33(self.hparams)
            
        elif self.hparams.model == 'chen_att_cat_g':
            self.backbone = ChenNet_att_cat_global(self.hparams)
        elif self.hparams.model == 'chen_att_cat_s1':
            self.backbone = ChenNet_att_cat_spatial_1(self.hparams)
        elif self.hparams.model == 'chen_att_cat_s2':
            self.backbone = ChenNet_att_cat_spatial_2(self.hparams)
        elif self.hparams.model == 'chen_att_cat_s3':
            self.backbone = ChenNet_att_cat_spatial_3(self.hparams)
        elif self.hparams.model == 'chen_att_over_g':
            self.backbone = ChenNet_att_over_global(self.hparams)
        elif self.hparams.model == 'chen_att_over_s1':
            self.backbone = ChenNet_att_over_spatial_1(self.hparams)
        elif self.hparams.model == 'chen_att_over_s2':
            self.backbone = ChenNet_att_over_spatial_2(self.hparams)
        elif self.hparams.model == 'chen_att_over_s3':
            self.backbone = ChenNet_att_over_spatial_3(self.hparams)

        elif self.hparams.model == 'chen_cat_g':
            self.backbone = ChenNet_cat_global(self.hparams)
        elif self.hparams.model == 'chen_att_cat_s11':
            self.backbone = ChenNet_att_cat_spatial_11(self.hparams)
        elif self.hparams.model == 'chen_att_cat_s33':
            self.backbone = ChenNet_att_cat_spatial_33(self.hparams)
        elif self.hparams.model == 'chen_over_g':
            self.backbone = ChenNet_over_global(self.hparams)
        elif self.hparams.model == 'chen_att_over_s11':
            self.backbone = ChenNet_att_over_spatial_11(self.hparams)
        elif self.hparams.model == 'chen_att_over_s33':
            self.backbone = ChenNet_att_over_spatial_33(self.hparams)

        elif self.hparams.model == 'cnn_att_weighted_cat':
            self.backbone = BasicCNN_att_weighted_cat_spatial(self.hparams)
        elif self.hparams.model == 'cnn_att_weighted_over':
            self.backbone = BasicCNN_att_weighted_over_spatial(self.hparams)
        elif self.hparams.model == 'chen_att_weighted_cat':
            self.backbone = ChenNet_att_weighted_cat_spatial(self.hparams)
        elif self.hparams.model == 'chen_att_weighted_over':
            self.backbone = ChenNet_att_weighted_over_spatial(self.hparams)

        elif self.hparams.model == 'chen_light_att_cat_s1':
            self.backbone = ChenLight_att_cat_spatial_1(self.hparams)
        elif self.hparams.model == 'chen_light_att_cat_s2':
            self.backbone = ChenLight_att_cat_spatial_2(self.hparams)
        elif self.hparams.model == 'chen_light_att_cat_s3':
            self.backbone = ChenLight_att_cat_spatial_3(self.hparams)
        elif self.hparams.model == 'chen_light_att_over_s1':
            self.backbone = ChenLight_att_over_spatial_1(self.hparams)
        elif self.hparams.model == 'chen_light_att_over_s2':
            self.backbone = ChenLight_att_over_spatial_2(self.hparams)
        elif self.hparams.model == 'chen_light_att_over_s3':
            self.backbone = ChenLight_att_over_spatial_3(self.hparams)
        elif self.hparams.model == 'chen_light_att_cat_s11':
            self.backbone = ChenLight_att_cat_spatial_11(self.hparams)
        elif self.hparams.model == 'chen_light_att_cat_s33':
            self.backbone = ChenLight_att_cat_spatial_33(self.hparams)
        elif self.hparams.model == 'chen_light_att_over_s11':
            self.backbone = ChenLight_att_over_spatial_11(self.hparams)
        elif self.hparams.model == 'chen_light_att_over_s33':
            self.backbone = ChenLight_att_over_spatial_33(self.hparams)

        elif self.hparams.model == 'chen_light_cat_g':
            self.backbone = ChenLight_cat_global(self.hparams)
        elif self.hparams.model == 'chen_light_over_g':
            self.backbone = ChenLight_over_global(self.hparams)

        elif self.hparams.model == 'chen_light_att_weighted_cat':
            self.backbone = ChenLight_att_weighted_cat_spatial(self.hparams)
        elif self.hparams.model == 'chen_light_att_weighted_over':
            self.backbone = ChenLight_att_weighted_over_spatial(self.hparams)        
        elif self.hparams.model == 'chen_light_att_weighted_cat_':
            self.backbone = ChenLight_att_weighted_cat_spatial_(self.hparams)
        elif self.hparams.model == 'chen_light_att_weighted_over_':
            self.backbone = ChenLight_att_weighted_over_spatial_(self.hparams)

        elif self.hparams.model == 'agg_att_cat_s1':
            self.backbone = Agg_att_cat_spatial_1(self.hparams)
        elif self.hparams.model == 'agg_att_cat_s2':
            self.backbone = Agg_att_cat_spatial_2(self.hparams)
        elif self.hparams.model == 'agg_att_cat_s3':
            self.backbone = Agg_att_cat_spatial_3(self.hparams)
        elif self.hparams.model == 'agg_att_over_s1':
            self.backbone = Agg_att_over_spatial_1(self.hparams)
        elif self.hparams.model == 'agg_att_over_s2':
            self.backbone = Agg_att_over_spatial_2(self.hparams)
        elif self.hparams.model == 'agg_att_over_s3':
            self.backbone = Agg_att_over_spatial_3(self.hparams)
        elif self.hparams.model == 'agg_att_cat_s11':
            self.backbone = Agg_att_cat_spatial_11(self.hparams)
        elif self.hparams.model == 'agg_att_cat_s33':
            self.backbone = Agg_att_cat_spatial_33(self.hparams)
        elif self.hparams.model == 'agg_att_over_s11':
            self.backbone = Agg_att_over_spatial_11(self.hparams)
        elif self.hparams.model == 'agg_att_over_s33':
            self.backbone = Agg_att_over_spatial_33(self.hparams)

        elif self.hparams.model == 'agg_cat_g':
            self.backbone = Agg_cat_global(self.hparams)
        elif self.hparams.model == 'agg_over_g':
            self.backbone = Agg_over_global(self.hparams)

        elif self.hparams.model == 'agg_att_weighted_cat':
            self.backbone = Agg_att_weighted_cat_spatial(self.hparams)
        elif self.hparams.model == 'agg_att_weighted_over':
            self.backbone = Agg_att_weighted_over_spatial(self.hparams)        
        elif self.hparams.model == 'agg_att_weighted_cat_':
            self.backbone = Agg_att_weighted_cat_spatial_(self.hparams)
        elif self.hparams.model == 'agg_att_weighted_over_':
            self.backbone = Agg_att_weighted_over_spatial_(self.hparams)
        elif self.hparams.model == 'agg_sample':
            self.backbone = Agg_sample()

        elif self.hparams.model == 'unet+':
            self.backbone = UNetPlus(in_chans=nr_channels, dropout=self.hparams.dropout)
        elif self.hparams.model == 'unet++up_deep':
            self.backbone = UNetPlusPlus_upsampling(num_classes=1, input_channels=nr_channels, dropout=self.hparams.dropout, deep_supervision=True)
        elif self.hparams.model == 'unet++up':
            self.backbone = UNetPlusPlus_upsampling(num_classes=1, input_channels=nr_channels, dropout=self.hparams.dropout, deep_supervision=False)


        elif self.hparams.model == 'resnet':
            self.backbone = ResNet(in_chans=nr_channels)
        elif self.hparams.model == 'resnet_deep':
            self.backbone = ResNet_deep(in_chans=nr_channels)
        elif self.hparams.model == 'resnet_att':
            self.backbone = ResNet_attention()
        elif self.hparams.model == 'resnet_sig':
            self.backbone = ResNet_sigmoid(in_chans=nr_channels)
        elif self.hparams.model == 'resnet_att_cross':
            self.backbone = ResNet_crossattention()
        elif self.hparams.model == 'resnet_att_cross_deep':
            self.backbone = ResNet_crossattention_dp()

        ## ResNet_attention2, ResNet_attention_final, ResNet_attention_weight, ResNet_attention_weight_final
        elif self.hparams.model == 'resnet_att2':
            self.backbone = ResNet_attention2()
        elif self.hparams.model == 'resnet_att_final':
            self.backbone = ResNet_attention_final()
        elif self.hparams.model == 'resnet_att_w':
            self.backbone = ResNet_attention_weight()
        elif self.hparams.model == 'resnet_att_w_final':
            self.backbone = ResNet_attention_weight_final()
        elif self.hparams.model == 'resnet_att_now_final':
            self.backbone = ResNet_attention_noweight_final()
        # ResNet_indep_final, ResNet_indep_final2, ResNet_indep, ResNet_indep2
        elif self.hparams.model == 'resnet_indep_final':
            self.backbone = ResNet_indep_final()
        elif self.hparams.model == 'resnet_indep_final2':
            self.backbone = ResNet_indep_final2()
        elif self.hparams.model == 'resnet_indep':
            self.backbone = ResNet_indep()
        elif self.hparams.model == 'resnet_indep2':
            self.backbone = ResNet_indep2()

        elif self.hparams.model == 'renet_mul_att_op1':
            self.backbone = MultiModalAttentionResNet(att_op=1)
        elif self.hparams.model == 'renet_mul_att_op2':
            self.backbone = MultiModalAttentionResNet(att_op=2)

        elif self.hparams.model == 'renet_mul_att_op11':
            self.backbone = MultiModalAttentionResNet2(att_op=1)
        elif self.hparams.model == 'renet_mul_att_op22':
            self.backbone = MultiModalAttentionResNet2(att_op=2)


        elif self.hparams.model == 'renet_mul_att_rev_op1':
            self.backbone = MultiModalAttentionResNet_reverse(att_op=1)
        elif self.hparams.model == 'renet_mul_att_rev_op2':
            self.backbone = MultiModalAttentionResNet_reverse(att_op=2)

        elif self.hparams.model == 'renet_mul_att_rev_op11':
            self.backbone = MultiModalAttentionResNet_reverse2(att_op=1)
        elif self.hparams.model == 'renet_mul_att_rev_op22':
            self.backbone = MultiModalAttentionResNet_reverse2(att_op=2) # reverse option


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
        elif self.hparams.loss == "mse_cat":
            label = getattr(classifier.dataset_utils, self.hparams.float_to_class_fn)(y)
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
        elif self.hparams.loss == "mse_cat":
            label = getattr(classifier.dataset_utils, self.hparams.float_to_class_fn)(y.detach())
        elif self.hparams.loss == "kld":
            label = y.detach() 
        return label

    def training_step(self, batch, batch_idx):
        x, y = batch
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
            y_hat = self.backbone(x)  # [bs, nr_classes]
            y_hat = y_hat[:, 0] if self.hparams.loss.startswith("mse") else y_hat

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
        x, y = batch
        # if self.hparams.model 
        if self.hparams.model == 'resnet_deep':
            y_hat_prob, aux_out2, aux_out3, aux_out4 = self.backbone(x)
            y_hat_prob = y_hat_prob[:, 0]
        elif self.hparams.model == 'resnet_att_cross_deep':
            y_hat_prob, _, _, _, _ = self.backbone(x)
            y_hat_prob = y_hat_prob[:, 0]
        
        else:
            y_hat_prob = self.backbone(x)
            y_hat_prob = y_hat_prob[:, 0] if self.hparams.loss.startswith("mse") else y_hat_prob
        
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

            # This is working but deactivated because shows many warnings. Activate if needed.

            # Same result as self.auroc_m
            
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
        
        x, y_raw = batch
        if self.hparams.model == 'resnet_deep':
            y_hat_raw, aux_out2, aux_out3, aux_out4 = self.backbone(x)
        elif self.hparams.model == 'resnet_att_cross_deep':
            y_hat_raw, _, _, _, _ = self.backbone(x)
        
        else:
            y_hat_raw = self.backbone(x)
        
        if self.hparams.loss != "kld":
            y_hat_raw = torch.round(y_hat_raw)
        y_hat_raw = y_hat_raw[:, 0] if self.hparams.loss.startswith("mse") else y_hat_raw
        y = self._decode_y(y_raw)

        ### CUSTOM to discrete number


        y_hat_raw = self._y_hat_to_prob(y_hat_raw)
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

        elif self.hparams.loss == "kld":
            mse = torchmetrics.functional.mean_squared_error(y_hat_raw, y_raw).item()
            # This is very close to run_inference.py -> mae, but not completely identical -> why?
            mae = torchmetrics.functional.mean_absolute_error(y_hat_raw, y_raw).item()  # identical to: torch.abs(y_hat_raw-y_raw).mean()
            pearsonr = torchmetrics.functional.pearson_corrcoef(y_hat_raw, y_raw).item()
            spearmanr = torchmetrics.functional.spearman_corrcoef(y_hat_raw, y_raw).item()
            r2 = torchmetrics.functional.r2_score(y_hat_raw, y_raw).item()

            self.logger.experiment.add_text("MSE", f"MSE: {mse:.4f}, MAE: {mae:.4f}, PEAR: {pearsonr:.4f}, SPEAR: {spearmanr:.4f}, R2: {r2:.4f}", self.current_epoch)

        # update hp_metric
        val = self.trainer.checkpoint_callback.best_model_score
        self.log('hp_metric', val)  # mse
        if self.hparams.loss.startswith("mse"):
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
        output_path = Path("$PATH/MyelinAge-master/result_0422_for_RSNA")

        model_dir = output_path / self.hparams.model 
        model_dir.mkdir(parents=True, exist_ok=True)

        fn = model_dir / f"{self.hparams.fold}_{self.hparams.exp_name}_results.csv"

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

        json_fn = model_dir / f"{self.hparams.fold}_{self.hparams.exp_name}_metrics.json"
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
