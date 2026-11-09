import numpy as np
import torch
from torch import nn

class CbrTiny(nn.Module):
    """
    Network from here: https://arxiv.org/pdf/1902.07208.pdf
    lr: 0.001
    bs: 32
    optimizer: Adam on Retina; SGD+Momentum with 0.9 on ChestXRay

    Main difference to basic_cnn:
    - only 1 dense layer (instead of 2)
    - kernel size 5 for conv (instead of 3)
    - kernel size 3 for maxpool (instead of 2)

    For my small datasets nr_filt > 8 not really necessary.

    With nr_filt=8: roughly 4MB of weights. basiccnn has roughly 26MB, mainly because of 
    2 instead of 1 dense layers.

    Bigger receptive window (5 here compared to 3 for basic_cnn (or 3 vs 2 in maxpool) will make
    image shrink even faster)

    For 3D 2-3x faster than 2d efficientnet_b0 with tiles
    """

    def __init__(self, in_chans=1, nr_filt=8, return_features=False):

        self.return_features = return_features

        Conv = nn.Conv3d
        BatchNorm = nn.BatchNorm3d
        MaxPool = nn.MaxPool3d
        AdaptiveAvgPool = nn.AdaptiveAvgPool3d
        # Rough heuristic: 
        # s should be slightly bigger than avg_pool_size
        avg_pool_size = [4, 4, 4]  # good for 3d isotropic with 4x maxpool

        nn.Module.__init__(self)
        
        # Good for 3d isotropic
        
        # Better for 3d with small z
        kernel_size = 3
        kernel_size_mp = 2

        self.Conv_1 = Conv(in_chans, nr_filt, kernel_size)
        self.Conv_1_bn = BatchNorm(nr_filt)
        self.Conv_1_mp = MaxPool(kernel_size_mp, 2)
        self.Conv_2 = Conv(nr_filt, nr_filt*2, kernel_size)
        self.Conv_2_bn = BatchNorm(nr_filt*2)
        self.Conv_2_mp = MaxPool(kernel_size_mp, 2)
        self.Conv_3 = Conv(nr_filt*2, nr_filt*4, kernel_size)
        self.Conv_3_bn = BatchNorm(nr_filt*4)
        self.Conv_3_mp = MaxPool(kernel_size_mp, 2)
        self.Conv_4 = Conv(nr_filt*4, nr_filt*8, kernel_size)
        self.Conv_4_bn = BatchNorm(nr_filt*8)

        # Jump to AvgPool should not be too big (in terms of shape) to avoid performance loss
        # (12->4: performance loss, 12->8: ok, 12->maxpool to 5->4: ok)
        #
        # TIMM efficientnet afaik uses AdaptiveAvgPool.
        self.Conv_4_ap = AdaptiveAvgPool(avg_pool_size)  # specify output shape (instead of stride+kernel size)
        if not self.return_features:
            self.dense_1 = nn.Linear(nr_filt*8*np.prod(avg_pool_size), 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
        x = self.Conv_1_mp(x)
        x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
        x = self.Conv_2_mp(x)
        x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
        x = self.Conv_3_mp(x)
        x = self.relu(self.Conv_4_bn(self.Conv_4(x)))
        x = self.Conv_4_ap(x)
        x = torch.flatten(x, 1)
        if not self.return_features:
            x = self.dense_1(x)
        # Do not use any final activation function, because often part of loss function to make it
        # numerically more stable.
        return x


class AttentionBlock(nn.Module):
    """
    Attention block that applies attention maps to the input tensor.
    This can be used to focus on specific regions of the feature map.
    """
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention_map = self.conv(x)
        attention_map = self.sigmoid(attention_map)
        return x * attention_map


class CbrTiny_att(nn.Module):

    def __init__(self, in_chans=1, nr_filt=8, return_features=False):

        self.return_features = return_features

        Conv = nn.Conv3d
        BatchNorm = nn.BatchNorm3d
        MaxPool = nn.MaxPool3d
        AdaptiveAvgPool = nn.AdaptiveAvgPool3d
        # Rough heuristic: 
        # s should be slightly bigger than avg_pool_size
        avg_pool_size = [4, 4, 4]  # good for 3d isotropic with 4x maxpool

        nn.Module.__init__(self)
        
        # Good for 3d isotropic
        
        # Better for 3d with small z
        kernel_size = 3
        kernel_size_mp = 2

        self.Conv_1 = Conv(in_chans, nr_filt, kernel_size)
        self.Conv_1_bn = BatchNorm(nr_filt)
        self.Conv_1_mp = MaxPool(kernel_size_mp, 2)
        self.Conv_2 = Conv(nr_filt, nr_filt*2, kernel_size)
        self.Conv_2_bn = BatchNorm(nr_filt*2)
        self.Conv_2_mp = MaxPool(kernel_size_mp, 2)
        self.att_block_2 = AttentionBlock(nr_filt*2)
        self.Conv_3 = Conv(nr_filt*2, nr_filt*4, kernel_size)
        self.Conv_3_bn = BatchNorm(nr_filt*4)
        self.Conv_3_mp = MaxPool(kernel_size_mp, 2)
        self.att_block_3 = AttentionBlock(nr_filt*4)
        self.Conv_4 = Conv(nr_filt*4, nr_filt*8, kernel_size)
        self.Conv_4_bn = BatchNorm(nr_filt*8)

        # Jump to AvgPool should not be too big (in terms of shape) to avoid performance loss
        # (12->4: performance loss, 12->8: ok, 12->maxpool to 5->4: ok)
        #
        # TIMM efficientnet afaik uses AdaptiveAvgPool.
        self.Conv_4_ap = AdaptiveAvgPool(avg_pool_size)  # specify output shape (instead of stride+kernel size)
        if not self.return_features:
            self.dense_1 = nn.Linear(nr_filt*8*np.prod(avg_pool_size), 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
        x = self.Conv_1_mp(x)
        x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
        x = self.Conv_2_mp(x)
        x = self.att_block_2(x)
        x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
        x = self.Conv_3_mp(x)
        x = self.att_block_3(x)
        x = self.relu(self.Conv_4_bn(self.Conv_4(x)))
        x = self.Conv_4_ap(x)
        x = torch.flatten(x, 1)
        if not self.return_features:
            x = self.dense_1(x)
        # Do not use any final activation function, because often part of loss function to make it
        # numerically more stable.
        return x


class FiLMBlock(nn.Module):
    def __init__(self, feature_dim, condition_dim):
        super(FiLMBlock, self).__init__()
        self.gamma_fc = nn.Linear(condition_dim, feature_dim)
        self.beta_fc = nn.Linear(condition_dim, feature_dim)

    def forward(self, x, cond):
        gamma = self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta 


class CbrTiny_film(nn.Module):

    def __init__(self, in_chans=1, nr_filt=8, return_features=False):

        self.return_features = return_features

        Conv = nn.Conv3d
        BatchNorm = nn.BatchNorm3d
        MaxPool = nn.MaxPool3d
        AdaptiveAvgPool = nn.AdaptiveAvgPool3d
        # Rough heuristic: 
        # s should be slightly bigger than avg_pool_size
        avg_pool_size = [4, 4, 4]  # good for 3d isotropic with 4x maxpool

        nn.Module.__init__(self)
        
        # Good for 3d isotropic
        
        # Better for 3d with small z
        kernel_size = 3
        kernel_size_mp = 2

        self.Conv_1 = Conv(in_chans, nr_filt, kernel_size)
        self.Conv_1_bn = BatchNorm(nr_filt)
        self.Conv_1_mp = MaxPool(kernel_size_mp, 2)
        self.Conv_2 = Conv(nr_filt, nr_filt*2, kernel_size)
        self.Conv_2_bn = BatchNorm(nr_filt*2)
        self.Conv_2_mp = MaxPool(kernel_size_mp, 2)
        self.att_block_2 = FiLMBlock(nr_filt*2, 2)
        self.Conv_3 = Conv(nr_filt*2, nr_filt*4, kernel_size)
        self.Conv_3_bn = BatchNorm(nr_filt*4)
        self.Conv_3_mp = MaxPool(kernel_size_mp, 2)
        self.att_block_3 = FiLMBlock(nr_filt*4, 2)
        self.Conv_4 = Conv(nr_filt*4, nr_filt*8, kernel_size)
        self.Conv_4_bn = BatchNorm(nr_filt*8)

        # Jump to AvgPool should not be too big (in terms of shape) to avoid performance loss
        # (12->4: performance loss, 12->8: ok, 12->maxpool to 5->4: ok)
        #
        # TIMM efficientnet afaik uses AdaptiveAvgPool.
        self.Conv_4_ap = AdaptiveAvgPool(avg_pool_size)  # specify output shape (instead of stride+kernel size)
        if not self.return_features:
            self.dense_1 = nn.Linear(nr_filt*8*np.prod(avg_pool_size), 1)
        self.relu = nn.ReLU()

    def forward(self, x, info):

        wm, icv = info
        wm = wm.reshape(-1, 1).float()
        icv = icv.reshape(-1, 1).float()
        cond = torch.concat((wm, icv), dim=1)  # [B, 2]


        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
        x = self.Conv_1_mp(x)
        x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
        x = self.Conv_2_mp(x)
        x = self.att_block_2(x, cond)
        x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
        x = self.Conv_3_mp(x)
        x = self.att_block_3(x, cond)
        x = self.relu(self.Conv_4_bn(self.Conv_4(x)))
        x = self.Conv_4_ap(x)
        x = torch.flatten(x, 1)
        if not self.return_features:
            x = self.dense_1(x)
        # Do not use any final activation function, because often part of loss function to make it
        # numerically more stable.
        return x
