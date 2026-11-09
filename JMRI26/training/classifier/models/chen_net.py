import numpy as np
import torch
from torch import nn

class ChenNet(nn.Module):
    """
    Network from here: https://pubs.rsna.org/doi/10.1148/radiol.211860

    init: He
    epochs: 50
    bs: 16
    optimizer: adam
    dropout: 0.5
    lr: 0.001 at start then decaying

        stopping = EarlyStopping(patience=30)
        reduce_lr = ReduceLROnPlateau(
            factor=0.1,
            patience=8,
            min_lr=configs['learning_rate'] * 0.001)

    input image size: 86, 110, 78   (roughly downsampled by 2x from original size)

    (my images after resampling to 1.5mm isotropic: 86, 94, 84)

    Data was put into 10 bins by age. Then during training each bin was sampled equally often (inverse
    frequency).

    Data augmentation: 
    - radom rotation by up to 10 degree
    - translation up to 3 voxels
    - flipping along sagittal plane
    """
    def __init__(self, in_chans=1, dropout=0.5, dim=2, nr_filt=16):

        Conv = nn.Conv3d
        BatchNorm = nn.BatchNorm3d
        MaxPool = nn.MaxPool3d
        AdaptiveAvgPool = nn.AdaptiveAvgPool3d
        avg_pool_size = [2, 2, 2]

        nn.Module.__init__(self)
        self.Conv_1 = Conv(in_chans, nr_filt, 3, 1, "same")
        self.Conv_1_bn = BatchNorm(nr_filt)
        self.Conv_1_mp = MaxPool(2)
        self.Conv_2 = Conv(nr_filt, nr_filt*2, 3, 1, "same")
        self.Conv_2_bn = BatchNorm(nr_filt*2)
        self.Conv_2_mp = MaxPool(2)
        self.Conv_3 = Conv(nr_filt*2, nr_filt*4, 3, 1, "same")
        self.Conv_3_bn = BatchNorm(nr_filt*4)
        self.Conv_3_mp = MaxPool(2)
        self.Conv_4 = Conv(nr_filt*4, nr_filt*8, 3, 1, "same")
        self.Conv_4_bn = BatchNorm(nr_filt*8)
        self.Conv_4_mp = MaxPool(2)
        self.Conv_5 = Conv(nr_filt*8, nr_filt*16, 3, 1, "same")
        self.Conv_5_bn = BatchNorm(nr_filt*16)
        self.Conv_5_mp = MaxPool(2)

        self.Conv_6 = Conv(nr_filt*16, nr_filt*4, 1, 1, "same")
        self.Conv_6_bn = BatchNorm(nr_filt*4)
        self.Conv_6_ap = AdaptiveAvgPool(avg_pool_size)  # 6

        self.dropout = nn.Dropout(dropout)
        self.dense_1 = nn.Linear(nr_filt*4*np.prod(avg_pool_size), 512)
        self.dense_2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
        x = self.Conv_1_mp(x)
        x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
        x = self.Conv_2_mp(x)
        x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
        x = self.Conv_3_mp(x)
        x = self.relu(self.Conv_4_bn(self.Conv_4(x)))
        x = self.Conv_4_mp(x)
        x = self.relu(self.Conv_5_bn(self.Conv_5(x)))
        x = self.Conv_5_mp(x)
        x = self.relu(self.Conv_6_bn(self.Conv_6(x)))
        x = self.Conv_6_ap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.relu(self.dense_1(x))
        x = self.dense_2(x)
        # Note that no sigmoid is applied here, because the network is used in combination with BCEWithLogitsLoss,
        # which applies sigmoid and BCELoss at the same time to make it numerically stable.
        
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

class ChenNet_att(nn.Module):
    def __init__(self, in_chans=1, dropout=0.5, dim=2, nr_filt=16):
        super(ChenNet_att, self).__init__()
        Conv = nn.Conv3d
        BatchNorm = nn.BatchNorm3d
        MaxPool = nn.MaxPool3d
        AdaptiveAvgPool = nn.AdaptiveAvgPool3d
        avg_pool_size = [2, 2, 2]

        self.Conv_1 = Conv(in_chans, nr_filt, 3, 1, "same")
        self.Conv_1_bn = BatchNorm(nr_filt)
        self.Conv_1_mp = MaxPool(2)

        self.Conv_2 = Conv(nr_filt, nr_filt*2, 3, 1, "same")
        self.Conv_2_bn = BatchNorm(nr_filt*2)
        self.Conv_2_mp = MaxPool(2)
        self.se_2 = AttentionBlock(nr_filt*2)

        self.Conv_3 = Conv(nr_filt*2, nr_filt*4, 3, 1, "same")
        self.Conv_3_bn = BatchNorm(nr_filt*4)
        self.Conv_3_mp = MaxPool(2)
        self.se_3 = AttentionBlock(nr_filt*4)

        self.Conv_4 = Conv(nr_filt*4, nr_filt*8, 3, 1, "same")
        self.Conv_4_bn = BatchNorm(nr_filt*8)
        self.Conv_4_mp = MaxPool(2)
        self.se_4 = AttentionBlock(nr_filt*8)

        self.Conv_5 = Conv(nr_filt*8, nr_filt*16, 3, 1, "same")
        self.Conv_5_bn = BatchNorm(nr_filt*16)
        self.Conv_5_mp = MaxPool(2)
        self.se_5 = AttentionBlock(nr_filt*16)

        self.Conv_6 = Conv(nr_filt*16, nr_filt*4, 1, 1, "same")
        self.Conv_6_bn = BatchNorm(nr_filt*4)
        self.Conv_6_ap = AdaptiveAvgPool(avg_pool_size)

        self.dropout = nn.Dropout(dropout)
        self.dense_1 = nn.Linear(nr_filt*4*np.prod(avg_pool_size), 512)
        self.dense_2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
        x = self.Conv_1_mp(x)
        
        x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
        x = self.se_2(x)
        x = self.Conv_2_mp(x)
        
        x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
        x = self.se_3(x)
        x = self.Conv_3_mp(x)
        
        x = self.relu(self.Conv_4_bn(self.Conv_4(x)))
        x = self.se_4(x)
        x = self.Conv_4_mp(x)
        
        x = self.relu(self.Conv_5_bn(self.Conv_5(x)))
        x = self.se_5(x)
        x = self.Conv_5_mp(x)
        
        x = self.relu(self.Conv_6_bn(self.Conv_6(x)))
        x = self.Conv_6_ap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.relu(self.dense_1(x))
        x = self.dense_2(x)
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


class ChenNet_film(nn.Module):
    def __init__(self, in_chans=1, dropout=0.5, dim=2, nr_filt=16):
        super(ChenNet_film, self).__init__()
        Conv = nn.Conv3d
        BatchNorm = nn.BatchNorm3d
        MaxPool = nn.MaxPool3d
        AdaptiveAvgPool = nn.AdaptiveAvgPool3d
        avg_pool_size = [2, 2, 2]

        self.Conv_1 = Conv(in_chans, nr_filt, 3, 1, "same")
        self.Conv_1_bn = BatchNorm(nr_filt)
        self.Conv_1_mp = MaxPool(2)

        self.Conv_2 = Conv(nr_filt, nr_filt*2, 3, 1, "same")
        self.Conv_2_bn = BatchNorm(nr_filt*2)
        self.Conv_2_mp = MaxPool(2)
        self.se_2 = FiLMBlock(nr_filt*2, 2)

        self.Conv_3 = Conv(nr_filt*2, nr_filt*4, 3, 1, "same")
        self.Conv_3_bn = BatchNorm(nr_filt*4)
        self.Conv_3_mp = MaxPool(2)
        self.se_3 = FiLMBlock(nr_filt*4, 2)

        self.Conv_4 = Conv(nr_filt*4, nr_filt*8, 3, 1, "same")
        self.Conv_4_bn = BatchNorm(nr_filt*8)
        self.Conv_4_mp = MaxPool(2)
        self.se_4 = FiLMBlock(nr_filt*8, 2)

        self.Conv_5 = Conv(nr_filt*8, nr_filt*16, 3, 1, "same")
        self.Conv_5_bn = BatchNorm(nr_filt*16)
        self.Conv_5_mp = MaxPool(2)
        self.se_5 = FiLMBlock(nr_filt*16, 2)

        self.Conv_6 = Conv(nr_filt*16, nr_filt*4, 1, 1, "same")
        self.Conv_6_bn = BatchNorm(nr_filt*4)
        self.Conv_6_ap = AdaptiveAvgPool(avg_pool_size)

        self.dropout = nn.Dropout(dropout)
        self.dense_1 = nn.Linear(nr_filt*4*np.prod(avg_pool_size), 512)
        self.dense_2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()

    def forward(self, x, info):

        wm, icv = info
        wm = wm.reshape(-1, 1).float()
        icv = icv.reshape(-1, 1).float()
        cond = torch.concat((wm, icv), dim=1)  # [B, 2]

        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
        x = self.Conv_1_mp(x)
        
        x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
        x = self.se_2(x, cond)
        x = self.Conv_2_mp(x)
        
        x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
        x = self.se_3(x, cond)
        x = self.Conv_3_mp(x)
        
        x = self.relu(self.Conv_4_bn(self.Conv_4(x)))
        x = self.se_4(x, cond)
        x = self.Conv_4_mp(x)
        
        x = self.relu(self.Conv_5_bn(self.Conv_5(x)))
        x = self.se_5(x, cond)
        x = self.Conv_5_mp(x)
        
        x = self.relu(self.Conv_6_bn(self.Conv_6(x)))
        x = self.Conv_6_ap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.relu(self.dense_1(x))
        x = self.dense_2(x)
        return x
