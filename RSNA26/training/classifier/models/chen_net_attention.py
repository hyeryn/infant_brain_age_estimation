


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    def __init__(self, crop_size, num_classes=2, in_chans=1, dropout=0.5, dim=2, nr_filt=16):
        super(ChenNet_att, self).__init__()
        Conv = nn.Conv3d
        BatchNorm = nn.BatchNorm3d
        MaxPool = nn.MaxPool3d
        AdaptiveAvgPool = nn.AdaptiveAvgPool3d
        avg_pool_size = [2, 2, 2]

        self.Conv_1 = Conv(in_chans, nr_filt, 3, 1, "same")
        self.Conv_1_bn = BatchNorm(nr_filt)
        self.Conv_1_mp = MaxPool(2)
        self.se_1 = AttentionBlock(nr_filt)  # Attention Block

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
        self.dense_2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
        x = self.se_1(x)  # Apply SE Attention
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
