import numpy as np
import torch
from torch import nn


class CNN_light(nn.Module):
    def __init__(self, in_chans=1, dropout=0.4):
        super().__init__()
        AdaptiveAvgPool = nn.AdaptiveAvgPool3d
        avg_pool_size = [8, 8, 8]

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_chans, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 1))
        )
        self.Conv_4_ap = AdaptiveAvgPool(avg_pool_size)  # 6
        self.dense_1 = nn.Linear(32*np.prod(avg_pool_size), 128)
        self.dense_2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  
        x = self.Conv_4_ap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.relu(self.dense_1(x))
        x = self.dropout2(x)
        x = self.dense_2(x)

        return x
