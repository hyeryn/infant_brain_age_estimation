import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention3D, self).__init__()
        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool3d(x, 1)
        max_pool = F.adaptive_max_pool3d(x, 1)
        avg_out = self.fc2(self.relu(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu(self.fc1(max_pool)))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, 1, keepdim=True)
        max_pool, _ = torch.max(x, 1, keepdim=True)
        x_cat = torch.cat([avg_pool, max_pool], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)
    
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)

class CBAM3D(nn.Module):
    def __init__(self, in_channels):
        super(CBAM3D, self).__init__()

        # Channel Attention
        if in_channels <= 32:
            reduction = 4
        elif in_channels <= 128:
            reduction = 8
        else:
            reduction = 16

        self.channel_attention = ChannelAttention3D(in_channels, reduction)
        self.spatial_attention = SpatialAttention3D()

    def forward(self, x):
        x_raw = x.clone()

        # Channel Attention
        ch_att = self.channel_attention(x)
        x_ch = x * ch_att

        # Spatial Attention
        sp_att = self.spatial_attention(x_ch)
        x_sp = x_ch * sp_att

        return x_raw, x_ch, x_sp, ch_att, sp_att

class UNetPlusPlus(nn.Module):
    def __init__(self, in_chans=1, base_ch=16, dropout=0.3):
        super().__init__()
        self.conv00 = ConvBlock(in_chans, base_ch)
        self.pool0 = nn.MaxPool3d((2, 2, 1))
        self.attn0 = CBAM3D(base_ch)

        self.conv10 = ConvBlock(base_ch, base_ch * 2)
        self.pool1 = nn.MaxPool3d((2, 2, 1))
        self.attn1 = CBAM3D(base_ch * 2)

        self.conv20 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pool2 = nn.MaxPool3d((2, 2, 2))
        self.attn2 = CBAM3D(base_ch * 4)

        self.conv30 = ConvBlock(base_ch * 4, base_ch * 8)
        self.pool3 = nn.MaxPool3d((2, 2, 2))
        self.attn3 = CBAM3D(base_ch * 8)

        self.conv40 = ConvBlock(base_ch * 8, base_ch * 4)
        self.attn4 = CBAM3D(base_ch * 4)


        self.gap0 = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.gap1 = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.gap2 = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.gap3 = nn.AdaptiveAvgPool3d((6, 6, 6))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(base_ch * (1 + 2 + 4 + 8) * 6 * 6 * 6, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x0 = self.conv00(x)
        x1 = self.conv10(self.pool0(x0))
        x2 = self.conv20(self.pool1(x1))
        x3 = self.conv30(self.pool2(x2))
        x4 = self.conv40(self.pool3(x3))

        _, _, x4, _, _ = self.attn4(x4)

        v0 = self.gap0(x0).view(x0.size(0), -1)
        v1 = self.gap1(x1).view(x1.size(0), -1)
        v2 = self.gap2(x2).view(x2.size(0), -1)
        v3 = self.gap3(x3).view(x3.size(0), -1)

        features = torch.cat([v0, v1, v2, v3], dim=1)
        out = self.fc(self.dropout(features))

        return out, x4


#         # 16, 32, 64, 128. 


#         # Deep Supervision: GAP from each stage
