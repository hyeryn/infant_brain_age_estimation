import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNetEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super(UNetEncoder, self).__init__()

        self.enc1 = nn.Sequential(
            conv_block(in_channels, base_channels),
            conv_block(base_channels, base_channels)
        )
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = nn.Sequential(
            conv_block(base_channels, base_channels * 2),
            conv_block(base_channels * 2, base_channels * 2)
        )
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = nn.Sequential(
            conv_block(base_channels * 2, base_channels * 4),
            conv_block(base_channels * 4, base_channels * 4)
        )
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = nn.Sequential(
            conv_block(base_channels * 4, base_channels * 8),
            conv_block(base_channels * 8, base_channels * 8)
        )
        self.pool4 = nn.MaxPool3d(2)

        self.enc5 = nn.Sequential(
            conv_block(base_channels * 8, base_channels * 16),
            conv_block(base_channels * 16, base_channels * 16)
        )

    def forward(self, x):
        x1 = self.enc1(x)  # [B, base_channels, D, H, W]
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        x5 = self.enc5(self.pool4(x4))

        print(x1.shape, self.pool1(x1).shape, x2.shape)
        return x5


class UNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, dropout=True):
        super(UNet, self).__init__()
        self.encoder = UNetEncoder(in_channels, base_channels)
        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(base_channels * 16, base_channels * 4),
            nn.ReLU(),
            nn.Linear(base_channels * 4, 1)  
        )

    def forward(self, x):
        features = self.encoder(x)  # [B, base_channels*16, D, H, W]
        # Global average pooling
        pooled = F.adaptive_avg_pool3d(features, (1, 1, 1)).view(x.size(0), -1)  # [B, base_channels*16]
        pooled = self.dropout(pooled)
        out = self.fc(pooled)  # [B, 1]
        return out


##### ------- #####

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            diffZ = skip.size(2) - x.size(2)
            diffY = skip.size(3) - x.size(3)
            diffX = skip.size(4) - x.size(4)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2,
                          diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x

class AgeRegressor(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.fc = nn.Linear(in_ch, 1)
    def forward(self, x):
        return self.fc(x).squeeze(1)

class UNet_AgeSegRecon(nn.Module):
    def __init__(self, in_channels=1, base_ch=32, num_classes=1):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch*2)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4)
        self.enc4 = ConvBlock(base_ch*4, base_ch*8)
        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch*8, base_ch*16)

        # Decoder
        self.up4 = UpBlock(base_ch*16, base_ch*8)
        self.up3 = UpBlock(base_ch*8, base_ch*4)
        self.up2 = UpBlock(base_ch*4, base_ch*2)
        self.up1 = UpBlock(base_ch*2, base_ch)

        self.seg_head = nn.Conv3d(base_ch, num_classes, kernel_size=1)

        self.recon_head = nn.Conv3d(base_ch, in_channels, kernel_size=1)

        # Age regressors (deep supervision) for encoder features
        self.age_regressors = nn.ModuleList([
            AgeRegressor(base_ch*8),   # enc4 output channels
            AgeRegressor(base_ch*4),   # enc3 output channels
            AgeRegressor(base_ch*2),   # enc2 output channels
            AgeRegressor(base_ch),     # enc1 output channels
        ])

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)   # [B, base_ch, D, H, W]
        p1 = self.pool(x1)

        x2 = self.enc2(p1)  # [B, base_ch*2, D/2, H/2, W/2]
        p2 = self.pool(x2)

        x3 = self.enc3(p2)  # [B, base_ch*4, D/4, H/4, W/4]
        p3 = self.pool(x3)

        x4 = self.enc4(p3)  # [B, base_ch*8, D/8, H/8, W/8]
        p4 = self.pool(x4)

        bottleneck = self.bottleneck(p4)  # [B, base_ch*16, D/16, H/16, W/16]

        # Decoder
        d4 = self.up4(bottleneck, x4)  # [B, base_ch*8, D/8, H/8, W/8]
        d3 = self.up3(d4, x3)          # [B, base_ch*4, D/4, H/4, W/4]
        d2 = self.up2(d3, x2)          # [B, base_ch*2, D/2, H/2, W/2]
        d1 = self.up1(d2, x1)          # [B, base_ch, D, H, W]

        # Segmentation output
        seg_out = self.seg_head(d1)    # [B, num_classes, D, H, W]

        # Reconstruction output
        recon_out = self.recon_head(d1) # [B, in_channels, D, H, W]

        age_feats = [x4, x3, x2, x1]
        age_preds = []
        for feat, reg in zip(age_feats, self.age_regressors):
            pooled = F.adaptive_avg_pool3d(feat, (1,1,1)).view(feat.size(0), -1)
            age_pred = reg(pooled).squeeze()
            age_preds.append(age_pred)

        return seg_out, recon_out, age_preds


# #
