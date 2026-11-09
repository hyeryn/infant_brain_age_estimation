import torch
import torch.nn as nn
import torch.nn.functional as F


def center_crop_to_match(source, target):
    _, _, d_s, h_s, w_s = source.shape
    _, _, d_t, h_t, w_t = target.shape
    
    d_start = (d_s - d_t) // 2
    h_start = (h_s - h_t) // 2
    w_start = (w_s - w_t) // 2

    cropped = source[:, :, d_start:d_start+d_t, h_start:h_start+h_t, w_start:w_start+w_t]
    return cropped


class Unet_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm3d(out_channels)        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class UNetPlusPlus_upsampling(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False, dropout=0.3):
        super().__init__()

        num_filter = [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision
        self.pool1 = nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1))
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.up1 = nn.Upsample(scale_factor=(2,2,1), mode='trilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        ### forecasting
        self.pool = nn.AdaptiveAvgPool3d((20, 20, 20))  # Global Avg Pooling
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Flatten(),                     # (B, C, 1, 1, 1) → (B, C)
            nn.Linear(20*20*20, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1000, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(300, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)          # Output: scalar (e.g., age)
        )

        # DownSampling
        self.conv0_0 = Unet_block(input_channels, num_filter[0], num_filter[0])
        self.conv1_0 = Unet_block(num_filter[0], num_filter[1], num_filter[1])
        self.conv2_0 = Unet_block(num_filter[1], num_filter[2], num_filter[2])
        self.conv3_0 = Unet_block(num_filter[2], num_filter[3], num_filter[3])
        self.conv4_0 = Unet_block(num_filter[3], num_filter[4], num_filter[2])

        # Upsampling & Dense skip
        # N to 1 skip
        self.conv0_1 = Unet_block(num_filter[0] + num_filter[1], num_filter[0], num_filter[0])
        self.conv1_1 = Unet_block(num_filter[1] + num_filter[2], num_filter[1], num_filter[1])
        self.conv2_1 = Unet_block(num_filter[2] + num_filter[3], num_filter[2], num_filter[2])
        self.conv3_1 = Unet_block(num_filter[3] + num_filter[2], num_filter[3], num_filter[3])
       
        # N to 2 skip
        self.conv0_2 = Unet_block(num_filter[0]*2 + num_filter[1], num_filter[0], num_filter[0])
        self.conv1_2 = Unet_block(num_filter[1]*2 + num_filter[2], num_filter[1], num_filter[1])
        self.conv2_2 = Unet_block(num_filter[2]*2 + num_filter[3], num_filter[2], num_filter[2])

        # N to 3 skip
        self.conv0_3 = Unet_block(num_filter[0]*3 + num_filter[1], num_filter[0], num_filter[0])
        self.conv1_3 = Unet_block(num_filter[1]*3 + num_filter[2], num_filter[1], num_filter[1])

        # N to 4 skip
        self.conv0_4 = Unet_block(num_filter[0]*4 + num_filter[1], num_filter[0], num_filter[0])

        if self.deep_supervision:
            self.output1 = nn.Conv3d(num_filter[0], num_classes, kernel_size=1)
            self.output2 = nn.Conv3d(num_filter[0], num_classes, kernel_size=1)
            self.output3 = nn.Conv3d(num_filter[0], num_classes, kernel_size=1)
            self.output4 = nn.Conv3d(num_filter[0], num_classes, kernel_size=1)

        else:
            self.output = nn.Conv3d(num_filter[0], num_classes, kernel_size=1)

    def forward(self, x):  
        # Encoder path
        x0_0 = self.conv0_0(x)
        print(x0_0.shape)
        print(self.pool1(x0_0).shape)
        x1_0 = self.conv1_0(self.pool1(x0_0))
        x2_0 = self.conv2_0(self.pool1(x1_0))
        x3_0 = self.conv3_0(self.pool2(x2_0))
        x4_0 = self.conv4_0(self.pool2(x3_0))


        # Decoder path with proper crop before concat
        # x0_1
        x1_0_up = self.up1(x1_0)
        x0_0_crop = center_crop_to_match(x0_0, x1_0_up)
        print(x0_0_crop.shape, x1_0_up.shape)
        cat_0 = torch.cat([x0_0_crop, x1_0_up], dim=1)
        print(cat_0.shape)
        x0_1 = self.conv0_1(cat_0)
        print(x0_1.shape)
        # x1_1
        x2_0_up = self.up1(x2_0)
        x1_0_crop = center_crop_to_match(x1_0, x2_0_up)
        x1_1 = self.conv1_1(torch.cat([x1_0_crop, x2_0_up], dim=1))

        # x0_2
        x1_1_up = self.up1(x1_1)
        x0_0_crop2 = center_crop_to_match(x0_0, x1_1_up)
        x0_1_crop = center_crop_to_match(x0_1, x1_1_up)
        x0_2 = self.conv0_2(torch.cat([x0_0_crop2, x0_1_crop, x1_1_up], dim=1))

        # x2_1
        x3_0_up = self.up2(x3_0)
        x2_0_crop = center_crop_to_match(x2_0, x3_0_up)
        x2_1 = self.conv2_1(torch.cat([x2_0_crop, x3_0_up], dim=1))

        # x1_2
        x2_1_up = self.up1(x2_1)
        x1_0_crop2 = center_crop_to_match(x1_0, x2_1_up)
        x1_1_crop = center_crop_to_match(x1_1, x2_1_up)
        x1_2 = self.conv1_2(torch.cat([x1_0_crop2, x1_1_crop, x2_1_up], dim=1))

        # x0_3
        x1_2_up = self.up1(x1_2)
        x0_0_crop3 = center_crop_to_match(x0_0, x1_2_up)
        x0_1_crop2 = center_crop_to_match(x0_1, x1_2_up)
        x0_2_crop = center_crop_to_match(x0_2, x1_2_up)
        x0_3 = self.conv0_3(torch.cat([x0_0_crop3, x0_1_crop2, x0_2_crop, x1_2_up], dim=1))

        # x3_1
        x4_0_up = self.up2(x4_0)
        x3_0_crop = center_crop_to_match(x3_0, x4_0_up)
        x3_1 = self.conv3_1(torch.cat([x3_0_crop, x4_0_up], dim=1))

        # x2_2
        x3_1_up = self.up2(x3_1)
        x2_0_crop2 = center_crop_to_match(x2_0, x3_1_up)
        x2_1_crop = center_crop_to_match(x2_1, x3_1_up)
        x2_2 = self.conv2_2(torch.cat([x2_0_crop2, x2_1_crop, x3_1_up], dim=1))

        # x1_3
        x2_2_up = self.up1(x2_2)
        x1_0_crop3 = center_crop_to_match(x1_0, x2_2_up)
        x1_1_crop2 = center_crop_to_match(x1_1, x2_2_up)
        x1_2_crop = center_crop_to_match(x1_2, x2_2_up)
        x1_3 = self.conv1_3(torch.cat([x1_0_crop3, x1_1_crop2, x1_2_crop, x2_2_up], dim=1))

        # x0_4
        x1_3_up = self.up1(x1_3)
        x0_0_crop4 = center_crop_to_match(x0_0, x1_3_up)
        x0_1_crop3 = center_crop_to_match(x0_1, x1_3_up)
        x0_2_crop2 = center_crop_to_match(x0_2, x1_3_up)
        x0_3_crop = center_crop_to_match(x0_3, x1_3_up)
        x0_4 = self.conv0_4(torch.cat([x0_0_crop4, x0_1_crop3, x0_2_crop2, x0_3_crop, x1_3_up], dim=1))

        # Output
        if self.deep_supervision:
            output1 = self.output1(x0_1)
            output2 = self.output2(x0_2)
            output3 = self.output3(x0_3)
            output4 = self.output4(x0_4)

            output1_crop = center_crop_to_match(output1, output4)
            output2_crop = center_crop_to_match(output2, output4)
            output3_crop = center_crop_to_match(output3, output4)

            output = (output1_crop + output2_crop + output3_crop + output4) / 4
            forecasting = self.fc(self.pool(output))
        else:
            output = self.output(x0_4)
            forecasting = self.fc(self.pool(output))

        return forecasting     # embedding / concat / prediction (age)
