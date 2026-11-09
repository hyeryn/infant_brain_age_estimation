import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1,
                 channel_size=[64,64,128,256,512],
                 dropout=True, in_chans=1):
        super(ResNet, self).__init__()
        c = channel_size
        self.inplanes = c[0]

        self.conv1 = nn.Conv3d(in_chans, c[0], kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(c[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, c[1], layers[0])
        self.layer2 = self._make_layer(block, c[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, c[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, c[4], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        if dropout is True:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = nn.Identity()

        self.classifier = nn.Linear(c[4] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, origin):

        x = self.conv1(origin)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        out = self.avgpool(x4)

        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out_main = self.classifier(out)

        return out_main
    
    def extract_features(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feats = []
        for i in range(1, 5):  # layer1 ~ layer4
            x = getattr(self, f'layer{i}')(x)
            feats.append(x)

        return feats


class ResNet_deep(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1,
                 channel_size=[64,64,128,256,512],
                 dropout=True, in_chans=3):
        super(ResNet_deep, self).__init__()
        c = channel_size
        self.inplanes = c[0]

        self.conv1 = nn.Conv3d(in_chans, c[0], kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(c[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, c[1], layers[0])
        self.layer2 = self._make_layer(block, c[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, c[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, c[4], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        if dropout is True:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = nn.Identity()

        self.classifier = nn.Linear(c[4] * block.expansion, num_classes)

        # Auxiliary heads
        self.aux2 = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(c[2] * block.expansion, num_classes)
        )
        self.aux3 = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(c[3] * block.expansion, num_classes)
        )
        self.aux4 = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(c[4] * block.expansion, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, origin):

        x = self.conv1(origin)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # main output
        out = self.avgpool(x4)

        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out_main = self.classifier(out)

        # auxiliary outputs
        aux_out2 = self.aux2(x2)
        aux_out3 = self.aux3(x3)
        aux_out4 = self.aux4(x4)

        return out_main, aux_out2, aux_out3, aux_out4


class ResNet_indep_final(nn.Module):
    def __init__(self, base_encoder_fn=ResNet, num_layers=4):
        super().__init__()
        self.encoder_t1 = base_encoder_fn()
        self.encoder_t2 = base_encoder_fn()
        self.encoder_ratio = base_encoder_fn()
        self.num_layers = num_layers

        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x_t1, x_t2, x_ratio = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        feats_t1 = self.encoder_t1.extract_features(x_t1)  # list of 4-level features
        feats_t2 = self.encoder_t2.extract_features(x_t2)
        feats_ratio = self.encoder_ratio.extract_features(x_ratio)

        fused_feats = []
        i = 3
        ft1, ft2, fr = feats_t1[i], feats_t2[i], feats_ratio[i]

        # concat and apply attention
        combined = ft1 + ft2 + fr  # or torch.cat([...], dim=1)
        fused_feats.append(F.adaptive_avg_pool3d(combined, 1).view(combined.size(0), -1))

        final_feat = torch.cat(fused_feats, dim=1)  # (B, C*levels)
        out = self.regressor(final_feat)
        return out
    
class ResNet_indep_final2(nn.Module):
    def __init__(self, base_encoder_fn=ResNet, num_layers=4):
        super().__init__()
        self.encoder_t1 = base_encoder_fn()
        self.encoder_t2 = base_encoder_fn()
        self.encoder_ratio = base_encoder_fn()
        self.num_layers = num_layers

        self.regressor = nn.Sequential(
            nn.Linear(512*3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x_t1, x_t2, x_ratio = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        feats_t1 = self.encoder_t1.extract_features(x_t1)  # list of 4-level features
        feats_t2 = self.encoder_t2.extract_features(x_t2)
        feats_ratio = self.encoder_ratio.extract_features(x_ratio)

        fused_feats = []
        i = 3
        ft1, ft2, fr = feats_t1[i], feats_t2[i], feats_ratio[i]

        # concat and apply attention
        combined = torch.cat([ft1, ft2, fr], dim=1) 
        fused_feats.append(F.adaptive_avg_pool3d(combined, 1).view(combined.size(0), -1))

        final_feat = torch.cat(fused_feats, dim=1)  # (B, C*levels)
        out = self.regressor(final_feat)
        return out


class ResNet_indep(nn.Module):
    def __init__(self, base_encoder_fn=ResNet, num_layers=4):
        super().__init__()
        self.encoder_t1 = base_encoder_fn()
        self.encoder_t2 = base_encoder_fn()
        self.encoder_ratio = base_encoder_fn()
        self.num_layers = num_layers

        self.attn_blocks = nn.ModuleList([
            nn.Sequential(
                ChannelAttention3D(in_channels=feat_c),
                SpatialAttention3D()
            ) for feat_c in [64, 128, 256, 512]
        ])

        self.regressor = nn.Sequential(
            nn.Linear(960, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x_t1, x_t2, x_ratio = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        feats_t1 = self.encoder_t1.extract_features(x_t1)  # list of 4-level features
        feats_t2 = self.encoder_t2.extract_features(x_t2)
        feats_ratio = self.encoder_ratio.extract_features(x_ratio)

        fused_feats = []
        for i in range(self.num_layers):
            ft1, ft2, fr = feats_t1[i], feats_t2[i], feats_ratio[i]

            # concat and apply attention
            combined = ft1 + ft2 + fr  # or torch.cat([...], dim=1)
            fused_feats.append(F.adaptive_avg_pool3d(combined, 1).view(combined.size(0), -1))

        final_feat = torch.cat(fused_feats, dim=1)  # (B, C*levels)
        out = self.regressor(final_feat)
        return out

class ResNet_indep2(nn.Module):
    def __init__(self, base_encoder_fn=ResNet, num_layers=4):
        super().__init__()
        self.encoder_t1 = base_encoder_fn()
        self.encoder_t2 = base_encoder_fn()
        self.encoder_ratio = base_encoder_fn()
        self.num_layers = num_layers

        self.attn_blocks = nn.ModuleList([
            nn.Sequential(
                ChannelAttention3D(in_channels=feat_c),
                SpatialAttention3D()
            ) for feat_c in [64*3, 128*3, 256*3, 512*3]
        ])

        self.regressor = nn.Sequential(
            nn.Linear(960*3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x_t1, x_t2, x_ratio = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        feats_t1 = self.encoder_t1.extract_features(x_t1)  # list of 4-level features
        feats_t2 = self.encoder_t2.extract_features(x_t2)
        feats_ratio = self.encoder_ratio.extract_features(x_ratio)

        fused_feats = []
        for i in range(self.num_layers):
            ft1, ft2, fr = feats_t1[i], feats_t2[i], feats_ratio[i]

            # concat and apply attention
            combined = torch.cat([ft1, ft2, fr], dim=1)   # or torch.cat([...], dim=1)
            fused_feats.append(F.adaptive_avg_pool3d(combined, 1).view(combined.size(0), -1))

        final_feat = torch.cat(fused_feats, dim=1)  # (B, C*levels)
        out = self.regressor(final_feat)
        return out

class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)  # shape: (B, C, 1, 1, 1)

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # shape: (B, 1, D, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)  # shape: (B, 2, D, H, W)
        attn = self.conv(x_cat)
        return self.sigmoid(attn)  # shape: (B, 1, D, H, W)


class ResNet_attention(nn.Module):
    def __init__(self, base_encoder_fn=ResNet, num_layers=4):
        super().__init__()
        self.encoder_t1 = base_encoder_fn()
        self.encoder_t2 = base_encoder_fn()
        self.encoder_ratio = base_encoder_fn()
        self.num_layers = num_layers

        self.attn_blocks = nn.ModuleList([
            nn.Sequential(
                ChannelAttention3D(in_channels=feat_c),
                SpatialAttention3D()
            ) for feat_c in [64, 128, 256, 512]
        ])

        self.regressor = nn.Sequential(
            nn.Linear(960, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x_t1, x_t2, x_ratio = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        feats_t1 = self.encoder_t1.extract_features(x_t1)  # list of 4-level features
        feats_t2 = self.encoder_t2.extract_features(x_t2)
        feats_ratio = self.encoder_ratio.extract_features(x_ratio)

        fused_feats = []
        for i in range(self.num_layers):
            ft1, ft2, fr = feats_t1[i], feats_t2[i], feats_ratio[i]

            # concat and apply attention
            combined = ft1 + ft2 + fr  # or torch.cat([...], dim=1)
            ca = self.attn_blocks[i][0](combined)
            sa = self.attn_blocks[i][1](combined)
            attn_feat = combined * ca * sa
            fused_feats.append(F.adaptive_avg_pool3d(attn_feat, 1).view(attn_feat.size(0), -1))

        final_feat = torch.cat(fused_feats, dim=1)  # (B, C*levels)
        out = self.regressor(final_feat)
        return out

class ResNet_attention2(nn.Module):
    def __init__(self, base_encoder_fn=ResNet, num_layers=4):
        super().__init__()
        self.encoder_t1 = base_encoder_fn()
        self.encoder_t2 = base_encoder_fn()
        self.encoder_ratio = base_encoder_fn()
        self.num_layers = num_layers

        self.attn_blocks = nn.ModuleList([
            nn.Sequential(
                ChannelAttention3D(in_channels=feat_c),
                SpatialAttention3D()
            ) for feat_c in [64*3, 128*3, 256*3, 512*3]
        ])

        self.regressor = nn.Sequential(
            nn.Linear(960*3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x_t1, x_t2, x_ratio = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        feats_t1 = self.encoder_t1.extract_features(x_t1)  # list of 4-level features
        feats_t2 = self.encoder_t2.extract_features(x_t2)
        feats_ratio = self.encoder_ratio.extract_features(x_ratio)

        fused_feats = []
        for i in range(self.num_layers):
            ft1, ft2, fr = feats_t1[i], feats_t2[i], feats_ratio[i]

            # concat and apply attention
            combined = torch.cat([ft1, ft2, fr], dim=1)  # or torch.cat([...], dim=1)
            ca = self.attn_blocks[i][0](combined)
            sa = self.attn_blocks[i][1](combined)
            attn_feat = combined * ca * sa
            fused_feats.append(F.adaptive_avg_pool3d(attn_feat, 1).view(attn_feat.size(0), -1))

        final_feat = torch.cat(fused_feats, dim=1)  # (B, C*levels)
        out = self.regressor(final_feat)
        return out


class ResNet_attention_final(nn.Module):
    def __init__(self, base_encoder_fn=ResNet, num_layers=4):
        super().__init__()
        self.encoder_t1 = base_encoder_fn()
        self.encoder_t2 = base_encoder_fn()
        self.encoder_ratio = base_encoder_fn()
        self.num_layers = num_layers

        self.attn_blocks = nn.ModuleList([
            nn.Sequential(
                ChannelAttention3D(in_channels=feat_c),
                SpatialAttention3D()
            ) for feat_c in [64*3, 128*3, 256*3, 512*3]
        ])

        self.regressor = nn.Sequential(
            nn.Linear(512*3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x_t1, x_t2, x_ratio = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        feats_t1 = self.encoder_t1.extract_features(x_t1)  # list of 4-level features
        feats_t2 = self.encoder_t2.extract_features(x_t2)
        feats_ratio = self.encoder_ratio.extract_features(x_ratio)

        i = 3
        ft1, ft2, fr = feats_t1[i], feats_t2[i], feats_ratio[i]

        # concat and apply attention
        combined = torch.cat([ft1, ft2, fr], dim=1) 
        ca = self.attn_blocks[i][0](combined)
        sa = self.attn_blocks[i][1](combined)
        attn_feat = combined * ca * sa

        final_feat = F.adaptive_avg_pool3d(attn_feat, 1).view(attn_feat.size(0), -1) 
        out = self.regressor(final_feat)
        return out


class ResNet_attention_weight(nn.Module):
    def __init__(self, base_encoder_fn=ResNet, num_layers=4):
        super().__init__()
        self.encoder_t1 = base_encoder_fn()
        self.encoder_t2 = base_encoder_fn()
        self.encoder_ratio = base_encoder_fn()
        self.num_layers = num_layers

        self.attn_blocks = nn.ModuleList([
            nn.Sequential(
                ChannelAttention3D(in_channels=feat_c),
                SpatialAttention3D()
            ) for feat_c in [64, 128, 256, 512]
        ])

        self.regressor = nn.Sequential(
            nn.Linear(960, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x_t1, x_t2, x_ratio = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        feats_t1 = self.encoder_t1.extract_features(x_t1)  # list of 4-level features
        feats_t2 = self.encoder_t2.extract_features(x_t2)
        feats_ratio = self.encoder_ratio.extract_features(x_ratio)

        fused_feats = []
        for i in range(self.num_layers):
            ft1, ft2, fr = feats_t1[i], feats_t2[i], feats_ratio[i]

            # concat and apply attention
            combined = 0.5 * ft1 + 0.5 * ft2 + fr  # or torch.cat([...], dim=1)
            ca = self.attn_blocks[i][0](combined)
            sa = self.attn_blocks[i][1](combined)
            attn_feat = combined * ca * sa
            fused_feats.append(F.adaptive_avg_pool3d(attn_feat, 1).view(attn_feat.size(0), -1))

        final_feat = torch.cat(fused_feats, dim=1)  # (B, C*levels)
        out = self.regressor(final_feat)
        return out


class ResNet_attention_weight_final(nn.Module):
    def __init__(self, base_encoder_fn=ResNet, num_layers=4):
        super().__init__()
        self.encoder_t1 = base_encoder_fn()
        self.encoder_t2 = base_encoder_fn()
        self.encoder_ratio = base_encoder_fn()
        self.num_layers = num_layers

        self.attn_blocks = nn.ModuleList([
            nn.Sequential(
                ChannelAttention3D(in_channels=feat_c),
                SpatialAttention3D()
            ) for feat_c in [64, 128, 256, 512]
        ])

        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x_t1, x_t2, x_ratio = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        feats_t1 = self.encoder_t1.extract_features(x_t1)  # list of 4-level features
        feats_t2 = self.encoder_t2.extract_features(x_t2)
        feats_ratio = self.encoder_ratio.extract_features(x_ratio)

        i = 3
        ft1, ft2, fr = feats_t1[i], feats_t2[i], feats_ratio[i]

        # concat and apply attention
        combined = 0.5 * ft1 + 0.5 * ft2 + fr  # or torch.cat([...], dim=1)
        ca = self.attn_blocks[i][0](combined)
        sa = self.attn_blocks[i][1](combined)
        attn_feat = combined * ca * sa

        final_feat = F.adaptive_avg_pool3d(attn_feat, 1).view(attn_feat.size(0), -1) 
        out = self.regressor(final_feat)
        return out


class ResNet_attention_noweight_final(nn.Module):
    def __init__(self, base_encoder_fn=ResNet, num_layers=4):
        super().__init__()
        self.encoder_t1 = base_encoder_fn()
        self.encoder_t2 = base_encoder_fn()
        self.encoder_ratio = base_encoder_fn()
        self.num_layers = num_layers

        self.attn_blocks = nn.ModuleList([
            nn.Sequential(
                ChannelAttention3D(in_channels=feat_c),
                SpatialAttention3D()
            ) for feat_c in [64, 128, 256, 512]
        ])

        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x_t1, x_t2, x_ratio = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        feats_t1 = self.encoder_t1.extract_features(x_t1)  # list of 4-level features
        feats_t2 = self.encoder_t2.extract_features(x_t2)
        feats_ratio = self.encoder_ratio.extract_features(x_ratio)

        i = 3
        ft1, ft2, fr = feats_t1[i], feats_t2[i], feats_ratio[i]

        # concat and apply attention
        combined = ft1 + ft2 + fr  # or torch.cat([...], dim=1)
        ca = self.attn_blocks[i][0](combined)
        sa = self.attn_blocks[i][1](combined)
        attn_feat = combined * ca * sa

        final_feat = F.adaptive_avg_pool3d(attn_feat, 1).view(attn_feat.size(0), -1) 
        out = self.regressor(final_feat)
        return out


class SigmoidBlock(nn.Module):
    """
    Attention block that applies attention maps to the input tensor.
    This can be used to focus on specific regions of the feature map.
    """
    def __init__(self, in_channels):
        super(SigmoidBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention_map = self.conv(x)
        attention_map = self.sigmoid(attention_map)
        return x * attention_map


class ResNet_sigmoid(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1,
                 channel_size=[64,64,128,256,512],
                 dropout=True, in_chans=1):
        super(ResNet_sigmoid, self).__init__()
        c = channel_size
        self.inplanes = c[0]

        self.conv1 = nn.Conv3d(in_chans, c[0], kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(c[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, c[1], layers[0])
        self.layer2 = self._make_layer(block, c[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, c[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, c[4], layers[3], stride=2)

        self.att1 = SigmoidBlock(c[1])
        self.att2 = SigmoidBlock(c[2])
        self.att3 = SigmoidBlock(c[3])

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        if dropout is True:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = nn.Identity()

        self.classifier = nn.Linear(c[4] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, origin):

        x = self.conv1(origin)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(self.att1(x1))
        x3 = self.layer3(self.att2(x2))
        x4 = self.layer4(self.att3(x3))
        out = self.avgpool(x4)

        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out_main = self.classifier(out)

        return out_main


class AxialCrossAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.out_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def compute_attention(self, Q, K, V, B, C, axis_len, reshape_shape, reverse_shape):
        Q = Q.reshape(B, C, -1, axis_len)  # (B, C, N, L)
        K = K.reshape(B, C, -1, axis_len)
        V = V.reshape(B, C, -1, axis_len)

        Q = Q.permute(0, 2, 1, 3).reshape(B * Q.shape[2], C, axis_len)  # (B*N, C, L)
        K = K.permute(0, 2, 1, 3).reshape(B * K.shape[2], C, axis_len)
        V = V.permute(0, 2, 1, 3).reshape(B * V.shape[2], C, axis_len)

        attn = torch.bmm(Q.transpose(1, 2), K) / (C ** 0.5)  # (B*N, L, L)
        attn = self.softmax(attn)
        out = torch.bmm(attn, V.transpose(1, 2))  # (B*N, L, C)
        out = out.transpose(1, 2).reshape(B, -1, C, axis_len).permute(0, 2, 1, 3)  # (B, C, N, L)
        out = out.reshape(*reverse_shape)  # Original 3D shape
        return out

    def forward(self, query_feat, kv_feat):
        B, C, D, H, W = query_feat.shape
        Q = self.query_conv(query_feat)
        K = self.key_conv(kv_feat)
        V = self.value_conv(kv_feat)

        # x-axis attention
        qx = Q.permute(0, 1, 3, 4, 2)  # B, C, H, W, D
        kx = K.permute(0, 1, 3, 4, 2)
        vx = V.permute(0, 1, 3, 4, 2)
        out_x = self.compute_attention(qx, kx, vx, B, C, D, (B, C, H*W, D), (B, C, H, W, D))
        out_x = out_x.permute(0, 1, 4, 2, 3)  # B, C, D, H, W

        # y-axis attention
        qy = Q.permute(0, 1, 2, 4, 3)  # B, C, D, W, H
        ky = K.permute(0, 1, 2, 4, 3)
        vy = V.permute(0, 1, 2, 4, 3)
        out_y = self.compute_attention(qy, ky, vy, B, C, H, (B, C, D*W, H), (B, C, D, W, H))
        out_y = out_y.permute(0, 1, 2, 4, 3)  # B, C, D, H, W

        # z-axis attention
        qz = Q  # B, C, D, H, W
        kz = K
        vz = V
        out_z = self.compute_attention(qz, kz, vz, B, C, W, (B, C, D*H, W), (B, C, D, H, W))

        fused = (out_x + out_y + out_z) / 3
        return self.out_conv(fused)


class CBAM3D(nn.Module):
    def __init__(self, in_channels):
        super(CBAM3D, self).__init__()
        self.channel_attention = ChannelAttention3D(in_channels)
        self.spatial_attention = SpatialAttention3D()

    def forward(self, x):
        # Channel Attention
        ch_att = self.channel_attention(x)
        x_ch = x * ch_att

        # Spatial Attention
        sp_att = self.spatial_attention(x_ch)
        x_sp = x_ch * sp_att

        return x_sp


class ResNet_crossattention(nn.Module):
    def __init__(self):
        super().__init__()
        self.t1_encoder = ResNet()
        self.t2_encoder = ResNet()
        self.ratio_encoder = ResNet()

        self.cross_att_t1 = nn.ModuleList([AxialCrossAttentionBlock(c) for c in [64, 128, 256, 512]])
        self.cross_att_t2 = nn.ModuleList([AxialCrossAttentionBlock(c) for c in [64, 128, 256, 512]])

        self.regressor = nn.Sequential(
            nn.Linear(2880, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

        self.cbam = nn.ModuleList([CBAM3D(c) for c in [64*3, 128*3, 256*3, 512*3]])

    def forward(self, x):
        t1, t2, ratio = x[:, 0:1], x[:, 1:2], x[:, 2:3]

        t1_feats = self.t1_encoder.extract_features(t1)
        t2_feats = self.t2_encoder.extract_features(t2)
        ratio_feats = self.ratio_encoder.extract_features(ratio)

        fused_feats = []
        for lvl in range(4):
            t1_att = self.cross_att_t1[lvl](ratio_feats[lvl], t1_feats[lvl])
            t2_att = self.cross_att_t2[lvl](ratio_feats[lvl], t2_feats[lvl])
            ratio_feat = ratio_feats[lvl]

            fused = t1_att + t2_att + ratio_feat[lvl]
            fused = torch.cat([t1_att, t2_att, ratio_feat], dim=1)
            cbam = self.cbam[lvl](fused)
            pooled = F.adaptive_avg_pool3d(cbam, 1).flatten(1)
            fused_feats.append(pooled)

        x = torch.cat(fused_feats, dim=1)
        out = self.regressor(x)
        return out


class ResNet_crossattention_dp(nn.Module):
    def __init__(self):
        super().__init__()
        self.t1_encoder = ResNet()
        self.t2_encoder = ResNet()
        self.ratio_encoder = ResNet()

        self.cross_att_t1 = nn.ModuleList([AxialCrossAttentionBlock(c) for c in [64, 128, 256, 512]])
        self.cross_att_t2 = nn.ModuleList([AxialCrossAttentionBlock(c) for c in [64, 128, 256, 512]])

        self.regressor = nn.Sequential(
            nn.Linear(2880, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

        self.cbam = nn.ModuleList([CBAM3D(c) for c in [64*3, 128*3, 256*3, 512*3]])
        self.aux_regressors = nn.ModuleList([
            nn.Linear(c, 1) for c in [64*3, 128*3, 256*3, 512*3]
        ])


    def forward(self, x):
        t1, t2, ratio = x[:, 0:1], x[:, 1:2], x[:, 2:3]

        t1_feats = self.t1_encoder.extract_features(t1)
        t2_feats = self.t2_encoder.extract_features(t2)
        ratio_feats = self.ratio_encoder.extract_features(ratio)

        fused_feats = []
        aux_preds = []
        for lvl in range(4):
            t1_att = self.cross_att_t1[lvl](ratio_feats[lvl], t1_feats[lvl])
            t2_att = self.cross_att_t2[lvl](ratio_feats[lvl], t2_feats[lvl])
            ratio_feat = ratio_feats[lvl]

            fused = t1_att + t2_att + ratio_feat[lvl]
            fused = torch.cat([t1_att, t2_att, ratio_feat], dim=1)
            cbam = self.cbam[lvl](fused)
            pooled = F.adaptive_avg_pool3d(cbam, 1).flatten(1)
            fused_feats.append(pooled)
            aux_pred = self.aux_regressors[lvl](pooled)
            aux_preds.append(aux_pred)

        x = torch.cat(fused_feats, dim=1)
        out = self.regressor(x)
        return out, aux_preds[0], aux_preds[1], aux_preds[2], aux_preds[3]


class SepConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        #   depthwise and pointwise convolution, downsample by 2
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv3d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv3d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv3d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv3d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)
    
class ResNet_contrast(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1,
                 channel_size=[64,64,128,256,512],
                 dropout=True, in_chans=1):
        super(ResNet_contrast, self).__init__()
        c = channel_size
        self.inplanes = c[0]

        self.conv1 = nn.Conv3d(in_chans, c[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(c[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, c[1], layers[0])
        self.layer2 = self._make_layer(block, c[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, c[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, c[4], layers[3], stride=2)


        self.auxiliary1 = nn.Sequential(
            SepConv(
                channel_in=64 * block.expansion,
                channel_out=128 * block.expansion
            ),
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion
            ),
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion
            ),
            nn.AvgPool3d(4)
        )

        self.auxiliary2 = nn.Sequential(
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion,
            ),
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            nn.AvgPool3d(4)
        )
        self.auxiliary3 = nn.Sequential(
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            nn.AvgPool3d(4)
        )
        self.auxiliary4 = nn.AvgPool3d(4)

        self.fc = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, origin):


        feature_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        feature_list.append(x)
        x = self.layer2(x)
        feature_list.append(x)
        x = self.layer3(x)
        feature_list.append(x)
        x = self.layer4(x)
        feature_list.append(x)

        out1_feature = self.auxiliary1(feature_list[0]).view(x.size(0), -1)
        out2_feature = self.auxiliary2(feature_list[1]).view(x.size(0), -1)
        out3_feature = self.auxiliary3(feature_list[2]).view(x.size(0), -1)
        out4_feature = self.auxiliary4(feature_list[3]).view(x.size(0), -1)
        out = self.fc(out4_feature)
        feat_list = [out4_feature, out3_feature, out2_feature, out1_feature]

        for index in range(len(feat_list)):
            feat_list[index] = F.normalize(feat_list[index], dim=1)
        
        return out, feat_list
