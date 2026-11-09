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
    
    def forward(self, x):

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        out = self.avgpool(x4)

        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        pred = self.classifier(out)

        return pred


class InitialEncoder(nn.Module):
    """ResNet stem: Conv → BN → ReLU → MaxPool"""
    def __init__(self, in_ch, out_ch=64):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=7, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        return x

class SpatialAttention(nn.Module):
    """Ratio as Query, T1/T2 as Key-Value for spatial attention"""
    def __init__(self, in_ch, mid_ch):
        super().__init__()
        self.q_conv = nn.Conv3d(in_ch, mid_ch, kernel_size=1)
        self.k_conv = nn.Conv3d(in_ch, mid_ch, kernel_size=1)
        self.v_conv = nn.Conv3d(in_ch, mid_ch, kernel_size=1)

    def forward(self, q_feat, kv_feat):
        B, C, D, H, W = q_feat.shape
        q = self.q_conv(q_feat).view(B, -1, D*H*W).permute(0, 2, 1)     # [B, N, C]
        k = self.k_conv(kv_feat).view(B, -1, D*H*W)                     # [B, C, N]
        v = self.v_conv(kv_feat).view(B, -1, D*H*W).permute(0, 2, 1)    # [B, N, C]

        attn = torch.bmm(q, k) / (k.shape[1] ** 0.5)   # [B, N, N]
        attn = torch.softmax(attn, dim=-1)

        weighted = torch.bmm(attn, v)                 # [B, N, C]
        weighted = weighted.permute(0, 2, 1).view(B, -1, D, H, W)
        return weighted + q_feat


class SpatialAttention_ver1(nn.Module):
    """Ratio as Query, T1/T2 as Key-Value for spatial attention with downsampling"""
    def __init__(self, in_ch, mid_ch):
        super().__init__()
        self.q_conv = nn.Conv3d(in_ch, mid_ch, kernel_size=1)
        self.k_conv = nn.Conv3d(in_ch, mid_ch, kernel_size=1)
        self.v_conv = nn.Conv3d(in_ch, mid_ch, kernel_size=1)
        
        # Downsampling layer
        self.downsample = nn.Conv3d(in_ch, in_ch, kernel_size=3, stride=2, padding=1)
        
        # Upsampling layer to restore original size
        self.upsample = nn.ConvTranspose3d(mid_ch, in_ch, kernel_size=3, stride=2, padding=1, output_padding=0)

    def forward(self, q_feat, kv_feat):
        # Apply downsampling
        q_feat_ds = self.downsample(q_feat)  # Downsample input Q feature map
        kv_feat_ds = self.downsample(kv_feat)  # Downsample input KV feature map

        B, C, D, H, W = q_feat_ds.shape

        q = self.q_conv(q_feat_ds).view(B, 1, -1).permute(0, 2, 1)  # [B, N, C]
        k = self.k_conv(kv_feat_ds).view(B, 1, -1)  # [B, C, N]
        v = self.v_conv(kv_feat_ds).view(B, 1, -1).permute(0, 2, 1)  # [B, N, C]

        attn = torch.bmm(q, k) / (k.shape[1] ** 0.5)  # [B, N, N]
        attn = torch.softmax(attn, dim=-1)

        weighted = torch.bmm(attn, v)  # [B, N, C]
        weighted = weighted.permute(0, 2, 1).view(B, -1, D, H, W)  # Upsample dimensions to match
        weighted = self.upsample(weighted)  # Restore original spatial size

        return weighted + q_feat  # residual connection


class SpatialAttention_ver2(nn.Module):
    """Ratio as Query, T1/T2 as Key-Value for spatial attention with downsampling"""
    def __init__(self, in_ch, mid_ch):
        super().__init__()
        self.q_conv = nn.Conv3d(in_ch, mid_ch, kernel_size=1)
        self.k_conv = nn.Conv3d(in_ch, mid_ch, kernel_size=1)
        self.v_conv = nn.Conv3d(in_ch, in_ch, kernel_size=1)
        
        # Downsampling layer
        self.downsample = nn.Conv3d(in_ch, in_ch, kernel_size=3, stride=2, padding=1)
        
        # Upsampling layer to restore original size
        self.upsample = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, output_padding=0)

    def forward(self, q_feat, kv_feat):
        _, C, _, _, _ = q_feat.shape
        
        # Apply downsampling
        q_feat_ds = self.downsample(q_feat)  # Downsample input Q feature map
        kv_feat_ds = self.downsample(kv_feat)  # Downsample input KV feature map

        B, _, D, H, W = q_feat_ds.shape

        q = self.q_conv(q_feat_ds).view(B, 1, -1).permute(0, 2, 1)  # [B, N, C]
        k = self.k_conv(kv_feat_ds).view(B, 1, -1)  # [B, C, N]
        v = self.v_conv(kv_feat_ds).view(B, C, -1).permute(0, 2, 1)  # [B, N, C]

        attn = torch.bmm(q, k) / (k.shape[1] ** 0.5)  # [B, N, N]
        attn = torch.softmax(attn, dim=-1)

        weighted = torch.bmm(attn, v)  # [B, N, C]
        weighted = weighted.permute(0, 2, 1).view(B, -1, D, H, W)  # Upsample dimensions to match
        weighted = self.upsample(weighted)  # Restore original spatial size

        return weighted + q_feat  # residual connection


class SpatialAttention_ver11(nn.Module):
    """Ratio as Query, T1/T2 as Key-Value for spatial attention with downsampling"""
    def __init__(self, in_ch, mid_ch):
        super().__init__()
        self.q_conv = nn.Conv3d(in_ch, mid_ch, kernel_size=1)
        self.k_conv = nn.Conv3d(in_ch, mid_ch, kernel_size=1)
        self.v_conv = nn.Conv3d(in_ch, mid_ch, kernel_size=1)
        
        # Downsampling layer
        self.downsample = nn.Conv3d(in_ch, in_ch, kernel_size=3, stride=2, padding=1)
        
        # Upsampling layer to restore original size
        self.upsample = nn.ConvTranspose3d(mid_ch, in_ch, kernel_size=3, stride=2, padding=1, output_padding=0)

    def forward(self, q_feat, kv_feat):
        # Apply downsampling
        q_feat_ds = self.downsample(q_feat)  # Downsample input Q feature map
        kv_feat_ds = self.downsample(kv_feat)  # Downsample input KV feature map

        B, C, D, H, W = q_feat_ds.shape

        q = self.q_conv(q_feat_ds).view(B, 1, -1).permute(0, 2, 1)  # [B, N, C]
        k = self.k_conv(kv_feat_ds).view(B, 1, -1)  # [B, C, N]
        v = self.v_conv(kv_feat_ds).view(B, 1, -1).permute(0, 2, 1)  # [B, N, C]

        attn = torch.bmm(q, k) / (k.shape[1] ** 0.5)  # [B, N, N]
        attn = torch.softmax(attn, dim=-1)

        weighted = torch.bmm(attn, v)  # [B, N, C]
        weighted = weighted.permute(0, 2, 1).view(B, -1, D, H, W)  # Upsample dimensions to match
        weighted = self.upsample(weighted)  # Restore original spatial size

        return weighted


class SpatialAttention_ver22(nn.Module):
    """Ratio as Query, T1/T2 as Key-Value for spatial attention with downsampling"""
    def __init__(self, in_ch, mid_ch):
        super().__init__()
        self.q_conv = nn.Conv3d(in_ch, mid_ch, kernel_size=1)
        self.k_conv = nn.Conv3d(in_ch, mid_ch, kernel_size=1)
        self.v_conv = nn.Conv3d(in_ch, in_ch, kernel_size=1)
        
        # Downsampling layer
        self.downsample = nn.Conv3d(in_ch, in_ch, kernel_size=3, stride=2, padding=1)
        
        # Upsampling layer to restore original size
        self.upsample = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, output_padding=0)

    def forward(self, q_feat, kv_feat):
        _, C, _, _, _ = q_feat.shape
        
        # Apply downsampling
        q_feat_ds = self.downsample(q_feat)  # Downsample input Q feature map
        kv_feat_ds = self.downsample(kv_feat)  # Downsample input KV feature map

        B, _, D, H, W = q_feat_ds.shape

        q = self.q_conv(q_feat_ds).view(B, 1, -1).permute(0, 2, 1)  # [B, N, C]
        k = self.k_conv(kv_feat_ds).view(B, 1, -1)  # [B, C, N]
        v = self.v_conv(kv_feat_ds).view(B, C, -1).permute(0, 2, 1)  # [B, N, C]

        attn = torch.bmm(q, k) / (k.shape[1] ** 0.5)  # [B, N, N]
        attn = torch.softmax(attn, dim=-1)

        weighted = torch.bmm(attn, v)  # [B, N, C]
        weighted = weighted.permute(0, 2, 1).view(B, -1, D, H, W)  # Upsample dimensions to match
        weighted = self.upsample(weighted)  # Restore original spatial size

        return weighted


class FusionConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class MultiModalAttentionResNet(nn.Module):
    def __init__(self, base_block=BasicBlock, layers=[2,2,2,2], att_op=1):
        super().__init__()
        self.initial_t1 = InitialEncoder(1)
        self.initial_t2 = InitialEncoder(1)
        self.initial_ratio = InitialEncoder(1)

        if att_op == 1:
            self.attn_t1 = SpatialAttention_ver1(in_ch=64, mid_ch=1)
            self.attn_t2 = SpatialAttention_ver1(in_ch=64, mid_ch=1)
        if att_op == 2:
            self.attn_t1 = SpatialAttention_ver2(in_ch=64, mid_ch=1)
            self.attn_t2 = SpatialAttention_ver2(in_ch=64, mid_ch=1)

        # Fusion + projection to original ResNet input channel
        self.fusion = FusionConv(in_ch=64*2, out_ch=64)

        # Backbone ResNet
        self.resnet_body = ResNet(block=base_block, layers=layers, num_classes=1, in_chans=64)

    def forward(self, x):
        t1, t2, ratio = x[:, 0:1], x[:, 1:2], x[:, 2:3]

        t1_feat = self.initial_t1(t1)
        t2_feat = self.initial_t2(t2)
        ratio_feat = self.initial_ratio(ratio)

        attn_t1 = self.attn_t1(ratio_feat, t1_feat)
        attn_t2 = self.attn_t2(ratio_feat, t2_feat)

        fused = torch.cat([attn_t1, attn_t2], dim=1)
        fused = self.fusion(fused)

        out = self.resnet_body(fused)
        return out


class MultiModalAttentionResNet2(nn.Module):
    def __init__(self, base_block=BasicBlock, layers=[2,2,2,2], att_op=1):
        super().__init__()
        self.initial_t1 = InitialEncoder(1)
        self.initial_t2 = InitialEncoder(1)
        self.initial_ratio = InitialEncoder(1)

        if att_op == 1:
            self.attn_t1 = SpatialAttention_ver11(in_ch=64, mid_ch=1)
            self.attn_t2 = SpatialAttention_ver11(in_ch=64, mid_ch=1)
        if att_op == 2:
            self.attn_t1 = SpatialAttention_ver22(in_ch=64, mid_ch=1)
            self.attn_t2 = SpatialAttention_ver22(in_ch=64, mid_ch=1)

        # Fusion + projection to original ResNet input channel
        self.fusion = FusionConv(in_ch=64*3, out_ch=64)

        # Backbone ResNet
        self.resnet_body = ResNet(block=base_block, layers=layers, num_classes=1, in_chans=64)

    def forward(self, x):
        t1, t2, ratio = x[:, 0:1], x[:, 1:2], x[:, 2:3]

        t1_feat = self.initial_t1(t1)
        t2_feat = self.initial_t2(t2)
        ratio_feat = self.initial_ratio(ratio)

        attn_t1 = self.attn_t1(ratio_feat, t1_feat)
        attn_t2 = self.attn_t2(ratio_feat, t2_feat)

        fused = torch.cat([attn_t1, ratio_feat, attn_t2], dim=1)
        fused = self.fusion(fused)

        out = self.resnet_body(fused)
        return out


class MultiModalAttentionResNet_reverse(nn.Module):
    def __init__(self, base_block=BasicBlock, layers=[2,2,2,2], att_op=1):
        super().__init__()
        self.initial_t1 = InitialEncoder(1)
        self.initial_t2 = InitialEncoder(1)
        self.initial_ratio = InitialEncoder(1)

        if att_op == 1:
            self.attn_t1 = SpatialAttention_ver1(in_ch=64, mid_ch=1)
            self.attn_t2 = SpatialAttention_ver1(in_ch=64, mid_ch=1)
        if att_op == 2:
            self.attn_t1 = SpatialAttention_ver2(in_ch=64, mid_ch=1)
            self.attn_t2 = SpatialAttention_ver2(in_ch=64, mid_ch=1)

        # Fusion + projection to original ResNet input channel
        self.fusion = FusionConv(in_ch=64*2, out_ch=64)

        # Backbone ResNet
        self.resnet_body = ResNet(block=base_block, layers=layers, num_classes=1, in_chans=64)

    def forward(self, x):
        t1, t2, ratio = x[:, 0:1], x[:, 1:2], x[:, 2:3]

        t1_feat = self.initial_t1(t1)
        t2_feat = self.initial_t2(t2)
        ratio_feat = self.initial_ratio(ratio)

        attn_t1 = self.attn_t1(t1_feat, ratio_feat)
        attn_t2 = self.attn_t2(t2_feat, ratio_feat)

        fused = torch.cat([attn_t1, attn_t2], dim=1)
        fused = self.fusion(fused)

        out = self.resnet_body(fused)
        return out


class MultiModalAttentionResNet_reverse2(nn.Module):
    def __init__(self, base_block=BasicBlock, layers=[2,2,2,2], att_op=1):
        super().__init__()
        self.initial_t1 = InitialEncoder(1)
        self.initial_t2 = InitialEncoder(1)
        self.initial_ratio = InitialEncoder(1)

        if att_op == 1:
            self.attn_t1 = SpatialAttention_ver11(in_ch=64, mid_ch=1)
            self.attn_t2 = SpatialAttention_ver11(in_ch=64, mid_ch=1)
        if att_op == 2:
            self.attn_t1 = SpatialAttention_ver22(in_ch=64, mid_ch=1)
            self.attn_t2 = SpatialAttention_ver22(in_ch=64, mid_ch=1)

        # Fusion + projection to original ResNet input channel
        self.fusion = FusionConv(in_ch=64*4, out_ch=64)

        # Backbone ResNet
        self.resnet_body = ResNet(block=base_block, layers=layers, num_classes=1, in_chans=64)

    def forward(self, x):
        t1, t2, ratio = x[:, 0:1], x[:, 1:2], x[:, 2:3]

        t1_feat = self.initial_t1(t1)
        t2_feat = self.initial_t2(t2)
        ratio_feat = self.initial_ratio(ratio)

        attn_t1 = self.attn_t1(t1_feat, ratio_feat)
        attn_t2 = self.attn_t2(t2_feat, ratio_feat)

        fused = torch.cat([attn_t1, attn_t2, t1_feat, t2_feat], dim=1)
        fused = self.fusion(fused)

        out = self.resnet_body(fused)
        return out
