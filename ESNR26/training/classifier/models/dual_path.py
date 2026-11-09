import torch
from torch import nn as nn
from torch.nn import functional as F


class ChannelSELayer3D(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor


class SpatialSELayer3D(nn.Module):
    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels

        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor


class ChannelSpatialSELayer3D(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


class ProjectExciteLayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ProjectExciteLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.relu = nn.ReLU()
        self.conv_c = nn.Conv3d(in_channels=num_channels, out_channels=num_channels_reduced, kernel_size=1, stride=1)
        self.conv_cT = nn.Conv3d(in_channels=num_channels_reduced, out_channels=num_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()

        # Project:
        # Average along channels and different axes
        squeeze_tensor_w = F.adaptive_avg_pool3d(input_tensor, (1, 1, W))
        squeeze_tensor_h = F.adaptive_avg_pool3d(input_tensor, (1, H, 1))
        squeeze_tensor_d = F.adaptive_avg_pool3d(input_tensor, (D, 1, 1))

        # tile tensors to original size and add:
        final_squeeze_tensor = sum([squeeze_tensor_w.view(batch_size, num_channels, 1, 1, W),
                                    squeeze_tensor_h.view(batch_size, num_channels, 1, H, 1),
                                    squeeze_tensor_d.view(batch_size, num_channels, D, 1, 1)])

        # Excitation:
        final_squeeze_tensor = self.sigmoid(self.conv_cT(self.relu(self.conv_c(final_squeeze_tensor))))
        output_tensor = torch.mul(input_tensor, final_squeeze_tensor)

        return output_tensor


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global Average Pooling
        avg_pool = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        avg_pool = avg_pool.view(avg_pool.size(0), -1)

        # Two fully connected layers with ReLU and Sigmoid
        avg_pool = self.fc1(avg_pool)
        avg_pool = self.relu(avg_pool)
        avg_pool = self.fc2(avg_pool)
        avg_pool = self.sigmoid(avg_pool)

        # Rescale the input feature map by the learned attention weights
        avg_pool = avg_pool.view(avg_pool.size(0), avg_pool.size(1), 1, 1, 1)
        return x * avg_pool


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None, se_block=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        if se_block == 'cse':
            self.se_block = ChannelSELayer3D(planes)
        elif se_block == 'sse':
            self.se_block = SpatialSELayer3D(planes)
        elif se_block == 'csse':
            self.se_block = ChannelSpatialSELayer3D(planes)
        elif se_block == 'pse':
            self.se_block = ProjectExciteLayer(planes)
        elif se_block == 'se':
            self.se_block = SEBlock(planes)
        else:
            self.se_block = False

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.se_block:
            out = self.se_block(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class Encoder3D(nn.Module):
    def __init__(self, in_chans=1, base_channels=64, block=BasicBlock, layers=[2, 2, 2, 2], use_se=False):
        super().__init__()
        self.inplanes = base_channels
        self.conv1 = nn.Conv3d(in_chans, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, base_channels, layers[0], use_se=False)
        self.layer2 = self._make_layer(block, base_channels*2, layers[1], stride=2, use_se=use_se)
        self.layer3 = self._make_layer(block, base_channels*4, layers[2], stride=2, use_se=use_se)
        self.layer4 = self._make_layer(block, base_channels*8, layers[3], stride=2, use_se=use_se)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _make_layer(self, block, planes, blocks, stride=1, use_se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )

        layers = [block(self.inplanes, planes, stride, downsample, se_block=use_se)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, se_block=use_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class DualPathBrainAge(nn.Module):
    def __init__(self, global_in_chans=2, local_in_chans=1, num_classes=1, use_se_local=False):
        super().__init__()
        self.global_encoder = Encoder3D(in_chans=global_in_chans)
        self.local_encoder = Encoder3D(in_chans=local_in_chans, use_se=use_se_local)

        feature_dim = 512  # Assuming output of each encoder is 512 after flattening
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1)
        )


    def forward(self, x):
        global_x, local_x = x[:, 0:2], x[:, 2:3]

        global_feat = self.global_encoder(global_x)
        local_feat = self.local_encoder(local_x)

        fused = torch.cat([global_feat, local_feat], dim=1)

        ## -- check sample size -- ##

        out = self.classifier(fused)
        return out


class DualPathBrainAge_info(nn.Module):
    def __init__(self, global_in_chans=2, local_in_chans=1, num_classes=1, use_aux_info=1, use_se_local=False):
        super().__init__()
        self.use_aux_info = use_aux_info
        self.global_encoder = Encoder3D(in_chans=global_in_chans)
        self.local_encoder = Encoder3D(in_chans=local_in_chans, use_se=use_se_local)

        feature_dim = 512  # Assuming output of each encoder is 512 after flattening
        self.classifier = nn.Linear(feature_dim * 2 + self.use_aux_info, num_classes)


    def forward(self, x, info):
        global_x, local_x = x[:, 0:2], x[:, 2:3]

        global_feat = self.global_encoder(global_x)
        local_feat = self.local_encoder(local_x)

        fused = torch.cat([global_feat, local_feat], dim=1)

        if self.use_aux_info == 1:
            cond = info.reshape(-1, 1).float()
        else:
            wm, icv = info
            wm = wm.reshape(-1, 1).float()
            icv = icv.reshape(-1, 1).float()
            cond = torch.concat((wm, icv), dim=1)  # [B, 2]

        fused = torch.cat([fused, cond], dim=1)

        out = self.classifier(fused)
        return out


class DualPathBrainAge_origin(nn.Module):
    def __init__(self, global_in_chans=2, local_in_chans=1, num_classes=1, use_se_local=False):
        super().__init__()
        self.global_encoder = Encoder3D(in_chans=global_in_chans)
        self.local_encoder = Encoder3D(in_chans=local_in_chans, use_se=use_se_local)

        feature_dim = 512  # Assuming output of each encoder is 512 after flattening
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1)
        )


    def forward(self, x):
        global_feat = self.global_encoder(x)
        local_feat = self.local_encoder(x)

        fused = torch.cat([global_feat, local_feat], dim=1)
        out = self.classifier(fused)
        return out


class DualPathBrainAge_origin_info(nn.Module):
    def __init__(self, global_in_chans=2, local_in_chans=1, num_classes=1, use_aux_info=1, use_se_local=False):
        super().__init__()
        self.use_aux_info = use_aux_info
        self.global_encoder = Encoder3D(in_chans=global_in_chans)
        self.local_encoder = Encoder3D(in_chans=local_in_chans, use_se=use_se_local)

        feature_dim = 512  # Assuming output of each encoder is 512 after flattening
        self.classifier = nn.Linear(feature_dim * 2 + self.use_aux_info, num_classes)


    def forward(self, x, info):
        global_feat = self.global_encoder(x)
        local_feat = self.local_encoder(x)

        fused = torch.cat([global_feat, local_feat], dim=1)

        if self.use_aux_info == 1:
            cond = info.reshape(-1, 1).float()
        else:
            wm, icv = info
            wm = wm.reshape(-1, 1).float()
            icv = icv.reshape(-1, 1).float()
            cond = torch.concat((wm, icv), dim=1)  # [B, 2]

        fused = torch.cat([fused, cond], dim=1)

        out = self.classifier(fused)
        return out


# Example usage
if __name__ == '__main__':
    model = DualPathBrainAge(global_in_chans=2, local_in_chans=1)
    x_global = torch.randn(2, 2, 86, 94, 84)  # T1 or T2 whole brain
    x_local = torch.randn(2, 1, 86, 94, 84)   # RI-masked ratio map or ROI crop
    aux = torch.tensor([[0.7, 1.2], [0.5, 1.0]])  # WM, ICV
    
    out = model(x_global, x_local, aux)
    print(out.shape)  # [2, 1]
