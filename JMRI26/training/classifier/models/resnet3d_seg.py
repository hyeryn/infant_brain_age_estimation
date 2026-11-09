import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

# Based on torchvision implementation 
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1,
                 channel_size=[64,64,128,256,512],
                 dropout=False, in_chans=1):
        c = channel_size
        self.inplanes = c[0]
        super(ResNet, self).__init__()

        net = nn.Sequential()
        net.add_module('conv', ConvBlock(in_chans, c[0]))
        net.add_module('conv1', nn.Conv3d( c[0], c[0], kernel_size=7, stride=2, padding=0,
                               bias=False))
        net.add_module('bn1', nn.BatchNorm3d(c[0]))
        net.add_module('relu', nn.ReLU(inplace=True))
        net.add_module('maxpool', nn.MaxPool3d(kernel_size=3, stride=2, padding=1))
        net.add_module('layer1', self._make_layer(block, c[1], layers[0]))
        net.add_module('layer2', self._make_layer(block, c[2], layers[1], stride=2))
        net.add_module('layer3', self._make_layer(block, c[3], layers[2], stride=2))
        net.add_module('layer4', self._make_layer(block, c[4], layers[3], stride=2))
        net.add_module('avgpool', nn.AdaptiveAvgPool3d((1, 1, 1))) 
        if dropout is True:
            net.add_module('dropout', nn.Dropout(0.5))
        self.feature_extractor = net
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
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    Args:
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    Args:
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    Args:
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    Args:
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    Args:
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.resnet = resnet18(in_chans=in_channels, dropout=False)

        self.conv = self.resnet.feature_extractor.conv

        # Extract intermediate layers from the ResNet
        self.initial = nn.Sequential(
            self.resnet.feature_extractor.conv1,
            self.resnet.feature_extractor.bn1,
            self.resnet.feature_extractor.relu,
            self.resnet.feature_extractor.maxpool
        )
        self.layer1 = self.resnet.feature_extractor.layer1
        self.layer2 = self.resnet.feature_extractor.layer2
        self.layer3 = self.resnet.feature_extractor.layer3
        self.layer4 = self.resnet.feature_extractor.layer4

    def forward(self, x):
        x_conv = self.conv(x)
        x0 = self.initial(x_conv)    # After conv1 and maxpool
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x_conv, x0, x1, x2, x3, x4]  # Return multi-scale features for deep supervision


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        if in_ch != out_ch:
            self.conv = ConvBlock(in_ch, out_ch)
        else:
            self.conv = ConvBlock(in_ch * 2, out_ch)

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
        return self.conv(x)


class AgeRegressor(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.fc = nn.Linear(in_ch, 1)

    def forward(self, x):
        return self.fc(x).squeeze(1)


class ResNet_AgeSegRecon(nn.Module):
    def __init__(self, in_channels=1, base_ch=64, num_classes=1):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels)

        self.up4 = UpBlock(base_ch * 8, base_ch * 4) 
        self.up3 = UpBlock(base_ch * 4, base_ch * 2)
        self.up2 = UpBlock(base_ch * 2, base_ch)
        self.up1 = UpBlock(base_ch, base_ch)
        self.up0 = UpBlock(base_ch, base_ch)

        # Heads
        self.seg_head = nn.Conv3d(base_ch, num_classes, kernel_size=1)
        self.recon_head = nn.Conv3d(base_ch, in_channels, kernel_size=1)

        self.age_regressors = nn.ModuleList([
            AgeRegressor(base_ch * 8),
            AgeRegressor(base_ch * 4),
            AgeRegressor(base_ch * 2),
            AgeRegressor(base_ch),
        ])

    def forward(self, x):
        feats = self.encoder(x)  # [x1, x2, x3, x4]
        x_conv, x0, x1, x2, x3, x4 = feats

        d3 = self.up4(x4, x3)
        d2 = self.up3(d3, x2)
        d1 = self.up2(d2, x1)
        d0 = self.up1(d1, x0)  # Use x1 again for deeper supervision consistency
        dconv = self.up0(d0, x_conv)

        seg_out = self.seg_head(dconv)
        recon_out = self.recon_head(dconv)

        # Age predictions from encoder features
        age_preds = []
        for f, reg in zip([x4, x3, x2, x1], self.age_regressors):
            pooled = F.adaptive_avg_pool3d(f, (1, 1, 1)).view(f.size(0), -1)
            age_preds.append(reg(pooled))

        return seg_out, recon_out, age_preds
