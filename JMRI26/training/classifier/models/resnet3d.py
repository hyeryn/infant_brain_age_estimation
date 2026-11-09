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
        net.add_module('conv1', nn.Conv3d(in_chans, c[0], kernel_size=7, stride=2, padding=0,
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


class ResNet_add(nn.Module):
    def __init__(self, block, layers, num_classes=1,
                 channel_size=[64,64,128,256,512],
                 dropout=False, in_chans=1):
        c = channel_size
        self.inplanes = c[0]
        super(ResNet_add, self).__init__()

        net = nn.Sequential()
        net.add_module('conv1', nn.Conv3d(in_chans, c[0], kernel_size=7, stride=2, padding=0,
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

    def forward(self, x, info):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        
        if self.option == 1:
            out = torch.concat((x,info.reshape(-1, 1)),dim=1)
            out_main = self.classifier(out.float())
        else:
            wm, icv = info
            wm = wm.reshape(-1, 1).float()
            icv = icv.reshape(-1, 1).float()
            wm_icv = torch.concat((wm, icv), dim=1)
            out = torch.concat((x,wm_icv),dim=1)
            out_main = self.classifier(out)

        return out_main


def resnet18_add(**kwargs):
    model = ResNet_add(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34_add(**kwargs):
    model = ResNet_add(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50_add(**kwargs):
    model = ResNet_add(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101_add(**kwargs):
    model = ResNet_add(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152_add(**kwargs):
    model = ResNet_add(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


class FiLMBlock(nn.Module):
    def __init__(self, feature_dim, condition_dim):
        super(FiLMBlock, self).__init__()
        self.gamma_fc = nn.Linear(condition_dim, feature_dim)
        self.beta_fc = nn.Linear(condition_dim, feature_dim)

    def forward(self, x, cond):
        gamma = self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta         # gamma = scale, bata = shift  (like FiLM)


class ResNet_FiLM(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1,
                 channel_size=[64, 64, 128, 256, 512],
                 dropout=True, in_chans=1, option=1):
        super(ResNet_FiLM, self).__init__()
        c = channel_size
        self.inplanes = c[0]
        self.option = option

        self.conv1 = nn.Conv3d(in_chans, c[0], kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(c[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, c[1], layers[0])
        self.layer2 = self._make_layer(block, c[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, c[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, c[4], layers[3], stride=2)

        # FiLM blocks for mid-level and deep-level feature modulation
        self.film2 = FiLMBlock(c[2]* block.expansion, option)  # for layer2
        self.film3 = FiLMBlock(c[3]* block.expansion, option)  # for layer3

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()

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
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, origin, info):
        if self.option == 1:
            cond = info.reshape(-1, 1).float()
        else:
            wm, icv = info
            wm = wm.reshape(-1, 1).float()
            icv = icv.reshape(-1, 1).float()
            cond = torch.concat((wm, icv), dim=1)  # [B, 2]

        x = self.conv1(origin)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.film2(self.layer2(x1), cond)  # FiLM on layer2 output
        x3 = self.film3(self.layer3(x2), cond)  # FiLM on layer3 output
        x4 = self.layer4(x3)

        out = self.avgpool(x4)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out_main = self.classifier(out)

        return out_main


def resnet18_film(**kwargs):
    model = ResNet_FiLM(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34_film(**kwargs):
    model = ResNet_FiLM(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50_film(**kwargs):
    model = ResNet_FiLM(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101_film(**kwargs):
    model = ResNet_FiLM(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152_film(**kwargs):
    model = ResNet_FiLM(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.resnet = resnet18(in_chans=in_channels, dropout=False)

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
        x0 = self.initial(x)    # After conv1 and maxpool
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x0, x1, x2, x3, x4]  # Return multi-scale features for deep supervision


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
        x0, x1, x2, x3, x4 = feats

        d3 = self.up4(x4, x3)
        d2 = self.up3(d3, x2)
        d1 = self.up2(d2, x1)
        d0 = self.up1(d1, x0)  # Use x1 again for deeper supervision consistency

        seg_out = self.seg_head(d0)
        recon_out = self.recon_head(d0)

        # Age predictions from encoder features
        age_preds = []
        for f, reg in zip([x4, x3, x2, x1], self.age_regressors):
            pooled = F.adaptive_avg_pool3d(f, (1, 1, 1)).view(f.size(0), -1)
            age_preds.append(reg(pooled))

        return seg_out, recon_out, age_preds


class ResNetAttention(nn.Module):
    def __init__(self, block, layers, num_classes=1,
                 channel_size=[64,64,128,256,512],
                 dropout=False, in_chans=1):
        super(ResNetAttention, self).__init__()
        c = channel_size
        self.inplanes = c[0]

        self.stage1 = nn.Sequential(
            nn.Conv3d(in_chans, c[0], kernel_size=7, stride=2, padding=0, bias=False),
            nn.BatchNorm3d(c[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(block, c[1], layers[0])
        self.layer2 = self._make_layer(block, c[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, c[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, c[4], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()
        self.classifier = nn.Linear(c[4] * block.expansion, num_classes)

        self._init_weights()

    def _init_weights(self):
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

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        x: input tensor with last channel as WM mask (B, C+1, D, H, W)
        - First C channels: image
        - Last channel: white matter mask (0~1)
        """
        channel = x.shape[1] - 1
        img = x[:, :channel]                  # [B, C, D, H, W]
        wm_mask = x[:, -1]                    # [B, D, H, W]

        feat = self.stage1(img)              # [B, C', D', H', W']

        # 3. WM mask downsample to feature size
        wm_mask_ds = F.interpolate(wm_mask.unsqueeze(1), size=feat.shape[2:], mode='trilinear', align_corners=False)  # [B, 1, D', H', W']

        soft_att = torch.sigmoid(wm_mask_ds)  # [B, 1, D', H', W']

        feat = feat * (1 + soft_att)

        x = self.layer1(feat)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


def resnet18_attention(**kwargs):
    return ResNetAttention(BasicBlock, [2, 2, 2, 2], **kwargs)

img = torch.randn(2, 2, 64, 128, 128)          # input image
model = resnet18_attention(in_chans=1)
output = model(img)
