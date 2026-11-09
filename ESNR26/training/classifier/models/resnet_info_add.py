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


class ResNet_add(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1,
                 channel_size=[64,64,128,256,512],
                 dropout=True, in_chans=1, option=1):
        super(ResNet_add, self).__init__()
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

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        if dropout is True:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = nn.Identity()

        self.classifier = nn.Linear(c[4] * block.expansion + self.option, num_classes)

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
    
    def forward(self, origin, info):

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

        if self.option == 1:
            out = torch.concat((out,info.reshape(-1, 1)),dim=1)
            out_main = self.classifier(out.float())
        else:
            wm, icv = info
            wm = wm.reshape(-1, 1).float()
            icv = icv.reshape(-1, 1).float()
            wm_icv = torch.concat((wm, icv), dim=1)
            out = torch.concat((out,wm_icv),dim=1)
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


class FCNet(nn.Module):
    def __init__(self, option=1):
        super(FCNet, self).__init__()

        self.option = option

        self.classifier = nn.Sequential(
            nn.Linear(self.option, 32),
            nn.Dropout(0.3),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
            )

    def forward(self, origin, info):

        if self.option == 1:
            out_main = self.classifier(info.reshape(-1,1).float())
        else:
            wm, icv = info
            wm = wm.reshape(-1, 1).float()
            icv = icv.reshape(-1, 1).float()
            wm_icv = torch.concat((wm, icv), dim=1)
            out_main = self.classifier(wm_icv)

        return out_main


# Define the FiLM layer
class FiLMBlock(nn.Module):
    def __init__(self, feature_dim, condition_dim):
        super(FiLMBlock, self).__init__()
        self.gamma_fc = nn.Linear(condition_dim, feature_dim)
        self.beta_fc = nn.Linear(condition_dim, feature_dim)

    def forward(self, x, cond):
        gamma = self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta         # gamma = scale, bata = shift  (like FiLM)


# Modified ResNet_add with FiLM applied to mid-level features
class ResNet_add_FiLM(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1,
                 channel_size=[64, 64, 128, 256, 512],
                 dropout=True, in_chans=1, option=1):
        super(ResNet_add_FiLM, self).__init__()
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
        self.film2 = FiLMBlock(c[2], option)  # for layer2
        self.film3 = FiLMBlock(c[3], option)  # for layer3

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
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out_main = self.classifier(out)

        return out_main


class ResNet_add_FiLM2(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1,
                 channel_size=[64,64,128,256,512],
                 dropout=True, in_chans=1):
        super(ResNet_add_FiLM2, self).__init__()
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
        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()

        self.classifier = nn.Linear(c[4] * block.expansion, num_classes)

        # FiLM modules
        self.film_wm = FiLMBlock(c[1], 1)   # layer1
        self.film_icv = FiLMBlock(c[3], 1)  # layer3

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
        # info: tuple of (wm_volume, icv_volume), both shaped (B,)
        wm, icv = info
        wm = wm.unsqueeze(1).float()  # (B, 1)
        icv = icv.unsqueeze(1).float()

        x = self.conv1(origin)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x1 = self.film_wm(x1, wm)

        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x3 = self.film_icv(x3, icv)

        x4 = self.layer4(x3)
        out = self.avgpool(x4)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out_main = self.classifier(out)
        return out_main


class FiLMBlock2(nn.Module):
    def __init__(self, feature_dim, condition_dim):
        super(FiLMBlock2, self).__init__()
        self.gamma_fc = nn.Linear(condition_dim, feature_dim)
        self.beta_fc = nn.Linear(condition_dim, feature_dim)

    def forward(self, x, icv, wm):
        gamma = self.gamma_fc(icv).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_fc(wm).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta         # gamma = scale, bata = shift  (like FiLM)


class ResNet_add_FiLM3(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1,
                 channel_size=[64,64,128,256,512],
                 dropout=True, in_chans=1):
        super(ResNet_add_FiLM3, self).__init__()
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
        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()

        self.classifier = nn.Linear(c[4] * block.expansion, num_classes)

        # FiLM modules
        self.film_wm = FiLMBlock2(c[1], 1)   # layer1
        self.film_icv = FiLMBlock2(c[3], 1)  # layer3

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
        # info: tuple of (wm_volume, icv_volume), both shaped (B,)
        wm, icv = info
        wm = wm.unsqueeze(1).float()  # (B, 1)
        icv = icv.unsqueeze(1).float()

        x = self.conv1(origin)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x1 = self.film_wm(x1, icv,wm)

        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x3 = self.film_icv(x3, icv,wm)

        x4 = self.layer4(x3)
        out = self.avgpool(x4)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out_main = self.classifier(out)
        return out_main
    
class BasicBlock_FiLM(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, cond_dim=1):
        super(BasicBlock_FiLM, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.film = FiLMBlock(planes, cond_dim)

        self.downsample = downsample

    def forward(self, x, cond):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)


        out = self.film(out, cond)
        out = self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return out


class ResNet_add_FiLM4(nn.Module):
    def __init__(self, layers=[2, 2, 2, 2], num_classes=1,
                 channel_size=[64,64,128,256,512],
                 dropout=True, in_chans=1):
        super(ResNet_add_FiLM4, self).__init__()
        c = channel_size
        self.inplanes = c[0]

        self.conv1 = nn.Conv3d(in_chans, c[0], kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(c[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock_FiLM, c[1], layers[0], cond_dim=1)
        self.layer2 = self._make_layer(BasicBlock_FiLM, c[2], layers[1], stride=2, cond_dim=1)
        self.layer3 = self._make_layer(BasicBlock_FiLM, c[3], layers[2], stride=2, cond_dim=1)
        self.layer4 = self._make_layer(BasicBlock_FiLM, c[4], layers[3], stride=2, cond_dim=1)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()
        self.classifier = nn.Linear(c[4], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, cond_dim=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, cond_dim=cond_dim))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, cond_dim=cond_dim))

        return nn.ModuleList(layers)

    def forward(self, origin, info):
        wm, icv = info
        wm = wm.unsqueeze(1).float()  # (B, 1)
        icv = icv.unsqueeze(1).float()

        x = self.conv1(origin)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for block in self.layer1:
            x = block(x, wm)

        for block in self.layer2:
            x = block(x, wm)  # no condition

        for block in self.layer3:
            x = block(x, icv)

        for block in self.layer4:
            x = block(x, icv)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        out = self.classifier(x)
        return out


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim_kv // heads) ** -0.5

        self.to_q = nn.Linear(dim_q, dim_kv)
        self.to_k = nn.Linear(dim_kv, dim_kv)
        self.to_v = nn.Linear(dim_kv, dim_kv)
        self.to_out = nn.Linear(dim_kv, dim_kv)

    def forward(self, q, kv, return_attn=False):
        """
        q: (B, 1, D_q)
        kv: (B, N, D_kv)
        """
        B, N, D = kv.shape

        q = self.to_q(q)  # (B, 1, D)
        k = self.to_k(kv)
        v = self.to_v(kv)

        # Multihead reshape
        q = q.view(B, 1, self.heads, D // self.heads).transpose(1, 2)  # (B, h, 1, d)
        k = k.view(B, N, self.heads, D // self.heads).transpose(1, 2)  # (B, h, N, d)
        v = v.view(B, N, self.heads, D // self.heads).transpose(1, 2)  # (B, h, N, d)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, h, 1, N)
        attn = attn.softmax(dim=-1)

        out = attn @ v  # (B, h, 1, d)
        out = out.transpose(1, 2).reshape(B, 1, D)  # (B, 1, D)
        out = self.to_out(out).squeeze(1)  # (B, D)

        if return_attn:
            return out, attn  # attn: (B, h, 1, N)
        return out


class ResNet_CrossAttention(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1,
                 channel_size=[64,64,128,256,512],
                 dropout=True, in_chans=1, option=1):
        super().__init__()
        c = channel_size
        self.inplanes = c[0]
        self.option = option

        # Encoder
        self.conv1 = nn.Conv3d(in_chans, c[0], kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(c[0])
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, c[1], layers[0])
        self.layer2 = self._make_layer(block, c[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, c[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, c[4], layers[3], stride=2)

        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()

        # Attention
        self.cross_attn = CrossAttentionBlock(dim_q=option, dim_kv=c[4])  # WM+ICV as query

        self.classifier = nn.Linear(c[4], num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, info, return_attn=False):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        B, C, D, H, W = x.shape
        x_flat = x.view(B, C, -1).transpose(1, 2)  # (B, N, C)

        if self.option == 2:
            wm, icv = info
            wm = wm.float()
            icv = icv.float()
            info_vec = torch.stack([wm, icv], dim=1).unsqueeze(1).float()
        else:
            info_vec = info.unsqueeze(1).float()

        if return_attn:
            attn_feat, attn_weights = self.cross_attn(info_vec, x_flat, return_attn=True)
        else:
            attn_feat = self.cross_attn(info_vec, x_flat)

        out = self.dropout(attn_feat)
        out = self.classifier(out)

        if return_attn:
            return out, attn_weights, x.shape  # shape to reshape attention later
        return out


class ResNet_CrossAttention2(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1,
                 channel_size=[64,64,128,256,512],
                 dropout=True, in_chans=1, option=1):
        super().__init__()
        c = channel_size
        self.inplanes = c[0]
        self.option = option

        # Encoder
        self.conv1 = nn.Conv3d(in_chans, c[0], kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(c[0])
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, c[1], layers[0])
        self.layer2 = self._make_layer(block, c[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, c[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, c[4], layers[3], stride=2)

        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()

        # Attention
        self.cross_attn = CrossAttentionBlock(dim_q=1, dim_kv=c[4])  # WM+ICV as query

        self.classifier = nn.Sequential(
            nn.Linear(c[4] * 2, c[4]),
            nn.ReLU(),
            nn.Linear(c[4], num_classes)
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, info, return_attn=False):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        B, C, D, H, W = x.shape
        x_flat = x.view(B, C, -1).transpose(1, 2)  # (B, N, C)

        wm, icv = info

        wm_query = wm.unsqueeze(1).float()  # (B, 1, 1)
        icv_query = icv.unsqueeze(1).float()  # (B, 1, 1)

        if return_attn:
            wm_attn_feat, wm_attn_weights = self.cross_attn(wm_query, x_flat, return_attn=True)
            icv_attn_feat, icv_attn_weights = self.cross_attn(icv_query, x_flat, return_attn=True)
        else:
            wm_attn_feat = self.cross_attn(wm_query, x_flat)
            icv_attn_feat = self.cross_attn(icv_query, x_flat)

        fused = torch.cat([wm_attn_feat, icv_attn_feat], dim=1)

        out = self.dropout(fused)
        out = self.classifier(out)

        if return_attn:
            return out, (wm_attn_weights, icv_attn_weights), x.shape  # shape to reshape attention later
        return out


if __name__ == '__main__':
    
    mod = ResNet_add_FiLM2()

    print(mod)
