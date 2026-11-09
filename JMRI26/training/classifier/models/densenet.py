import torch
import torch.nn as nn
import torch.nn.functional as F


class _DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, bn_size, dropout):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm3d(in_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv3d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))

        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))

        self.dropout = nn.Dropout3d(p=dropout) if dropout else nn.Identity()

    def forward(self, x):
        new_features = super().forward(x)
        new_features = self.dropout(new_features)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate, dropout):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, dropout)
            self.add_module(f'denselayer{i + 1}', layer)


class _Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet121(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                 bn_size=4, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential()

        # Initial convolution
        self.features.add_module('conv0', nn.Conv3d(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False))
        self.features.add_module('norm0', nn.BatchNorm3d(init_features))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        self.features.add_module('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1))

        num_features = init_features

        # DenseBlocks + Transitions
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, dropout)
            self.features.add_module(f'denseblock{i + 1}', block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm3d(num_features))

        # Classifier
        self.classifier = nn.Linear(num_features, out_channels)

    def forward(self, x):
        features = self.features(x)  # [B, C, D, H, W]
        out = F.adaptive_avg_pool3d(features, (1, 1, 1)).view(x.size(0), -1)
        out = self.classifier(out)
        return out


class FiLMBlock(nn.Module):
    def __init__(self, feature_dim, condition_dim):
        super(FiLMBlock, self).__init__()
        self.gamma_fc = nn.Linear(condition_dim, feature_dim)
        self.beta_fc = nn.Linear(condition_dim, feature_dim)

    def forward(self, x, cond):
        gamma = self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta 


class DenseNet121_film(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                 bn_size=4, dropout=0.3):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv3d(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        self.film_blocks = nn.ModuleList()

        num_features = init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, dropout)
            self.blocks.append(block)
            num_features += num_layers * growth_rate

            if i != len(block_config) - 1:
                # Transition layer
                trans = _Transition(num_features, num_features // 2)
                self.transitions.append(trans)

                num_features = num_features // 2
                self.film_blocks.append(FiLMBlock(num_features, condition_dim=2))

        self.norm_final = nn.BatchNorm3d(num_features)
        self.classifier = nn.Linear(num_features, out_channels)

    def forward(self, x, info):
        wm, icv = info
        cond = torch.cat([wm.view(-1, 1).float(), icv.view(-1, 1).float()], dim=1)  # [B, 2]

        x = self.initial(x)

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            if i < len(self.transitions):  # only if transition exists
                x = self.transitions[i](x)
                x = self.film_blocks[i](x, cond)  # apply FiLM after transition (channel matched)

        x = self.norm_final(x)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x
