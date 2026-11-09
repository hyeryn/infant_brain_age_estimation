import torch
import torch.nn as nn
import torch.nn.functional as F

class SFCN(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=37, dropout=True, in_chans=None):
        super(SFCN, self).__init__()
        self.in_channel = in_chans
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()

        for i in range(n_layer):
            if i == 0:
                in_ch = self.in_channel
            else:
                in_ch = channel_number[i - 1]
            out_ch = channel_number[i]
            if i < n_layer - 1:
                self.feature_extractor.add_module(f'conv_{i}',
                                                  self.conv_layer(in_ch,
                                                                  out_ch,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.feature_extractor.add_module(f'conv_{i}',
                                                  self.conv_layer(in_ch,
                                                                  out_ch,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))

        self.classifier = nn.Sequential()
        
        avg_shape = [2,2,2] # 15mm

        self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))

        if dropout:
            self.classifier.add_module('dropout', nn.Dropout(0.5))

        i = n_layer
        in_channel = channel_number[-1]
        out_channel = output_dim
        self.classifier.add_module('conv_%d' % i,
                                   nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x):

        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)
        x = x.squeeze()
        x = F.log_softmax(x, dim=1)

        return x


class FiLMBlock(nn.Module):
    def __init__(self, feature_dim, condition_dim):
        super(FiLMBlock, self).__init__()
        self.gamma_fc = nn.Linear(condition_dim, feature_dim)
        self.beta_fc = nn.Linear(condition_dim, feature_dim)

    def forward(self, x, cond):
        gamma = self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta 


class SFCN_film(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=37, dropout=True, in_chans=1):
        super(SFCN_film, self).__init__()
        self.in_channel = in_chans
        self.n_layer = len(channel_number)

        self.feature_extractor = nn.ModuleList()

        for i in range(self.n_layer):
            if i == 0:
                in_ch = self.in_channel
            else:
                in_ch = channel_number[i - 1]
            out_ch = channel_number[i]

            if i == 0:
                self.feature_extractor.append(
                    self.conv_layer(in_ch, out_ch, maxpool=True, kernel_size=3, padding=1)
                )
            elif i == self.n_layer - 1:
                self.feature_extractor.append(
                    self.conv_layer(in_ch, out_ch, maxpool=False, kernel_size=1, padding=0)
                )
            else:
                self.feature_extractor.append(
                    nn.ModuleList([
                        self.conv_layer(in_ch, out_ch, maxpool=True, kernel_size=1, padding=0),
                        FiLMBlock(out_ch, 2)
                    ])
                )

        self.classifier = nn.Sequential()
        self.classifier.add_module('average_pool', nn.AvgPool3d(kernel_size=[2, 2, 2]))  # e.g., 15mm downsample

        if dropout:
            self.classifier.add_module('dropout', nn.Dropout(0.5))

        self.classifier.add_module('final_conv',
                                   nn.Conv3d(channel_number[-1], output_dim, kernel_size=1, padding=0))

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool:
            return nn.Sequential(
                nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=2, stride=maxpool_stride)
            )
        else:
            return nn.Sequential(
                nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )

    def forward(self, x, info):
        wm, icv = info  # assume: each is [B]
        wm = wm.reshape(-1, 1).float()
        icv = icv.reshape(-1, 1).float()
        cond = torch.cat((wm, icv), dim=1)  # [B, 2]

        for i, layer in enumerate(self.feature_extractor):
            if isinstance(layer, nn.ModuleList):
                conv = layer[0]
                film = layer[1]
                x = conv(x)
                x = film(x, cond)
            else:
                x = layer(x)

        x = self.classifier(x)
        x = x.squeeze()  # shape: [B, output_dim]
        x = F.log_softmax(x, dim=1)
        return x
