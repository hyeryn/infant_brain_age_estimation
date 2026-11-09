import sys
from pathlib import Path
p_dir = str(Path(__file__).absolute().parents[2])
if p_dir not in sys.path: sys.path.insert(0, p_dir)

import os
from os.path import join

import torch.nn as nn


class efficient_net(nn.Module):
    """
    hparams from https://www.kaggle.com/rluethy/efficientnet3d-with-one-mri-type/notebook
    Adam + lr=0.001
    criterion = torch_functional.binary_cross_entropy_with_logits
    10 epochs
    patience=10

    With self-compiled torch this is not faster. (3D convs for nnUNet are a lot faster with
    self-compiled torch)
    """
    def __init__(self, backbone, in_channels, out_channels):
        super(efficient_net, self).__init__()

        # https://github.com/shijianjian/EfficientNet-PyTorch-3D
        #
        # pip install git+https://github.com/shijianjian/EfficientNet-PyTorch-3D
        #
        from classifier.models.efficientnet_pytorch_3d import EfficientNet3D

        self.enet = EfficientNet3D.from_name(backbone, in_channels=in_channels,
                                             override_params={'num_classes': out_channels})

        n_features = self.enet._fc.in_features
        # This is from https://www.kaggle.com/rluethy/efficientnet3d-with-one-mri-type/notebook
        # but since we do not change the nr of classes we would actually not need it
        self.enet._fc = nn.Linear(in_features=n_features, out_features=out_channels, bias=True)

    def forward(self, x):
        return self.enet(x)


class efficient_net_film(nn.Module):
    def __init__(self, backbone, in_channels, out_channels):
        super(efficient_net_film, self).__init__()

        from classifier.models.efficientnet_pytorch_3d import EfficientNet3D_FiLM

        self.enet = EfficientNet3D_FiLM.from_name(backbone, in_channels=in_channels,
                                             override_params={'num_classes': out_channels})

        n_features = self.enet._fc.in_features
        self.enet._fc = nn.Linear(in_features=n_features, out_features=out_channels, bias=True)

    def forward(self, x, info):
        return self.enet(x, info)
