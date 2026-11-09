import numpy as np
import torch
from torch import nn


class CNNTransformer(nn.Module):
    """
    https://github.com/i-pan/kaggle-rsna-pe/blob/main/src/beehive/models/sequence.py

    https://discuss.pytorch.org/t/nn-transformerencoder-for-classification/83021/3

    DistilBertModel: ep100_lr1e-5   (for DistilBertModel 50ep not enough)

    Pytorch Transformer working better (ep50_lr5e-6)

    Good for ICBbig gender: ep50, lr1e-4, tf_nr_layers 1, tf_nr_heads 4, tf_embed_multiplier 1.0
    """

    def __init__(self, crop_size, pretrained=False, num_classes=2, in_chans=1, dropout=0.4, hparams=None):
        super(CNNTransformer, self).__init__()
        import timm

        self.hparams = hparams

        # Rough heuristic: 
        # s should be slightly bigger than avg_pool_size
        avg_pool_size = [4, 4]


        # # final_nr_filt = 128

        self.cnn = timm.create_model("tf_efficientnet_b0_ns", pretrained=pretrained,
                                     num_classes=0, in_chans=in_chans,
                                     drop_rate=dropout, global_pool='avg')
        avg_pool_size = [1, 1]
        final_nr_filt = 1280


        embedding_dim = final_nr_filt*np.prod(avg_pool_size)  # 
        self.n_layers = self.hparams.tf_nr_layers  # 2
        self.n_heads = self.hparams.tf_nr_heads  # 4


        n_hidden = int(embedding_dim * self.hparams.tf_embed_multiplier)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=self.n_heads,
                                                   dim_feedforward=n_hidden, dropout=dropout, activation="relu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        self.classifier = nn.Linear(embedding_dim, num_classes)


    def forward(self, x):
        bs, C, X, Y, Z = x.size()  # input 3D: [bs, channel, x, y, z]
        x = x.permute(0, 4, 1, 2, 3)
        x = x.reshape(bs*Z, C, X, Y)  # transform to series of 2D: [bs*z, channel, x, y]
        x = self.cnn(x)  # [bs*z, embedding]
        x = x.view(bs, Z, -1)  # change to series: [bs, z, embedding]


        x = x.permute(1, 0, 2)  # [bs, z, embedding]  # move seq dim to front
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        x = x.mean(dim=1)  # mean over seqeuence; maybe slightly better?

        x = self.classifier(x)  # which element to take from sequence?? First and last works equally bad
        
        return x  # output: [bs, classes]
