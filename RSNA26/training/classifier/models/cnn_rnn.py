import numpy as np
import torch
from torch import nn


class CNNRNN(nn.Module):
    """
    Maybe further ideas:
    https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117210
    https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/194145

    For this to train properly needs >= 100ep
    Other lr does not help.
    LSTM and RNN not working, but GRU
    -> even with manual init not working

    Most GPU RAM is required by the CNN (because this is stored #nr_layers times)
    """

    def __init__(self, crop_size, pretrained=False, num_classes=2, in_chans=1, dropout=0.4, hparams=None):
        super(CNNRNN, self).__init__()
        import timm

        self.hparams = hparams


        # Rough heuristic: 
        # s should be slightly bigger than avg_pool_size
        avg_pool_size = [4, 4]


        # size of features: 
        self.cnn = timm.create_model("tf_efficientnet_b0_ns", pretrained=pretrained,
                                     num_classes=0, in_chans=in_chans,
                                     drop_rate=dropout, global_pool='avg')
        avg_pool_size = [1, 1]
        final_nr_filt = 1280


        final_nr_filt_fc = final_nr_filt
        bidirectional = False
        if bidirectional:
            final_nr_filt_fc *= 2

        seq_model = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[self.hparams.tf_seq_model]
        self.lstm = seq_model(input_size=final_nr_filt*np.prod(avg_pool_size),
                              hidden_size=int(final_nr_filt * self.hparams.tf_embed_multiplier),   # hidden to be nr_filt/2 best for ICBbig gender prediction
                              num_layers=self.hparams.tf_nr_layers,
                              batch_first=True, 
                              bidirectional=bidirectional)   # bidirectional: lstm output shape: [bs, z, nr_hidden*2]
        self.classifier = nn.Linear(int(final_nr_filt * self.hparams.tf_embed_multiplier), num_classes)


        # Without LSTM


    def forward(self, x):
        bs, C, X, Y, Z = x.size()  # input 3D: [bs, channel, x, y, z]
        x = x.permute(0, 4, 1, 2, 3)
        x = x.reshape(bs*Z, C, X, Y)  # transform to series of 2D: [bs*z, channel, x, y]
        x = self.cnn(x)  # [bs*z, embedding]

        x = x.view(bs, Z, -1)  # change to series: [bs, z, embedding]
        
        # initialize the hidden state (maybe slightly better)
        
        x, h_n = self.lstm(x)  # for GRU
        x = self.classifier(x[:, :, :]).mean(dim=1)  # for ICBbig gender prediction mean works slightly better

        # Without any recurrent layer
        
        return x  # output: [bs, classes]
