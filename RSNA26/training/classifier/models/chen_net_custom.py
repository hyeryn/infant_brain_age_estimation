import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import os


class ChenNet(nn.Module):
    """
    Network from here: https://pubs.rsna.org/doi/10.1148/radiol.211860

    init: He
    epochs: 50
    bs: 16
    optimizer: adam
    dropout: 0.5
    lr: 0.001 at start then decaying

        stopping = EarlyStopping(patience=30)
        reduce_lr = ReduceLROnPlateau(
            factor=0.1,
            patience=8,
            min_lr=configs['learning_rate'] * 0.001)

    input image size: 86, 110, 78   (roughly downsampled by 2x from original size)

    (my images after resampling to 1.5mm isotropic: 86, 94, 84)

    Data was put into 10 bins by age. Then during training each bin was sampled equally often (inverse
    frequency).

    Data augmentation: 
    - radom rotation by up to 10 degree
    - translation up to 3 voxels
    - flipping along sagittal plane
    """
    def __init__(self, crop_size, num_classes=2, in_chans=1, dropout=0.5, dim=2, nr_filt=16):


        if dim == 2:
            Conv = nn.Conv2d
            BatchNorm = nn.BatchNorm2d
            MaxPool = nn.MaxPool2d
            AdaptiveAvgPool = nn.AdaptiveAvgPool2d
            avg_pool_size = [2, 2]
        else:
            Conv = nn.Conv3d
            BatchNorm = nn.BatchNorm3d
            MaxPool = nn.MaxPool3d
            AdaptiveAvgPool = nn.AdaptiveAvgPool3d
            avg_pool_size = [2, 2, 2]

        nn.Module.__init__(self)
        self.Conv_1 = Conv(in_chans, nr_filt, 3, 1, "same")
        self.Conv_1_bn = BatchNorm(nr_filt)
        self.Conv_1_mp = MaxPool(2)
        self.Conv_2 = Conv(nr_filt, nr_filt*2, 3, 1, "same")
        self.Conv_2_bn = BatchNorm(nr_filt*2)
        self.Conv_2_mp = MaxPool(2)
        self.Conv_3 = Conv(nr_filt*2, nr_filt*4, 3, 1, "same")
        self.Conv_3_bn = BatchNorm(nr_filt*4)
        self.Conv_3_mp = MaxPool(2)
        self.Conv_4 = Conv(nr_filt*4, nr_filt*8, 3, 1, "same")
        self.Conv_4_bn = BatchNorm(nr_filt*8)
        self.Conv_4_mp = MaxPool(2)
        self.Conv_5 = Conv(nr_filt*8, nr_filt*16, 3, 1, "same")
        self.Conv_5_bn = BatchNorm(nr_filt*16)
        self.Conv_5_mp = MaxPool(2)

        self.Conv_6 = Conv(nr_filt*16, nr_filt*4, 1, 1, "same")
        self.Conv_6_bn = BatchNorm(nr_filt*4)
        self.Conv_6_ap = AdaptiveAvgPool(avg_pool_size)  # 6

        self.dropout = nn.Dropout(dropout)
        self.dense_1 = nn.Linear(nr_filt*4*np.prod(avg_pool_size), 512)
        self.dense_2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
        x = self.Conv_1_mp(x)
        x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
        x = self.Conv_2_mp(x)
        x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
        x = self.Conv_3_mp(x)
        x = self.relu(self.Conv_4_bn(self.Conv_4(x)))
        x = self.Conv_4_mp(x)
        x = self.relu(self.Conv_5_bn(self.Conv_5(x)))
        x = self.Conv_5_mp(x)
        x = self.relu(self.Conv_6_bn(self.Conv_6(x)))
        x = self.Conv_6_ap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.dense_1(x)
        # Note that no sigmoid is applied here, because the network is used in combination with BCEWithLogitsLoss,
        # which applies sigmoid and BCELoss at the same time to make it numerically stable.
        
        return x


class ChenNet2(nn.Module):
    """
    Network from here: https://pubs.rsna.org/doi/10.1148/radiol.211860

    init: He
    epochs: 50
    bs: 16
    optimizer: adam
    dropout: 0.5
    lr: 0.001 at start then decaying

        stopping = EarlyStopping(patience=30)
        reduce_lr = ReduceLROnPlateau(
            factor=0.1,
            patience=8,
            min_lr=configs['learning_rate'] * 0.001)

    input image size: 86, 110, 78   (roughly downsampled by 2x from original size)

    (my images after resampling to 1.5mm isotropic: 86, 94, 84)

    Data was put into 10 bins by age. Then during training each bin was sampled equally often (inverse
    frequency).

    Data augmentation: 
    - radom rotation by up to 10 degree
    - translation up to 3 voxels
    - flipping along sagittal plane
    """
    def __init__(self, crop_size, num_classes=2, in_chans=1, dropout=0.5, dim=2, nr_filt=16):


        if dim == 2:
            Conv = nn.Conv2d
            BatchNorm = nn.BatchNorm2d
            MaxPool = nn.MaxPool2d
            AdaptiveAvgPool = nn.AdaptiveAvgPool2d
            avg_pool_size = [2, 2]
        else:
            Conv = nn.Conv3d
            BatchNorm = nn.BatchNorm3d
            MaxPool = nn.MaxPool3d
            AdaptiveAvgPool = nn.AdaptiveAvgPool3d
            avg_pool_size = [2, 2, 2]

        nn.Module.__init__(self)
        self.Conv_1 = Conv(in_chans, nr_filt, 3, 1, "same")
        self.Conv_1_bn = BatchNorm(nr_filt)
        self.Conv_1_mp = MaxPool(2)
        self.Conv_2 = Conv(nr_filt, nr_filt*2, 3, 1, "same")
        self.Conv_2_bn = BatchNorm(nr_filt*2)
        self.Conv_2_mp = MaxPool(2)
        self.Conv_3 = Conv(nr_filt*2, nr_filt*4, 3, 1, "same")
        self.Conv_3_bn = BatchNorm(nr_filt*4)
        self.Conv_3_mp = MaxPool(2)
        self.Conv_4 = Conv(nr_filt*4, nr_filt*8, 3, 1, "same")
        self.Conv_4_bn = BatchNorm(nr_filt*8)
        self.Conv_4_mp = MaxPool(2)
        self.Conv_5 = Conv(nr_filt*8, nr_filt*16, 3, 1, "same")
        self.Conv_5_bn = BatchNorm(nr_filt*16)
        self.Conv_5_mp = MaxPool(2)

        self.Conv_6 = Conv(nr_filt*16, nr_filt*4, 1, 1, "same")
        self.Conv_6_bn = BatchNorm(nr_filt*4)
        self.Conv_6_ap = AdaptiveAvgPool(avg_pool_size)  # 6

        self.dropout = nn.Dropout(dropout)
        self.dense_1 = nn.Linear(nr_filt*4*np.prod(avg_pool_size), 512)
        self.dense_2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
        x = self.Conv_1_mp(x)
        x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
        x = self.Conv_2_mp(x)
        x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
        x = self.Conv_3_mp(x)
        x = self.relu(self.Conv_4_bn(self.Conv_4(x)))
        x = self.Conv_4_mp(x)
        x = self.relu(self.Conv_5_bn(self.Conv_5(x)))
        x = self.Conv_5_mp(x)
        x = self.relu(self.Conv_6_bn(self.Conv_6(x)))
        x = self.Conv_6_ap(x)
        # Note that no sigmoid is applied here, because the network is used in combination with BCEWithLogitsLoss,
        # which applies sigmoid and BCELoss at the same time to make it numerically stable.
        
        return x


class ChenNet_att_cat_global(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = ChenNet(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = ChenNet(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = ChenNet(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.attn_t1 = nn.MultiheadAttention(embed_dim=args.hidden_dim, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=args.hidden_dim, num_heads=args.num_heads, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(args.hidden_dim * 3, args.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim * 2, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, 1)
        )

    def forward(self, x, save_dir=None, epoch=None):
        t1 = x[:, 0:1]
        t2 = x[:, 1:2]
        ratio = x[:, 2:3]
        # Extract features
        feat_t1 = self.encoder_t1(t1)  # Shape: (B, D)
        feat_t2 = self.encoder_t2(t2)
        feat_ratio = self.encoder_ratio(ratio)

        feat_ratio_q = feat_ratio.unsqueeze(1)  # (B, 1, D)
        feat_t1_kv = feat_t1.unsqueeze(1)
        feat_t2_kv = feat_t2.unsqueeze(1)

        attn_t1, weights_t1 = self.attn_t1(feat_ratio_q, feat_t1_kv, feat_t1_kv)
        attn_t2, weights_t2 = self.attn_t2(feat_ratio_q, feat_t2_kv, feat_t2_kv)
        attn_t1 = attn_t1.squeeze(1)  # (B, D)
        attn_t2 = attn_t2.squeeze(1)
        # [Optional] Save attention weights
        if save_dir is not None and epoch is not None:
            print('save plot')
            epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
            self.save_attention_weights(weights_t1, 't1', epoch_dir)
            self.save_attention_weights(weights_t2, 't2', epoch_dir)


        fused = torch.cat([attn_t1, attn_t2, feat_ratio], dim=1)  # (B, 3D)
        out = self.classifier(fused)
        return out
    
    def save_attention_weights(self, attn_weights, modality_name, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        attn_weights = attn_weights.squeeze(-1).squeeze(-1).detach().cpu().numpy()  # (B, heads)
        batch_size = attn_weights.shape[0]

        for i in range(batch_size):
            plt.figure(figsize=(4, 3))
            plt.title(f'{modality_name.upper()} Attention - Sample {i}')
            plt.bar(range(attn_weights.shape[1]), attn_weights[i])
            plt.xlabel('Head')
            plt.ylabel('Attention weight')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{i}_{modality_name}_attn.png'))
            plt.close()


class ChenNet_att_over_global(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = ChenNet(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = ChenNet(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = ChenNet(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.attn_t1 = nn.MultiheadAttention(embed_dim=args.hidden_dim, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=args.hidden_dim, num_heads=args.num_heads, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, 1)
        )

    def forward(self, x):
        t1 = x[:, 0:1]
        t2 = x[:, 1:2]
        ratio = x[:, 2:3]
        # Extract features
        feat_t1 = self.encoder_t1(t1)  # Shape: (B, D)
        feat_t2 = self.encoder_t2(t2)
        feat_ratio = self.encoder_ratio(ratio)

        feat_ratio_q = feat_ratio.unsqueeze(1)  # (B, 1, D)
        feat_t1_kv = feat_t1.unsqueeze(1)
        feat_t2_kv = feat_t2.unsqueeze(1)

        attn_t1, _ = self.attn_t1(feat_ratio_q, feat_t1_kv, feat_t1_kv)
        attn_t2, _ = self.attn_t2(feat_ratio_q, feat_t2_kv, feat_t2_kv)

        attn_t1 = attn_t1.squeeze(1)  # (B, D)
        attn_t2 = attn_t2.squeeze(1)

        overlay = (attn_t1 + attn_t2 + feat_ratio) / 3.0
        out = self.classifier(overlay)
        return out
    
class ChenNet_att_cat_spatial_1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.attn_t1 = nn.MultiheadAttention(embed_dim=8, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=8, num_heads=args.num_heads, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(64*3, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, 1)
        )

    def forward(self, x):
        t1 = x[:, 0:1]
        t2 = x[:, 1:2]
        ratio = x[:, 2:3]
        # Extract features
        feat_t1 = self.encoder_t1(t1)  
        feat_t2 = self.encoder_t2(t2)
        feat_ratio = self.encoder_ratio(ratio)

        b, c = feat_ratio.shape[0], feat_ratio.shape[1]
        feat_ratio_q = feat_ratio.reshape(b, c, -1)
        feat_t1_kv = feat_t1.reshape(b, c, -1)
        feat_t2_kv = feat_t2.reshape(b, c, -1)

        attn_t1, weights_t1 = self.attn_t1(feat_ratio_q, feat_t1_kv, feat_t1_kv)
        attn_t2, weights_t2 = self.attn_t2(feat_ratio_q, feat_t2_kv, feat_t2_kv)

        attn_t1 = attn_t1.transpose(1, 2)  # (B, 512, 64)
        attn_t2 = attn_t2.transpose(1, 2)  # (B, 512, 64)
        feat_ratio_seq = feat_ratio_q.transpose(1, 2)  # (B, 512, 64)
        attn_t1_pooled = attn_t1.mean(dim=1)  # (B, 64)
        attn_t2_pooled = attn_t2.mean(dim=1)  # (B, 64)
        feat_ratio_pooled = feat_ratio_seq.mean(dim=1)  # (B, 64)
        fused = torch.cat([attn_t1_pooled, attn_t2_pooled, feat_ratio_pooled], dim=1)
        out = self.classifier(fused)
        return out


### attention + Max pooling
class ChenNet_att_cat_spatial_3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.attn_t1 = nn.MultiheadAttention(embed_dim=8, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=8, num_heads=args.num_heads, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(64*3, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, 1)
        )

    def forward(self, x):
        t1 = x[:, 0:1]
        t2 = x[:, 1:2]
        ratio = x[:, 2:3]
        # Extract features
        feat_t1 = self.encoder_t1(t1)  
        feat_t2 = self.encoder_t2(t2)
        feat_ratio = self.encoder_ratio(ratio)

        b, c = feat_ratio.shape[0], feat_ratio.shape[1]
        feat_ratio_q = feat_ratio.reshape(b, c, -1)
        feat_t1_kv = feat_t1.reshape(b, c, -1)
        feat_t2_kv = feat_t2.reshape(b, c, -1)

        attn_t1, weights_t1 = self.attn_t1(feat_ratio_q, feat_t1_kv, feat_t1_kv)
        attn_t2, weights_t2 = self.attn_t2(feat_ratio_q, feat_t2_kv, feat_t2_kv)

        attn_t1 = attn_t1.transpose(1, 2)  # (B, 512, 64)
        attn_t2 = attn_t2.transpose(1, 2)  # (B, 512, 64)
        feat_ratio_seq = feat_ratio_q.transpose(1, 2)  # (B, 512, 64)
        attn_t1_pooled = attn_t1.max(dim=1)[0]  # (B, 64)
        attn_t2_pooled = attn_t2.max(dim=1)[0]  # (B, 64)
        feat_ratio_pooled = feat_ratio_seq.max(dim=1)[0]  # (B, 64)
        fused = torch.cat([attn_t1_pooled, attn_t2_pooled, feat_ratio_pooled], dim=1)
        out = self.classifier(fused)
        return out


# ### attention + CLS token


#         # Extract features


#         # Flatten spatial → sequence

#          # Add [CLS] token to query

#         # Transpose to (B, Seq_len, Embed) → (B, 513, 64) → (B, 64, 513)


class ChenNet_att_cat_spatial_2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.hidden_dim = args.hidden_dim  # e.g., 64

        # Attention layers: embed_dim should match hidden_dim
        self.attn_t1 = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=args.num_heads, batch_first=True)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, 1)
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))  # (1, 1, 64)

    def forward(self, x):
        t1 = x[:, 0:1]
        t2 = x[:, 1:2]
        ratio = x[:, 2:3]

        # Feature extraction
        feat_t1 = self.encoder_t1(t1)  # (B, C=64, D, H, W)
        feat_t2 = self.encoder_t2(t2)
        feat_ratio = self.encoder_ratio(ratio)

        b, c = feat_ratio.shape[0], feat_ratio.shape[1]  # C = hidden_dim

        # Flatten 3D feature maps to (B, Seq_len, C)
        feat_ratio_q = feat_ratio.reshape(b, c, -1).transpose(1, 2)  # (B, 512, 64)
        feat_t1_kv = feat_t1.reshape(b, c, -1).transpose(1, 2)  # (B, 512, 64)
        feat_t2_kv = feat_t2.reshape(b, c, -1).transpose(1, 2)

        # Add [CLS] token only to Q
        cls_tokens = self.cls_token.expand(b, -1, -1)  # (B, 1, 64)
        feat_ratio_q_with_cls = torch.cat([cls_tokens, feat_ratio_q], dim=1)  # (B, 513, 64)

        # Attention: Q has CLS token, KV don't
        attn_t1, weights_t1 = self.attn_t1(feat_ratio_q_with_cls, feat_t1_kv, feat_t1_kv)  # (B, 513, 64)
        attn_t2, weights_t2 = self.attn_t2(feat_ratio_q_with_cls, feat_t2_kv, feat_t2_kv)

        # Extract [CLS] token output
        attn_t1_cls = attn_t1[:, 0, :]  # (B, 64)
        attn_t2_cls = attn_t2[:, 0, :]
        ratio_cls = feat_ratio_q_with_cls[:, 0, :]  # same CLS query

        fused = torch.cat([attn_t1_cls, attn_t2_cls, ratio_cls], dim=1)  # (B, 192)
        out = self.classifier(fused)
        return out


#         # Extract features


class ChenNet_att_over_spatial_1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.attn_t1 = nn.MultiheadAttention(embed_dim=8, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=8, num_heads=args.num_heads, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(64, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, 1)
        )

    def forward(self, x):
        t1 = x[:, 0:1]
        t2 = x[:, 1:2]
        ratio = x[:, 2:3]
        # Extract features
        feat_t1 = self.encoder_t1(t1)  
        feat_t2 = self.encoder_t2(t2)
        feat_ratio = self.encoder_ratio(ratio)

        b, c = feat_ratio.shape[0], feat_ratio.shape[1]
        feat_ratio_q = feat_ratio.reshape(b, c, -1)
        feat_t1_kv = feat_t1.reshape(b, c, -1)
        feat_t2_kv = feat_t2.reshape(b, c, -1)

        attn_t1, weights_t1 = self.attn_t1(feat_ratio_q, feat_t1_kv, feat_t1_kv)
        attn_t2, weights_t2 = self.attn_t2(feat_ratio_q, feat_t2_kv, feat_t2_kv)

        attn_t1 = attn_t1.transpose(1, 2)  # (B, 512, 64)
        attn_t2 = attn_t2.transpose(1, 2)  # (B, 512, 64)
        feat_ratio_seq = feat_ratio_q.transpose(1, 2)  # (B, 512, 64)
        attn_t1_pooled = attn_t1.mean(dim=1)  # (B, 64)
        attn_t2_pooled = attn_t2.mean(dim=1)  # (B, 64)
        feat_ratio_pooled = feat_ratio_seq.mean(dim=1)  # (B, 64)

        overlay = (attn_t1_pooled + attn_t2_pooled + feat_ratio_pooled) / 3.0
        out = self.classifier(overlay)

        return out


### attention + Max pooling
class ChenNet_att_over_spatial_3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.attn_t1 = nn.MultiheadAttention(embed_dim=8, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=8, num_heads=args.num_heads, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(64, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, 1)
        )

    def forward(self, x):
        t1 = x[:, 0:1]
        t2 = x[:, 1:2]
        ratio = x[:, 2:3]
        # Extract features
        feat_t1 = self.encoder_t1(t1)  
        feat_t2 = self.encoder_t2(t2)
        feat_ratio = self.encoder_ratio(ratio)

        b, c = feat_ratio.shape[0], feat_ratio.shape[1]
        feat_ratio_q = feat_ratio.reshape(b, c, -1)
        feat_t1_kv = feat_t1.reshape(b, c, -1)
        feat_t2_kv = feat_t2.reshape(b, c, -1)

        attn_t1, weights_t1 = self.attn_t1(feat_ratio_q, feat_t1_kv, feat_t1_kv)
        attn_t2, weights_t2 = self.attn_t2(feat_ratio_q, feat_t2_kv, feat_t2_kv)

        attn_t1 = attn_t1.transpose(1, 2)  # (B, 512, 64)
        attn_t2 = attn_t2.transpose(1, 2)  # (B, 512, 64)
        feat_ratio_seq = feat_ratio_q.transpose(1, 2)  # (B, 512, 64)
        attn_t1_pooled = attn_t1.max(dim=1)[0]  # (B, 64)
        attn_t2_pooled = attn_t2.max(dim=1)[0]  # (B, 64)
        feat_ratio_pooled = feat_ratio_seq.max(dim=1)[0]  # (B, 64)

        overlay = (attn_t1_pooled + attn_t2_pooled + feat_ratio_pooled) / 3.0
        out = self.classifier(overlay)

        return out
    
class ChenNet_att_over_spatial_2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.hidden_dim = args.hidden_dim  # e.g., 64

        # Attention layers: embed_dim should match hidden_dim
        self.attn_t1 = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=args.num_heads, batch_first=True)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, 1)
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))  # (1, 1, 64)

    def forward(self, x):
        t1 = x[:, 0:1]
        t2 = x[:, 1:2]
        ratio = x[:, 2:3]

        # Feature extraction
        feat_t1 = self.encoder_t1(t1)  # (B, C=64, D, H, W)
        feat_t2 = self.encoder_t2(t2)
        feat_ratio = self.encoder_ratio(ratio)

        b, c = feat_ratio.shape[0], feat_ratio.shape[1]  # C = hidden_dim

        # Flatten 3D feature maps to (B, Seq_len, C)
        feat_ratio_q = feat_ratio.reshape(b, c, -1).transpose(1, 2)  # (B, 512, 64)
        feat_t1_kv = feat_t1.reshape(b, c, -1).transpose(1, 2)  # (B, 512, 64)
        feat_t2_kv = feat_t2.reshape(b, c, -1).transpose(1, 2)

        # Add [CLS] token only to Q
        cls_tokens = self.cls_token.expand(b, -1, -1)  # (B, 1, 64)
        feat_ratio_q_with_cls = torch.cat([cls_tokens, feat_ratio_q], dim=1)  # (B, 513, 64)

        # Attention: Q has CLS token, KV don't
        attn_t1, weights_t1 = self.attn_t1(feat_ratio_q_with_cls, feat_t1_kv, feat_t1_kv)  # (B, 513, 64)
        attn_t2, weights_t2 = self.attn_t2(feat_ratio_q_with_cls, feat_t2_kv, feat_t2_kv)

        # Extract [CLS] token output
        attn_t1_cls = attn_t1[:, 0, :]  # (B, 64)
        attn_t2_cls = attn_t2[:, 0, :]
        ratio_cls = feat_ratio_q_with_cls[:, 0, :]  # same CLS query


        overlay = (attn_t1_cls + attn_t2_cls + ratio_cls) / 3.0
        out = self.classifier(overlay)

        return out


class ChenNet_cat_global(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = ChenNet(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = ChenNet(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = ChenNet(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.classifier = nn.Sequential(
            nn.Linear(args.hidden_dim * 3, args.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim * 2, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, 1)
        )

    def forward(self, x, save_dir=None, epoch=None):
        t1 = x[:, 0:1]
        t2 = x[:, 1:2]
        ratio = x[:, 2:3]
        # Extract features
        feat_t1 = self.encoder_t1(t1)  # Shape: (B, D)
        feat_t2 = self.encoder_t2(t2)
        feat_ratio = self.encoder_ratio(ratio)


        fused = torch.cat([feat_t1, feat_t2, feat_ratio], dim=1)  # (B, 3D)
        out = self.classifier(fused)
        return out


class ChenNet_over_global(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = ChenNet(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = ChenNet(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = ChenNet(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.classifier = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, 1)
        )

    def forward(self, x):
        t1 = x[:, 0:1]
        t2 = x[:, 1:2]
        ratio = x[:, 2:3]
        # Extract features
        feat_t1 = self.encoder_t1(t1)  # Shape: (B, D)
        feat_t2 = self.encoder_t2(t2)
        feat_ratio = self.encoder_ratio(ratio)

        overlay = (feat_t1 + feat_t2 + feat_ratio) / 3.0
        out = self.classifier(overlay)
        return out
    
class ChenNet_att_cat_spatial_11(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.attn_t1 = nn.MultiheadAttention(embed_dim=64, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=64, num_heads=args.num_heads, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(64*3, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, 1)
        )

    def forward(self, x):
        t1 = x[:, 0:1]
        t2 = x[:, 1:2]
        ratio = x[:, 2:3]
        # Extract features
        feat_t1 = self.encoder_t1(t1)  
        feat_t2 = self.encoder_t2(t2)
        feat_ratio = self.encoder_ratio(ratio)

        b, c = feat_ratio.shape[0], feat_ratio.shape[1]
        feat_ratio_q = feat_ratio.reshape(b, c, -1).permute(0, 2, 1)
        feat_t1_kv = feat_t1.reshape(b, c, -1).permute(0, 2, 1)
        feat_t2_kv = feat_t2.reshape(b, c, -1).permute(0, 2, 1)

        attn_t1, weights_t1 = self.attn_t1(feat_ratio_q, feat_t1_kv, feat_t1_kv)
        attn_t2, weights_t2 = self.attn_t2(feat_ratio_q, feat_t2_kv, feat_t2_kv)

        attn_t1 = attn_t1  # (B, 512, 64)
        attn_t2 = attn_t2  # (B, 512, 64)
        feat_ratio_seq = feat_ratio_q # (B, 512, 64)
        attn_t1_pooled = attn_t1.mean(dim=1)  # (B, 64)
        attn_t2_pooled = attn_t2.mean(dim=1)  # (B, 64)
        feat_ratio_pooled = feat_ratio_seq.mean(dim=1)  # (B, 64)
        fused = torch.cat([attn_t1_pooled, attn_t2_pooled, feat_ratio_pooled], dim=1)
        out = self.classifier(fused)
        return out


### attention + Max pooling
class ChenNet_att_cat_spatial_33(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.attn_t1 = nn.MultiheadAttention(embed_dim=64, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=64, num_heads=args.num_heads, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(64*3, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, 1)
        )

    def forward(self, x):
        t1 = x[:, 0:1]
        t2 = x[:, 1:2]
        ratio = x[:, 2:3]
        # Extract features
        feat_t1 = self.encoder_t1(t1)  
        feat_t2 = self.encoder_t2(t2)
        feat_ratio = self.encoder_ratio(ratio)

        b, c = feat_ratio.shape[0], feat_ratio.shape[1]
        feat_ratio_q = feat_ratio.reshape(b, c, -1).permute(0, 2, 1)
        feat_t1_kv = feat_t1.reshape(b, c, -1).permute(0, 2, 1)
        feat_t2_kv = feat_t2.reshape(b, c, -1).permute(0, 2, 1)

        attn_t1, weights_t1 = self.attn_t1(feat_ratio_q, feat_t1_kv, feat_t1_kv)
        attn_t2, weights_t2 = self.attn_t2(feat_ratio_q, feat_t2_kv, feat_t2_kv)

        attn_t1 = attn_t1
        attn_t2 = attn_t2
        feat_ratio_seq = feat_ratio_q
        attn_t1_pooled = attn_t1.max(dim=1)[0]  # (B, 64)
        attn_t2_pooled = attn_t2.max(dim=1)[0]  # (B, 64)
        feat_ratio_pooled = feat_ratio_seq.max(dim=1)[0]  # (B, 64)
        fused = torch.cat([attn_t1_pooled, attn_t2_pooled, feat_ratio_pooled], dim=1)
        out = self.classifier(fused)
        return out


class ChenNet_att_over_spatial_11(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.attn_t1 = nn.MultiheadAttention(embed_dim=64, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=64, num_heads=args.num_heads, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(64, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, 1)
        )

    def forward(self, x):
        t1 = x[:, 0:1]
        t2 = x[:, 1:2]
        ratio = x[:, 2:3]
        # Extract features
        feat_t1 = self.encoder_t1(t1)  
        feat_t2 = self.encoder_t2(t2)
        feat_ratio = self.encoder_ratio(ratio)

        b, c = feat_ratio.shape[0], feat_ratio.shape[1]
        feat_ratio_q = feat_ratio.reshape(b, c, -1).permute(0, 2, 1)
        feat_t1_kv = feat_t1.reshape(b, c, -1).permute(0, 2, 1)
        feat_t2_kv = feat_t2.reshape(b, c, -1).permute(0, 2, 1)

        attn_t1, weights_t1 = self.attn_t1(feat_ratio_q, feat_t1_kv, feat_t1_kv)
        attn_t2, weights_t2 = self.attn_t2(feat_ratio_q, feat_t2_kv, feat_t2_kv)

        attn_t1 = attn_t1
        attn_t2 = attn_t2
        feat_ratio_seq = feat_ratio_q
        attn_t1_pooled = attn_t1.mean(dim=1)  # (B, 64)
        attn_t2_pooled = attn_t2.mean(dim=1)  # (B, 64)
        feat_ratio_pooled = feat_ratio_seq.mean(dim=1)  # (B, 64)

        overlay = (attn_t1_pooled + attn_t2_pooled + feat_ratio_pooled) / 3.0
        out = self.classifier(overlay)

        return out


### attention + Max pooling
class ChenNet_att_over_spatial_33(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.attn_t1 = nn.MultiheadAttention(embed_dim=64, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=64, num_heads=args.num_heads, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(64, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, 1)
        )

    def forward(self, x):
        t1 = x[:, 0:1]
        t2 = x[:, 1:2]
        ratio = x[:, 2:3]
        # Extract features
        feat_t1 = self.encoder_t1(t1)  
        feat_t2 = self.encoder_t2(t2)
        feat_ratio = self.encoder_ratio(ratio)

        b, c = feat_ratio.shape[0], feat_ratio.shape[1]
        feat_ratio_q = feat_ratio.reshape(b, c, -1).permute(0, 2, 1)
        feat_t1_kv = feat_t1.reshape(b, c, -1).permute(0, 2, 1)
        feat_t2_kv = feat_t2.reshape(b, c, -1).permute(0, 2, 1)

        attn_t1, weights_t1 = self.attn_t1(feat_ratio_q, feat_t1_kv, feat_t1_kv)
        attn_t2, weights_t2 = self.attn_t2(feat_ratio_q, feat_t2_kv, feat_t2_kv)

        attn_t1 = attn_t1
        attn_t2 = attn_t2
        feat_ratio_seq = feat_ratio_q
        attn_t1_pooled = attn_t1.max(dim=1)[0]  # (B, 64)
        attn_t2_pooled = attn_t2.max(dim=1)[0]  # (B, 64)
        feat_ratio_pooled = feat_ratio_seq.max(dim=1)[0]  # (B, 64)

        overlay = (attn_t1_pooled + attn_t2_pooled + feat_ratio_pooled) / 3.0
        out = self.classifier(overlay)

        return out


class ChenNet_att_weighted_cat_spatial(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.attn_t1 = nn.MultiheadAttention(embed_dim=64, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=64, num_heads=args.num_heads, batch_first=True)

        self.score_fc = nn.Linear(64, 1)

        self.classifier = nn.Sequential(
            nn.Linear(64 * 3, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, 1)
        )

    def attention_pooling(self, x):
        attn_score = self.score_fc(x)  # (B, Seq_len, 1)
        attn_score = F.softmax(attn_score, dim=1)  # normalize over spatial positions
        pooled = torch.sum(attn_score * x, dim=1)  # weighted sum
        return pooled  # (B, C)

    def forward(self, x):
        t1 = x[:, 0:1]
        t2 = x[:, 1:2]
        ratio = x[:, 2:3]

        feat_t1 = self.encoder_t1(t1)  
        feat_t2 = self.encoder_t2(t2)
        feat_ratio = self.encoder_ratio(ratio)

        b, c = feat_ratio.shape[0], feat_ratio.shape[1]
        feat_ratio_q = feat_ratio.reshape(b, c, -1).permute(0, 2, 1)  # (B, Seq, C)
        feat_t1_kv = feat_t1.reshape(b, c, -1).permute(0, 2, 1)
        feat_t2_kv = feat_t2.reshape(b, c, -1).permute(0, 2, 1)

        attn_t1, _ = self.attn_t1(feat_ratio_q, feat_t1_kv, feat_t1_kv)  # (B, Seq, C)
        attn_t2, _ = self.attn_t2(feat_ratio_q, feat_t2_kv, feat_t2_kv)

        pooled_t1 = self.attention_pooling(attn_t1)       # (B, C)
        pooled_t2 = self.attention_pooling(attn_t2)       # (B, C)
        pooled_ratio = self.attention_pooling(feat_ratio_q)  # (B, C)

        fused = torch.cat([pooled_t1, pooled_t2, pooled_ratio], dim=1)  # (B, C*3)
        out = self.classifier(fused)
        return out


class ChenNet_att_weighted_over_spatial(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = ChenNet2(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.attn_t1 = nn.MultiheadAttention(embed_dim=64, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=64, num_heads=args.num_heads, batch_first=True)

        self.score_fc = nn.Linear(64, 1)

        self.classifier = nn.Sequential(
            nn.Linear(64, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, 1)
        )

    def attention_pooling(self, x):
        attn_score = self.score_fc(x)  # (B, Seq_len, 1)
        attn_score = F.softmax(attn_score, dim=1)  # normalize over spatial positions
        pooled = torch.sum(attn_score * x, dim=1)  # weighted sum
        return pooled  # (B, C)

    def forward(self, x):
        t1 = x[:, 0:1]
        t2 = x[:, 1:2]
        ratio = x[:, 2:3]

        feat_t1 = self.encoder_t1(t1)  
        feat_t2 = self.encoder_t2(t2)
        feat_ratio = self.encoder_ratio(ratio)

        b, c = feat_ratio.shape[0], feat_ratio.shape[1]
        feat_ratio_q = feat_ratio.reshape(b, c, -1).permute(0, 2, 1)  # (B, Seq, C)
        feat_t1_kv = feat_t1.reshape(b, c, -1).permute(0, 2, 1)
        feat_t2_kv = feat_t2.reshape(b, c, -1).permute(0, 2, 1)

        attn_t1, _ = self.attn_t1(feat_ratio_q, feat_t1_kv, feat_t1_kv)  # (B, Seq, C)
        attn_t2, _ = self.attn_t2(feat_ratio_q, feat_t2_kv, feat_t2_kv)

        pooled_t1 = self.attention_pooling(attn_t1)       # (B, C)
        pooled_t2 = self.attention_pooling(attn_t2)       # (B, C)
        pooled_ratio = self.attention_pooling(feat_ratio_q)  # (B, C)

        overlay = (pooled_t1 + pooled_t2 + pooled_ratio) / 3.0
        out = self.classifier(overlay)

        return out
