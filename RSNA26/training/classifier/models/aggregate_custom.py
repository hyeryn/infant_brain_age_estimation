import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import os


class Aggregate(nn.Module):
    def __init__(self, crop_size, num_classes, dim, in_chans=1, dropout=0.5, nr_filt=8):
        super().__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Block 1
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_chans, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 1))
        )
        self.gap1 = nn.AdaptiveAvgPool3d((1, 1, 1))  # Output: (B, 8, 1, 1, 1)
        self.fc1 = nn.Linear(8, 64)

        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 1))
        )
        self.gap2 = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc2 = nn.Linear(16, 64)

        # Block 3
        self.conv3 = nn.Sequential(
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 1))
        )
        self.gap3 = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc3 = nn.Linear(32, 64)


    def forward(self, x):
        # Block 1
        x1 = self.conv1(x)
        f1 = self.gap1(x1).flatten(1)  # (B, 8) → fc → (B, 64)
        f1 = self.dropout(self.relu(self.fc1(f1)))

        # Block 2
        x2 = self.conv2(x1)
        f2 = self.gap2(x2).flatten(1)
        f2 = self.dropout(self.relu(self.fc2(f2)))

        # Block 3
        x3 = self.conv3(x2)
        f3 = self.gap3(x3).flatten(1)
        f3 = self.dropout(self.relu(self.fc3(f3)))

        # Concatenate hierarchical features
        fused = torch.cat([f1, f2, f3], dim=1)  # (B, 64*3)

        return fused


class ZAwareEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()
        self.out_channels = base_channels * 4  # For attention input dimension

        def block(in_ch, out_ch, z_stride=1):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=(2,2,z_stride), padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU()
            )

        # Feature levels
        self.layer1 = block(in_channels, base_channels, z_stride=1)   # (42 z stays)
        self.layer2 = block(base_channels, base_channels*2, z_stride=1)  # (still 42 z)
        self.layer3 = block(base_channels*2, base_channels*4, z_stride=2)  # z → 21

    def forward(self, x):
        feats = []
        x = self.layer1(x); feats.append(x)
        x = self.layer2(x); feats.append(x)
        x = self.layer3(x); feats.append(x)
        return feats  # list of 3 feature levels


class Agg_att_cat_spatial_1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.attn_t1 = nn.MultiheadAttention(embed_dim=512, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=512, num_heads=args.num_heads, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(96, args.hidden_dim),
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
        print(feat_t1.shape, feat_t2.shape, feat_ratio.shape)

        b, c = feat_ratio.shape[0], feat_ratio.shape[1]
        feat_ratio_q = feat_ratio.reshape(b, c, -1)
        feat_t1_kv = feat_t1.reshape(b, c, -1)
        feat_t2_kv = feat_t2.reshape(b, c, -1)
        print(feat_ratio_q.shape, feat_t1_kv.shape, feat_t2_kv.shape)

        attn_t1, weights_t1 = self.attn_t1(feat_ratio_q, feat_t1_kv, feat_t1_kv)
        attn_t2, weights_t2 = self.attn_t2(feat_ratio_q, feat_t2_kv, feat_t2_kv)
        print(attn_t1.shape, weights_t1.shape)

        attn_t1 = attn_t1.transpose(1, 2)  # (B, 512, 64)
        attn_t2 = attn_t2.transpose(1, 2)  # (B, 512, 64)
        feat_ratio_seq = feat_ratio_q.transpose(1, 2)  # (B, 512, 64)
        print(attn_t1.shape)
        attn_t1_pooled = attn_t1.mean(dim=1)  # (B, 64)
        attn_t2_pooled = attn_t2.mean(dim=1)  # (B, 64)
        feat_ratio_pooled = feat_ratio_seq.mean(dim=1)  # (B, 64)
        print(attn_t1_pooled.shape)
        fused = torch.cat([attn_t1_pooled, attn_t2_pooled, feat_ratio_pooled], dim=1)
        print(fused.shape, attn_t1_pooled.shape)  # (B, 64*3)
        out = self.classifier(fused)
        print(out.shape)
        return out


### attention + Max pooling
class Agg_att_cat_spatial_3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.attn_t1 = nn.MultiheadAttention(embed_dim=512, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=512, num_heads=args.num_heads, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(96, args.hidden_dim),
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


class Agg_att_cat_spatial_2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.hidden_dim = 32

        # Attention layers: embed_dim should match hidden_dim
        self.attn_t1 = nn.MultiheadAttention(embed_dim=32, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=32, num_heads=args.num_heads, batch_first=True)

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


class Agg_att_over_spatial_1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.attn_t1 = nn.MultiheadAttention(embed_dim=512, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=512, num_heads=args.num_heads, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(24, 1)
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
class Agg_att_over_spatial_3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.attn_t1 = nn.MultiheadAttention(embed_dim=512, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=512, num_heads=args.num_heads, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(24, 1)
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
    
class Agg_att_over_spatial_2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.hidden_dim = 32

        # Attention layers: embed_dim should match hidden_dim
        self.attn_t1 = nn.MultiheadAttention(embed_dim=32, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=32, num_heads=args.num_heads, batch_first=True)

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


class Agg_cat_global(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.classifier = nn.Sequential(
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(96, 1)
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


class Agg_over_global(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(32, 1)
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
    
class Agg_att_cat_spatial_11(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.attn_t1 = nn.MultiheadAttention(embed_dim=32, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=32, num_heads=args.num_heads, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(32*3, 48),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(48, 1)
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
class Agg_att_cat_spatial_33(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.attn_t1 = nn.MultiheadAttention(embed_dim=32, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=32, num_heads=args.num_heads, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(32*3, 48),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(48, 1)
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


class Agg_att_over_spatial_11(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.attn_t1 = nn.MultiheadAttention(embed_dim=32, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=32, num_heads=args.num_heads, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(24, 1)
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
class Agg_att_over_spatial_33(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.attn_t1 = nn.MultiheadAttention(embed_dim=32, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=32, num_heads=args.num_heads, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(24, 1)
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


class Agg_att_weighted_cat_spatial(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.attn_t1 = nn.MultiheadAttention(embed_dim=512, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=512, num_heads=args.num_heads, batch_first=True)

        self.score_fc = nn.Linear(32, 1)

        self.classifier = nn.Sequential(
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(48, 1)
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
        feat_ratio_q = feat_ratio.reshape(b, c, -1)
        feat_t1_kv = feat_t1.reshape(b, c, -1)
        feat_t2_kv = feat_t2.reshape(b, c, -1)

        attn_t1, _ = self.attn_t1(feat_ratio_q, feat_t1_kv, feat_t1_kv)  # (B, Seq, C)
        attn_t2, _ = self.attn_t2(feat_ratio_q, feat_t2_kv, feat_t2_kv)

        pooled_t1 = self.attention_pooling(attn_t1.transpose(1, 2))       # (B, C)
        pooled_t2 = self.attention_pooling(attn_t2.transpose(1, 2))       # (B, C)
        pooled_ratio = self.attention_pooling(feat_ratio_q.transpose(1, 2))  # (B, C)

        fused = torch.cat([pooled_t1, pooled_t2, pooled_ratio], dim=1)  # (B, C*3)
        out = self.classifier(fused)
        return out


class Agg_att_weighted_over_spatial(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.attn_t1 = nn.MultiheadAttention(embed_dim=512, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=512, num_heads=args.num_heads, batch_first=True)

        self.score_fc = nn.Linear(32, 1)

        self.classifier = nn.Sequential(
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(24, 1)
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
        feat_ratio_q = feat_ratio.reshape(b, c, -1)
        feat_t1_kv = feat_t1.reshape(b, c, -1)
        feat_t2_kv = feat_t2.reshape(b, c, -1)

        attn_t1, _ = self.attn_t1(feat_ratio_q, feat_t1_kv, feat_t1_kv)  # (B, Seq, C)
        attn_t2, _ = self.attn_t2(feat_ratio_q, feat_t2_kv, feat_t2_kv)

        pooled_t1 = self.attention_pooling(attn_t1.transpose(1, 2))       # (B, C)
        pooled_t2 = self.attention_pooling(attn_t2.transpose(1, 2))       # (B, C)
        pooled_ratio = self.attention_pooling(feat_ratio_q.transpose(1, 2))  # (B, C)

        overlay = (pooled_t1 + pooled_t2 + pooled_ratio) / 3.0
        out = self.classifier(overlay)

        return out


class Agg_att_weighted_cat_spatial_(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.attn_t1 = nn.MultiheadAttention(embed_dim=32, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=32, num_heads=args.num_heads, batch_first=True)

        self.score_fc = nn.Linear(32, 1)

        self.classifier = nn.Sequential(
            nn.Linear(32 * 3, 64),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(64, 1)
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


class Agg_att_weighted_over_spatial_(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_t1 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_t2 = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)
        self.encoder_ratio = Aggregate(crop_size=args.crop_size, in_chans=1, num_classes=args.hidden_dim, dim=3, dropout=args.dropout)

        self.attn_t1 = nn.MultiheadAttention(embed_dim=32, num_heads=args.num_heads, batch_first=True)
        self.attn_t2 = nn.MultiheadAttention(embed_dim=32, num_heads=args.num_heads, batch_first=True)

        self.score_fc = nn.Linear(32, 1)

        self.classifier = nn.Sequential(
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(24, 1)
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
