import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class DLAFeatureTree(nn.Module):
    def __init__(self, levels, in_channels, out_channels):
        super().__init__()
        self.levels = levels

        if levels == 1:
            self.block = BasicBlock3D(in_channels, out_channels)
        else:
            self.left_tree = DLAFeatureTree(levels - 1, in_channels, out_channels)
            self.right_tree = DLAFeatureTree(levels - 1, out_channels, out_channels)
            self.fuse = BasicBlock3D(out_channels * 2, out_channels)

    def forward(self, x):
        if self.levels == 1:
            return self.block(x)
        else:
            left = self.left_tree(x)
            right = self.right_tree(left)
            fused = torch.cat([left, right], dim=1)
            return self.fuse(fused)

class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)

    def forward(self, query, key, value):
        B, C, D, H, W = query.shape
        q = query.view(B, C, -1).transpose(1, 2)  # (B, N, C)
        k = key.view(B, C, -1).transpose(1, 2)
        v = value.view(B, C, -1).transpose(1, 2)

        attn_out, _ = self.attn(q, k, v)  # (B, N, C)
        attn_out = attn_out.transpose(1, 2).view(B, C, D, H, W)
        return attn_out

class Agg_sample(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64, depth=3):
        super().__init__()
        self.t1_encoder = DLAFeatureTree(depth, in_channels, embed_dim)
        self.t2_encoder = DLAFeatureTree(depth, in_channels, embed_dim)
        self.ratio_encoder = DLAFeatureTree(depth, in_channels, embed_dim)

        self.attn_t1 = CrossModalAttention(embed_dim)
        self.attn_t2 = CrossModalAttention(embed_dim)

        self.fusion = nn.Sequential(
            nn.Conv3d(embed_dim * 3, embed_dim, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        t1 = x[:, 0:1]
        t2 = x[:, 1:2]
        ratio = x[:, 2:3]
        t1_feat = self.t1_encoder(t1)
        t2_feat = self.t2_encoder(t2)
        ratio_feat = self.ratio_encoder(ratio)

        print(t1.shape, t1_feat.shape)

        t1_attn = self.attn_t1(ratio_feat, t1_feat, t1_feat)
        t2_attn = self.attn_t2(ratio_feat, t2_feat, t2_feat)

        fused = torch.cat([t1_attn, t2_attn, ratio_feat], dim=1)
        pooled = self.fusion(fused)
        out = self.regressor(pooled)
        return out


#         # Encode inputs

#         # Aggregate multiscale features

#         # Project to same dim for attention

#         # Apply attention

#         # Final concat & regression
