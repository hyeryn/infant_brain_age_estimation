# Parameter Count Summary for 3D CNN Architectures

## Common Settings
- **Input**: 1 x 128 x 128 x 128 (single-channel 3D MRI)
- **Output**: regression (scalar, 1 neuron)
- All counts computed from actual model implementations in `classifier/models/`

---

## 1. BasicCNN — 4,267,713 (4.27M) — Moderate

| Layer | Parameters |
|---|---|
| Conv3d(1->8, k=3) + BN | 232 |
| Conv3d(8->16, k=3) + BN | 3,488 |
| Conv3d(16->32, k=3) + BN | 13,888 |
| Conv3d(32->64, k=3) + BN | 55,424 |
| AdaptiveAvgPool(8,8,8) -> FC(32768->128) | 4,194,432 |
| FC(128->1) | 129 |
| **Total** | **4,267,713** |

- Channels: 8 -> 16 -> 32 -> 64
- FC layer dominates (98%) due to AdaptiveAvgPool output size 8^3=512 x 64 = 32,768

---

## 2. CBR-tiny — 77,249 (77.2K) — Very lightweight

| Layer | Parameters |
|---|---|
| Conv3d(1->8, k=3) + BN | 232 |
| Conv3d(8->16, k=3) + BN | 3,488 |
| Conv3d(16->32, k=3) + BN | 13,888 |
| Conv3d(32->64, k=3) + BN | 55,424 |
| AdaptiveAvgPool(4,4,4) -> FC(4096->1) | 4,097 |
| **Total** | **77,249** |

- Same conv channels as BasicCNN but uses smaller pool (4^3) and single FC layer

---

## 3. ResNet-18 (3D) — 33,160,513 (33.16M) — Large

| Module | Parameters |
|---|---|
| conv1 (7x7x7, s=2, 1->64) | 21,952 |
| bn1 | 128 |
| layer1 (2 BasicBlocks, 64->64) | 442,880 |
| layer2 (2 BasicBlocks, 64->128) | 1,557,760 |
| layer3 (2 BasicBlocks, 128->256) | 6,228,480 |
| layer4 (2 BasicBlocks, 256->512) | 24,908,800 |
| FC(512->1) | 513 |
| **Total** | **33,160,513** |

- 3D convolutions are 27x larger than 2D (k^3 vs k^2)

---

## 4. ResNet-34 (3D) — 63,470,145 (63.47M) — Large

| Module | Parameters |
|---|---|
| conv1 + bn1 | 22,080 |
| layer1 (3 blocks) | 664,320 |
| layer2 (4 blocks) | 3,328,256 |
| layer3 (6 blocks) | 20,388,352 |
| layer4 (3 blocks) | 39,066,624 |
| FC(512->1) | 513 |
| **Total** | **63,470,145** |

---

## 5. ResNet-50 (3D, Bottleneck) — 46,157,121 (46.16M) — Large

| Module | Parameters |
|---|---|
| conv1 + bn1 | 22,080 |
| layer1 (3 Bottleneck, 64->256) | 436,992 |
| layer2 (4 Bottleneck, 128->512) | 2,399,232 |
| layer3 (6 Bottleneck, 256->1024) | 14,176,256 |
| layer4 (3 Bottleneck, 512->2048) | 29,120,512 |
| FC(2048->1) | 2,049 |
| **Total** | **46,157,121** |

- Bottleneck (1x1->3x3->1x1) is more efficient than BasicBlock in 3D: ResNet-50 < ResNet-34

---

## 6. DenseNet-121 (3D) — 11,243,649 (11.24M) — Moderate

| Module | Parameters |
|---|---|
| features (stem + 4 dense blocks + 3 transitions + final BN) | 11,242,624 |
| classifier FC(1024->1) | 1,025 |
| **Total** | **11,243,649** |

- Config: init_features=64, growth_rate=32, block_config=(6,12,24,16), bn_size=4

---

## 7. EfficientNet-B0 (3D) — 4,689,661 (4.69M) — Moderate

| Module | Parameters |
|---|---|
| Stem (Conv3d 1->32, k=3, s=2 + BN) | 928 |
| MBConv blocks (16 blocks with SE) | 4,275,292 |
| Head (Conv3d->1280 + BN) | 412,160 |
| FC(1280->1) | 1,281 |
| **Total** | **4,689,661** |

- Uses depthwise separable convolutions + SE blocks

---

## 8. U-Net Encoder — 14,198,753 (14.20M) — Moderate

| Module | Parameters |
|---|---|
| enc1: 2xConv(1->32->32) | 28,256 |
| enc2: 2xConv(32->64->64) | 332,160 |
| enc3: 2xConv(64->128->128) | 1,327,360 |
| enc4: 2xConv(128->256->256) | 5,307,904 |
| enc5: 2xConv(256->512->512) | 7,137,280 |
| FC(512->128->1) | 65,793 |
| **Total** | **14,198,753** |

- Uses encoder-only with 2-layer FC head: FC(512->128) + FC(128->1)

---

## 9. SFCN — 2,950,401 (2.95M) — Moderate

| Layer | Parameters |
|---|---|
| Conv3d(1->32, k=3)+BN+MaxPool | 928 |
| Conv3d(32->64, k=3)+BN+MaxPool | 55,424 |
| Conv3d(64->128, k=3)+BN+MaxPool | 221,568 |
| Conv3d(128->256, k=3)+BN+MaxPool | 885,248 |
| Conv3d(256->256, k=3)+BN+MaxPool | 1,770,240 |
| Conv3d(256->64, k=1)+BN (nonlinear) | 16,576 |
| AvgPool(2,2,2) + Conv1x1(64->1) | 65 |
| **Total** | **2,950,401** |

- channel_number=[32, 64, 128, 256, 256, 64], output_dim=1

---

## 10. AgeNet — 29,055,433 (29.06M) — Moderate-large

| Module | Parameters |
|---|---|
| Stem (3 convs + 2 maxpool) | 388,224 |
| Inception3b + 3c + Fire3 | 3,023,232 |
| Inception4b + 4c + Fire4 | 5,350,032 |
| Inception4d + 4e + Fire5 | 5,545,432 |
| Inception4f + 5b + Fire6 | 14,321,504 |
| FC(512->512->256->128->1) | 427,009 |
| **Total** | **29,055,433** |

- Hybrid CNN with Inception-like modules + Fire modules

---

## 11. ChenNet — 1,456,705 (1.46M) — Lightweight

| Layer | Parameters |
|---|---|
| Conv3d(1->16, k=3)+BN | 464 |
| Conv3d(16->32, k=3)+BN | 13,888 |
| Conv3d(32->64, k=3)+BN | 55,424 |
| Conv3d(64->128, k=3)+BN | 221,440 |
| Conv3d(128->256, k=3)+BN | 885,248 |
| Conv3d(256->64, k=1)+BN | 16,576 |
| AdaptiveAvgPool(2,2,2) -> FC(512->512) | 262,656 |
| FC(512->1) | 513 |
| **Total** | **1,456,705** |

- nr_filt=16, 6 conv layers with 1x1 bottleneck at layer 6

---

## Final Comparison Table (sorted by parameter count)

| Model | Parameters | Category |
|---|---|---|
| CBR-tiny | 77,249 (77K) | Very lightweight |
| ChenNet | 1,456,705 (1.46M) | Lightweight |
| SFCN | 2,950,401 (2.95M) | Moderate |
| BasicCNN | 4,267,713 (4.27M) | Moderate |
| EfficientNet-B0 | 4,689,661 (4.69M) | Moderate |
| DenseNet-121 | 11,243,649 (11.24M) | Moderate |
| U-Net Encoder | 14,198,753 (14.20M) | Moderate |
| AgeNet | 29,055,433 (29.06M) | Moderate-large |
| ResNet-18 | 33,160,513 (33.16M) | Large |
| ResNet-50 | 46,157,121 (46.16M) | Large |
| ResNet-34 | 63,470,145 (63.47M) | Large |

---

## Key Observations

1. **ResNet-50 < ResNet-34 in 3D**: Bottleneck blocks avoid repeated 3x3x3 convolutions (27x multiplier per kernel), making them more parameter-efficient than BasicBlocks in 3D.
2. **BasicCNN inflated by FC**: AdaptiveAvgPool(8^3) creates a large FC layer accounting for 98% of total params. The conv backbone itself is only ~73K.
3. **EfficientNet-B0** achieves competitive capacity with depthwise separable convolutions at only 4.69M params.
4. **CBR-tiny** is ultra-lightweight at 77K due to small pool size (4^3) and single output neuron FC.
5. All parameter counts were computed directly from the actual PyTorch model implementations (not theoretical estimates).
