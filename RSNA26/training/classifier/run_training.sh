#!/bin/bash
# 5-fold training at 0.6 mm for each architecture and input type.
# Usage: bash run_training.sh [GPU_ID]
GPU=${1:-0}
exps=(t1_06 t2_06 t1t2_06 ratio_06 ratio_origin_06 t1t2_ratio_06 t1t2_ratio_origin_06)

for exp in "${exps[@]}"; do
    for fold in {0..4}; do
        for model in resnet chen_net cbr_tiny basiccnn; do
            python lit_trainer2.py --model "$model" --exp_name "$exp" --fold "$fold" --gpu "$GPU"
        done
        python lit_trainer2.py --model sfcn --loss kld --exp_name "$exp" --fold "$fold" --gpu "$GPU"
    done
done
