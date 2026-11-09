#!/bin/bash
# 5-fold training at 1.5 mm with and without FiLM conditioning on WM and ICV.
# Usage: bash run_training.sh [GPU_ID]
GPU=${1:-0}
exps=(t1_15 t2_15 t1t2_15 ratio_origin_15 t1t2_ratio_origin_15)

for exp in "${exps[@]}"; do
    for fold in {0..4}; do
        python lit_trainer.py --model resnet --exp_name "$exp" --fold "$fold" --gpu "$GPU"
        for opt in wm ratio both; do
            python lit_trainer.py --model resnet_film --exp_name "$exp" --fold "$fold" -add_option "$opt" -data_option freesurfer --gpu "$GPU"
        done
        for model in resnet_film2 resnet_film3 resnet_film4; do
            python lit_trainer.py --model "$model" --exp_name "$exp" --fold "$fold" -add_option both -data_option freesurfer --gpu "$GPU"
        done
    done
done
