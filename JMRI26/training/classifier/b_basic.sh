#!/bin/bash


models=("t1" "t2" "t1t2" "ratio" "t1t2ratio")

for fold in {0..4}; do
    for model in "${models[@]}"; do
        exp_name="${model}"
        echo "Running model: $model, fold: $fold (Phase 1)"

        python lit_trainer.py --model agenet --exp_name "${exp_name}" --fold ${fold} --gpu 0 &
        sleep 1
        python lit_trainer.py --model basiccnn --exp_name "${exp_name}" --fold ${fold} --gpu 1 &
        sleep 1
        python lit_trainer.py --model cbr_tiny --exp_name "${exp_name}" --fold ${fold} --gpu 2 &
        sleep 1
        python lit_trainer.py --model chen_net --exp_name "${exp_name}" --fold ${fold} --gpu 3 &
        sleep 1
        python lit_trainer.py --model densenet --exp_name "${exp_name}" --fold ${fold} --gpu 4 &
        sleep 1
        python lit_trainer.py --model efficientnet --exp_name "${exp_name}" --fold ${fold} --gpu 0 &
        sleep 1
        python lit_trainer.py --model resnet18 --exp_name "${exp_name}" --fold ${fold} --gpu 1 &
        sleep 1
        python lit_trainer.py --model resnet34 --exp_name "${exp_name}" --fold ${fold} --gpu 2 &
        sleep 1
        python lit_trainer.py --model resnet50 --exp_name "${exp_name}" --fold ${fold} --gpu 3 &
        sleep 1
        python lit_trainer.py --model sfcn --loss kld --exp_name "${exp_name}" --fold ${fold} --gpu 4
        sleep 1

        wait

        python lit_trainer.py --model unet --exp_name t1 --fold ${fold} --gpu 0 &
        sleep 1
        python lit_trainer.py --model unet --exp_name t2 --fold ${fold} --gpu 1 &
        sleep 1
        python lit_trainer.py --model unet --exp_name t1t2 --fold ${fold} --gpu 2 &
        sleep 1
        python lit_trainer.py --model unet --exp_name ratio --fold ${fold} --gpu 3 &
        sleep 1
        python lit_trainer.py --model unet --exp_name t1t2ratio --fold ${fold} --gpu 4
        sleep 1

        wait
    done
done


for fold in {0..4}; do
    for model in "${models[@]}"; do
        exp_name="${model}"
        echo "Running model: $model, fold: $fold (Phase 1)"

        python lit_trainer.py --model agenet_film -data_option freesurfer -add_option both --exp_name "${exp_name}" --fold ${fold} --gpu 0 &
        sleep 1
        python lit_trainer.py --model basiccnn_film -data_option freesurfer -add_option both --exp_name "${exp_name}" --fold ${fold} --gpu 1 &
        sleep 1
        python lit_trainer.py --model cbr_tiny_film -data_option freesurfer -add_option both --exp_name "${exp_name}" --fold ${fold} --gpu 2 &
        sleep 1
        python lit_trainer.py --model chen_net_film -data_option freesurfer -add_option both --exp_name "${exp_name}" --fold ${fold} --gpu 3 &
        sleep 1
        python lit_trainer.py --model densenet_film -data_option freesurfer -add_option both --exp_name "${exp_name}" --fold ${fold} --gpu 4 &
        sleep 1
        python lit_trainer.py --model efficientnet_film -data_option freesurfer -add_option both --exp_name "${exp_name}" --fold ${fold} --gpu 0 &
        sleep 1
        python lit_trainer.py --model resnet18_film -data_option freesurfer -add_option both --exp_name "${exp_name}" --fold ${fold} --gpu 1 &
        sleep 1
        python lit_trainer.py --model resnet34_film -data_option freesurfer -add_option both --exp_name "${exp_name}" --fold ${fold} --gpu 2
        sleep 1
        python lit_trainer.py --model resnet50_film -data_option freesurfer -add_option both --exp_name "${exp_name}" --fold ${fold} --gpu 3 &
        sleep 1
        python lit_trainer.py --model sfcn_film -data_option freesurfer -add_option both --loss kld --exp_name "${exp_name}" --fold ${fold} --gpu 4
        sleep 1

        wait
    done
done
