python lit_trainer_seg.py --model resnet_seg --exp_name ratio_seg --recon_loss_weight 2 --gpu 1 &
python lit_trainer_seg.py --model resnet_seg --exp_name ratio_seg --recon_loss_weight 3 --gpu 2
wait
python lit_trainer_seg.py --model resnet_seg --exp_name ratio_seg --recon_loss_weight 4 --gpu 1 &
python lit_trainer_seg.py --model resnet_seg --exp_name ratio_seg --recon_loss_weight 0.7 --gpu 2
wait
python lit_trainer_seg.py --model resnet_seg --exp_name ratio_seg --recon_loss_weight 0.9 --gpu 1 &
python lit_trainer_seg.py --model resnet_seg --exp_name ratio_seg --recon_loss_weight 8 --gpu 2


# options=("ratio_seg")

#         model="unet_seg"


#         model="resnet_seg"


# options=("t1t2ratio_seg" "t1t2_seg" "t1_seg" "t2_seg")

#         model="unet_seg"


#         model="resnet_seg"
