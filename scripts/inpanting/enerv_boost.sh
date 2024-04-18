#1.8: 3M
#inpainting center
for video in blackswan bmx-trees breakdance camel car-roundabout car-shadow cows dance-twirl dog drift-chicane drift-straight goat horsejump-high kite-surf libby motocross-jump paragliding-launch parkour scooter-black soapbox
do
python train_nerv_all.py --outf inpanting_center/ENeRV_Boost/epoch_300 --model ENeRV_Boost --sft_block res_sft --ch_t 32 --block_dim 128 \
   --data_path ./dataset/DAVIS/JPEGImages/1080p/$video --vid $video \
   --optim_type Adan --conv_type convnext pshuffel_3x3 --act sin --norm none  --crop_list 1080_1920 \
   --resize_list -1 --loss Fusion10_freq --embed pe_1.25_80 --fc_hw 9_16 \
   --dec_strds 5 3 2 2 2 --ks 0_3_3 --reduce 2 --dec_blks 1 1 2 2 2 \
   --modelsize 1.8 -e 300 --eval_freq 30  --lower_width 12 -b 1 --lr 0.0015 --inpanting inpanting_center 
done 

#inpainting disperse
for video in blackswan bmx-trees breakdance camel car-roundabout car-shadow cows dance-twirl dog drift-chicane drift-straight goat horsejump-high kite-surf libby motocross-jump paragliding-launch parkour scooter-black soapbox
do
python train_nerv_all.py --outf inpanting_fixed_50/ENeRV_Boost/epoch_300 --model ENeRV_Boost --sft_block res_sft --ch_t 32 --block_dim 128 \
   --data_path ./DAVIS/JPEGImages/1080p/$video --vid $video \
   --optim_type Adan --conv_type convnext pshuffel_3x3 --act sin --norm none  --crop_list 1080_1920 \
   --resize_list -1 --loss Fusion10_freq --embed pe_1.25_80 --fc_hw 9_16 \
   --dec_strds 5 3 2 2 2 --ks 0_3_3 --reduce 2 --dec_blks 1 1 2 2 2 \
   --modelsize 1.8 -e 300 --eval_freq 30  --lower_width 12 -b 1 --lr 0.0015 --inpanting inpanting_fixed_50 
done 
