#inpainting center
for video in blackswan bmx-trees breakdance camel car-roundabout car-shadow cows dance-twirl dog drift-chicane drift-straight goat horsejump-high kite-surf libby motocross-jump paragliding-launch parkour scooter-black soapbox
do
python train_nerv_all.py --outf inpanting_center/HNeRV_Boost/epoch_300 --model HNeRV_Boost --sft_block res_sft --ch_t 32 \
   --data_path ./dataset/DAVIS/JPEGImages/1080p/$video --vid $video \
   --optim_type Adan --conv_type convnext pshuffel_3x3 --act sin --norm none  --crop_list 1080_1920 \
   --resize_list -1 --loss Fusion10_freq --embed pe_1.25_80 --enc_strds 5 3 2 2 2 --enc_dim 64_16  \
   --dec_strds 5 3 2 2 2 --ks 0_1_5 --reduce 1.2 --dec_blks 1 1 2 2 2 \
   --modelsize 3 -e 300 --eval_freq 30  --lower_width 12 -b 1 --lr 0.003 --inpanting inpanting_center 
done 

#inpainting disperse
for video in blackswan bmx-trees breakdance camel car-roundabout car-shadow cows dance-twirl dog drift-chicane drift-straight goat horsejump-high kite-surf libby motocross-jump paragliding-launch parkour scooter-black soapbox
do
python train_nerv_all.py --outf inpanting_fixed_50/HNeRV_Boost/epoch_300 --model HNeRV_Boost --sft_block res_sft --ch_t 32 \
   --data_path ./dataset/DAVIS/JPEGImages/1080p/$video --vid $video \
   --optim_type Adan --conv_type convnext pshuffel_3x3 --act sin --norm none  --crop_list 1080_1920 \
   --resize_list -1 --loss Fusion10_freq --embed pe_1.25_80 --enc_strds 5 3 2 2 2 --enc_dim 64_16  \
   --dec_strds 5 3 2 2 2 --ks 0_1_5 --reduce 1.2 --dec_blks 1 1 2 2 2 \
   --modelsize 3 -e 300 --eval_freq 30  --lower_width 12 -b 1 --lr 0.003 --inpanting inpanting_fixed_50 
done 
