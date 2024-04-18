#inpainting center
for video in blackswan bmx-trees breakdance camel car-roundabout car-shadow cows dance-twirl dog drift-chicane drift-straight goat horsejump-high kite-surf libby motocross-jump paragliding-launch parkour scooter-black soapbox
do
python train_nerv_all.py --outf inpanting_center/HNeRV/epoch_300 --model HNeRV \
   --data_path ./dataset/DAVIS/JPEGImages/1080p/$video$tail --vid $video \
   --optim_type Adam --conv_type convnext pshuffel --act gelu --norm none  --crop_list 1080_1920 \
   --resize_list -1 --loss L2 --enc_strds 5 3 2 2 2 --enc_dim 64_16  \
   --dec_strds 5 3 2 2 2 --ks 0_1_5 --reduce 1.2 --dec_blks 1 1 1 1 1 \
   --modelsize 3.0 -e 300 --eval_freq 30  --lower_width 12 -b 1 --lr 0.0005 --inpanting inpanting_center --clip_max_norm 1
done

#inpainting disperse
for video in blackswan bmx-trees breakdance camel car-roundabout car-shadow cows dance-twirl dog drift-chicane drift-straight goat horsejump-high kite-surf libby motocross-jump paragliding-launch parkour scooter-black soapbox
do
python train_nerv_all.py --outf inpanting_fixed_50/HNeRV/epoch_300 --model HNeRV \
   --data_path ./dataset/DAVIS/JPEGImages/1080p/$video$tail --vid $video \
   --optim_type Adam --conv_type convnext pshuffel --act gelu --norm none  --crop_list 1080_1920 \
   --resize_list -1 --loss L2 --enc_strds 5 3 2 2 2 --enc_dim 64_16  \
   --dec_strds 5 3 2 2 2 --ks 0_1_5 --reduce 1.2 --dec_blks 1 1 1 1 1 \
   --modelsize 3.0 -e 300 --eval_freq 30  --lower_width 12 -b 1 --lr 0.0005 --inpanting inpanting_fixed_50 --clip_max_norm 1
done

