#3.05: 3M, 5.1: 5M, 10.1: 10M, 15.1: 15M
tail="_1920x1080_120"
for size in 3.05 5.1 10.1 15.1
do
for video in Beauty Bosphorus HoneyBee Jockey ReadySteadyGo YachtRide ShakeNDry
do
python train_nerv_all.py --outf regression/HNeRV/epoch_300 --model HNeRV \
   --data_path ./dataset/UVG_Full/$video$tail --vid $video \
   --optim_type Adam --conv_type convnext pshuffel --act gelu --norm none  --crop_list 1080_1920 \
   --resize_list -1 --loss L2 --enc_strds 5 3 2 2 2 --enc_dim 64_16  \
   --dec_strds 5 3 2 2 2 --ks 0_1_5 --reduce 1.2 --dec_blks 1 1 1 1 1 \
   --modelsize $size -e 300 --eval_freq 30  --lower_width 12 -b 1 --lr 0.001
done 
done

