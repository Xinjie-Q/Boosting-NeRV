#2.75: 3M
tail="_1920x1080_120"
for video in Beauty Bosphorus HoneyBee Jockey ReadySteadyGo YachtRide
do
python train_nerv_all.py --outf regression/HNeRV_Boost/epoch_300 --model HNeRV_Boost --sft_block res_sft --ch_t 32 \
   --data_path ./dataset/UVG_Full/$video$tail --vid $video \
   --optim_type Adan --conv_type convnext pshuffel_3x3 --act sin --norm none  --crop_list 1080_1920 \
   --resize_list -1 --loss Fusion10_freq --embed pe_1.05_80 --enc_strds 5 3 2 2 2 --enc_dim 64_16  \
   --dec_strds 5 3 2 2 2 --ks 0_1_5 --reduce 1.2 --dec_blks 1 1 2 2 2 \
   --modelsize 2.75 -e 300 --eval_freq 30  --lower_width 12 -b 1 --lr 0.001 --interpolation --data_split 1_1_2 --embed_inter
done


for video in ShakeNDry
do
python train_nerv_all.py --outf regression/HNeRV_Boost/epoch_300 --model HNeRV_Boost --sft_block res_sft --ch_t 32 \
   --data_path ./dataset/UVG_Full/$video$tail --vid $video \
   --optim_type Adan --conv_type convnext pshuffel_3x3 --act sin --norm none  --crop_list 1080_1920 \
   --resize_list -1 --loss Fusion10_freq --embed pe_1.05_80 --enc_strds 5 3 2 2 2 --enc_dim 64_16  \
   --dec_strds 5 3 2 2 2 --ks 0_1_5 --reduce 1.2 --dec_blks 1 1 2 2 2 \
   --modelsize 2.7 -e 300 --eval_freq 30  --lower_width 12 -b 1 --lr 0.001 --interpolation --data_split 1_1_2 --embed_inter
done 
