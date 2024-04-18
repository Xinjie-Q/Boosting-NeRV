# 0.55: 0.75M, 1.25: 1.5M, 2: 3M
for size in 0.55 1.25 2.2
do
python train_nerv_all.py --outf regression/ENeRV_Boost/epoch_300 --model ENeRV_Boost --sft_block res_sft --ch_t 32 --block_dim 128 \
   --data_path ./dataset/bunny --vid bunny \
   --optim_type Adan --conv_type convnext pshuffel_3x3 --act sin --norm none  --crop_list 720_1280 \
   --resize_list -1 --loss Fusion10_freq --embed pe_1.25_80 --fc_hw 9_16 \
   --dec_strds 5 2 2 2 2 --ks 0_3_3 --reduce 2 --dec_blks 1 1 2 2 2 \
   --modelsize $size -e 300 --eval_freq 30  --lower_width 12 -b 1 --lr 0.0015
done 

# for epoch in 600 1200 2400 4800
# do
# python train_nerv_all.py --outf regression/ENeRV_Boost/epoch_$epoch --model ENeRV_Boost --sft_block res_sft --ch_t 32 --block_dim 128 \
#    --data_path ./dataset/bunny --vid bunny \
#    --optim_type Adan --conv_type convnext pshuffel_3x3 --act sin --norm none  --crop_list 720_1280 \
#    --resize_list -1 --loss Fusion10_freq --embed pe_1.25_80 --fc_hw 9_16 \
#    --dec_strds 5 2 2 2 2 --ks 0_3_3 --reduce 2 --dec_blks 1 1 2 2 2 \
#    --modelsize 1.25 -e $epoch --eval_freq 30  --lower_width 12 -b 1 --lr 0.0015
# done 

