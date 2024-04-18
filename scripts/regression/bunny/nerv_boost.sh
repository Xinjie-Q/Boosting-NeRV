#0.375: 0.75M, 0.8: 1.5M, 1.65: 3M
for size in 0.375 0.8 1.65
do
python train_nerv_all.py --outf regression/NeRV_Boost/epoch_300 --model NeRV_Boost --sft_block res_sft --ch_t 32 \
   --data_path ./dataset/bunny --vid bunny \
   --optim_type Adan --conv_type convnext pshuffel_3x3 --act sin --norm none  --crop_list 720_1280 \
   --resize_list -1 --loss Fusion10_freq --embed pe_1.25_80 --fc_hw 9_16 \
   --dec_strds 5 2 2 2 2 --ks 0_3_3 --reduce 2 --dec_blks 1 1 2 2 2 \
   --modelsize $size -e 300 --eval_freq 30  --lower_width 12 -b 1 --lr 0.003
done 

# for epoch in 600 1200 2400 4800
# do
# python train_nerv_all.py --outf regression/NeRV_Boost/epoch_$epoch --model NeRV_Boost --sft_block res_sft --ch_t 32 \
#    --data_path ./dataset/bunny --vid bunny \
#    --optim_type Adan --conv_type convnext pshuffel_3x3 --act sin --norm none  --crop_list 720_1280 \
#    --resize_list -1 --loss Fusion10_freq --embed pe_1.25_80 --fc_hw 9_16 \
#    --dec_strds 5 2 2 2 2 --ks 0_3_3 --reduce 2 --dec_blks 1 1 2 2 2 \
#    --modelsize 0.8 -e $epoch --eval_freq 30  --lower_width 12 -b 1 --lr 0.003
# done 