#0.64: 0.75M, 1.275: 1.5M, 2.65: 3M
for size in 0.64 1.275 2.65 
do
python train_nerv_all.py --outf regression/HNeRV_Boost/epoch_300 --model HNeRV_Boost --sft_block res_sft --ch_t 32 \
   --data_path ./dataset/bunny --vid bunny \
   --optim_type Adan --conv_type convnext pshuffel_3x3 --act sin --norm none  --crop_list 720_1280 \
   --resize_list -1 --loss Fusion10_freq --embed pe_1.25_80 --enc_strds 5 2 2 2 2 --enc_dim 64_16  \
   --dec_strds 5 2 2 2 2 --ks 0_1_5 --reduce 1.2 --dec_blks 1 1 2 2 2 \
   --modelsize $size -e 300 --eval_freq 30  --lower_width 12 -b 1 --lr 0.003
done 

# for epoch in 600 1200 2400 4800
# do
# python train_nerv_all.py --outf regression/HNeRV_Boost/epoch_$epoch --model HNeRV_Boost --sft_block res_sft --ch_t 32 \
#    --data_path ./dataset/bunny --vid bunny \
#    --optim_type Adan --conv_type convnext pshuffel_3x3 --act sin --norm none  --crop_list 720_1280 \
#    --resize_list -1 --loss Fusion10_freq --embed pe_1.25_80 --enc_strds 5 2 2 2 2 --enc_dim 64_16  \
#    --dec_strds 5 2 2 2 2 --ks 0_1_5 --reduce 1.2 --dec_blks 1 1 2 2 2 \
#    --modelsize 1.275 -e $epoch --eval_freq 30  --lower_width 12 -b 1 --lr 0.003
# done 
