#0.77: 0.75M, 1.525: 1.5M, 3.05: 3M
for size in 0.77 1.525 3.05 
do
python train_nerv_all.py --outf regression/HNeRV/epoch_300 --model HNeRV \
   --data_path ./dataset/bunny --vid bunny \
   --optim_type Adam --conv_type convnext pshuffel --act gelu --norm none  --crop_list 720_1280 \
   --resize_list -1 --loss L2 --enc_strds 5 2 2 2 2 --enc_dim 64_16  \
   --dec_strds 5 2 2 2 2 --ks 0_1_5 --reduce 1.2 --dec_blks 1 1 1 1 1 \
   --modelsize $size -e 300 --eval_freq 30  --lower_width 12 -b 1 --lr 0.001
done 

# for epoch in 600 1200 2400 4800
# do
# python train_nerv_all.py --outf regression/HNeRV/epoch_$epoch --model HNeRV \
#    --data_path ./dataset/bunny --vid bunny \
#    --optim_type Adam --conv_type convnext pshuffel --act gelu --norm none  --crop_list 720_1280 \
#    --resize_list -1 --loss L2 --enc_strds 5 2 2 2 2 --enc_dim 64_16  \
#    --dec_strds 5 2 2 2 2 --ks 0_1_5 --reduce 1.2 --dec_blks 1 1 1 1 1 \
#    --modelsize 1.525 -e $epoch --eval_freq 30  --lower_width 12 -b 1 --lr 0.001
# done 
