#2.8: 3M, 4.6: 5M, 9.1: 10M, 13.6: 15M
tail="_1920x1080_120"
for size in 2.8 4.6 9.1 13.6
do
for video in Beauty Bosphorus HoneyBee Jockey ReadySteadyGo YachtRide
do
python train_nerv_compression.py --outf compression/HNeRV_Boost/target4 --model HNeRV_Boost --sft_block res_sft --ch_t 32 \
   --data_path ./dataset/UVG_Full/$video$tail --vid $video \
   --optim_type Adan --conv_type convnext pshuffel_3x3 --act sin --norm none  --crop_list 1080_1920 \
   --resize_list -1 --loss Fusion10_freq --embed pe_1.25_80 --enc_strds 5 3 2 2 2 --enc_dim 64_16  \
   --dec_strds 5 3 2 2 2 --ks 0_1_5 --reduce 1.2 --dec_blks 1 1 2 2 2 \
   --modelsize $size -e 100 --eval_freq 30  --lower_width 12 -b 1 --lr 0.0005 \
   --weight ./output/regression/HNeRV_Boost/epoch_300/$video/Size$size/model_latest.pth \
   --lr_type cosine_0_1_0.1 --not_resume --embed_entropy \
   --quant --quant_model_bit 8 --quant_bias_bit 8 --quant_embed_bit 8 --quantizer_w scale \
   --quantizer_b scale --quantizer_e scalebeta --lambda_rate 0.05 --target_bit 4
done 
done

# for size in 2.75 4.5 9.05 13.55
# do
# for video in ShakeNDry
# do
# python train_nerv_compression.py --outf compression/HNeRV_Boost/target4 --model HNeRV_Boost --sft_block res_sft --ch_t 32 \
#    --data_path /mnt/cachenew2/zhangxinjie/dataset/UVG_Full/$video$tail --vid $video \
#    --optim_type Adan --conv_type convnext pshuffel_3x3 --act sin --norm none  --crop_list 1080_1920 \
#    --resize_list -1 --loss Fusion10_freq --embed pe_1.25_80 --enc_strds 5 3 2 2 2 --enc_dim 64_16  \
#    --dec_strds 5 3 2 2 2 --ks 0_1_5 --reduce 1.2 --dec_blks 1 1 2 2 2 \
#    --modelsize $size -e 100 --eval_freq 30  --lower_width 12 -b 1 --lr 0.0005 \
#    --weight /mnt/cachenew2/zhangxinjie/INR/BoostNeRV/output/regression/HNeRV_Boost/epoch_300/$video/Size$size/model_latest.pth \
#    --lr_type cosine_0_1_0.1 --not_resume --embed_entropy \
#    --quant --quant_model_bit 8 --quant_bias_bit 8 --quant_embed_bit 8 --quantizer_w scale \
#    --quantizer_b scale --quantizer_e scalebeta --lambda_rate 0.05 --target_bit 4
# done 
# done