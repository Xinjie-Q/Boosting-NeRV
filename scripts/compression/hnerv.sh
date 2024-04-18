#3.05: 3M, 5.1: 5M, 10.1: 10M, 15.1: 15M
#This is HNeRV model with our proposed CEM technique to achieve video compression
#In paper, the video compression results of original HNeRV use the three-step compression from NeRV.
tail="_1920x1080_120"
for size in 3.05 5.1 10.1 15.1
do
for video in Beauty Bosphorus HoneyBee Jockey ReadySteadyGo YachtRide ShakeNDry
do
python train_nerv_compression.py --outf compression/HNeRV/target4 --model HNeRV \
   --data_path ./dataset/UVG_Full/$video$tail --vid $video \
   --optim_type Adam --conv_type convnext pshuffel --act gelu --norm none  --crop_list 1080_1920 \
   --resize_list -1 --loss L2 --enc_strds 5 3 2 2 2 --enc_dim 64_16  \
   --dec_strds 5 3 2 2 2 --ks 0_1_5 --reduce 1.2 --dec_blks 1 1 1 1 1 \
   --modelsize $size -e 100 --eval_freq 30  --lower_width 12 -b 1 --lr 0.0005 \
   --weight ./output/regression/HNeRV/epoch_300/$video/Size$size/model_latest.pth \
   --lr_type cosine_0_1_0.1 --not_resume --embed_entropy \
   --quant --quant_model_bit 8 --quant_bias_bit 8 --quant_embed_bit 8 --quantizer_w scale \
   --quantizer_b scale --quantizer_e scalebeta --lambda_rate 0.05 --target_bit 4
done 
done

