import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions as dist
from einops import rearrange
from model_blocks import *
import time
from lib.quant_ops import CustomConv2d, CustomLinear

class HNeRV(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed = args.embed
        ks_enc, ks_dec1, ks_dec2 = [int(x) for x in args.ks.split('_')]
        enc_blks = args.enc_blks

        # BUILD Encoder LAYERS
        if len(args.enc_strds):         #HNeRV
            enc_dim1, enc_dim2 = [int(x) for x in args.enc_dim.split('_')]
            c_in_list, c_out_list = [enc_dim1] * len(args.enc_strds), [enc_dim1] * len(args.enc_strds)
            c_out_list[-1] = enc_dim2
            if args.conv_type[0] == 'convnext':
                self.encoder = ConvNeXt(stage_blocks=enc_blks, strds=args.enc_strds, dims=c_out_list,
                    drop_path_rate=0)
            else:
                c_in_list[0] = 3
                encoder_layers = []
                for c_in, c_out, strd in zip(c_in_list, c_out_list, args.enc_strds):
                    encoder_layers.append(NeRVBlock(dec_block=False, conv_type=args.conv_type[0], ngf=c_in,
                     new_ngf=c_out, ks=ks_enc, strd=strd, bias=True, norm=args.norm, act=args.act))
                self.encoder = nn.Sequential(*encoder_layers)
            hnerv_hw = np.prod(args.enc_strds) // np.prod(args.dec_strds)
            self.fc_h, self.fc_w = hnerv_hw, hnerv_hw
            ch_in = enc_dim2
        else:
            ch_in = 2 * int(args.embed.split('_')[-1])
            self.pe_embed = PositionEncoding(args.embed, args.lfreq)  
            self.encoder = nn.Identity()
            self.fc_h, self.fc_w = [int(x) for x in args.fc_hw.split('_')]

        # BUILD Decoder LAYERS  
        decoder_layers = []        
        ngf = args.fc_dim
        out_f = int(ngf * self.fc_h * self.fc_w)
        decoder_layer1 = NeRVBlock(dec_block=False, conv_type='conv', ngf=ch_in, new_ngf=out_f, ks=0, strd=1, 
            bias=True, norm=args.norm, act=args.act, sft_ngf=args.ch_t, args=args)
        decoder_layers.append(decoder_layer1)
        for i, strd in enumerate(args.dec_strds):                         
            reduction = sqrt(strd) if args.reduce==-1 else args.reduce
            new_ngf = int(max(round(ngf / reduction), args.lower_width))
            for j in range(args.dec_blks[i]):
                cur_blk = NeRVBlock(dec_block=True, conv_type=args.conv_type[1], ngf=ngf, new_ngf=new_ngf, 
                    ks=min(ks_dec1+2*i, ks_dec2), strd=1 if j else strd, bias=True, norm=args.norm, act=args.act, sft_ngf=args.ch_t, args=args)
                decoder_layers.append(cur_blk)
                ngf = new_ngf
        
        self.decoder = nn.ModuleList(decoder_layers)
        self.head_layer = CustomConv2d(ngf, 3, 3, 1, 1, args=args) 
        self.out_bias = args.out_bias
        if args.quant and args.model == "HNeRV":
            self.embed_quantizer = quant_map[args.quantizer_e](args.quant_embed_bit, signed=False, per_channel=args.per_channel_e)
            self.bitrate_e_dict = {}
        else:
            self.embed_quantizer = None

    def forward(self, input, input_embed=None, entropy_model=None, pre_img=None, post_img=None, norm_idx=None):
        if input_embed != None:
            img_embed = input_embed
        else:
            if 'pe' in self.embed:
                input = self.pe_embed(input[:,None]).float()
            img_embed = self.encoder(input)

        if self.embed_quantizer is not None:
            self.embed_quantizer.init_data(img_embed)
            code_e, quant_e, img_embed = self.embed_quantizer(img_embed)
            if entropy_model is not None:
                self.bitrate_e_dict.update(entropy_model.cal_bitrate(code_e, quant_e, self.training))

        if pre_img is not None and post_img is not None:
            img_embed = 0.5*(self.encoder(pre_img)+self.encoder(post_img))
        embed_list = [img_embed]
        dec_start = time.time()

        output = self.decoder[0](img_embed)
        n, c, h, w = output.shape
        output = output.view(n, -1, self.fc_h, self.fc_w, h, w).permute(0,1,4,2,5,3).reshape(n,-1,self.fc_h * h, self.fc_w * w)
        embed_list.append(output)
        for layer in self.decoder[1:]:
            output = layer(output) 
            embed_list.append(output)

        img_out = OutImg(self.head_layer(output), self.out_bias)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dec_time = time.time() - dec_start
        return  img_out, embed_list, dec_time


    def forward_encoder(self, input):
        img_embed = self.encoder(input)
        return img_embed

    def forward_embed_quant(self, img_embed, entropy_model=None):
        code, quant, img_embed = self.embed_quantizer(img_embed)
        if entropy_model is not None:
            self.bitrate_e_dict.update(entropy_model.cal_bitrate(code, quant, self.training))
        return code, quant, img_embed

    def forward_decoder(self, img_embed, norm_idx):
        embed_list = [img_embed]
        dec_start = time.time()
        output = self.decoder[0](img_embed)
        n, c, h, w = output.shape
        output = output.view(n, -1, self.fc_h, self.fc_w, h, w).permute(0,1,4,2,5,3).reshape(n,-1,self.fc_h * h, self.fc_w * w)
        embed_list.append(output)
        for layer in self.decoder[1:]:
            output = layer(output) 
            embed_list.append(output)
        img_out = OutImg(self.head_layer(output), self.out_bias)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dec_time = time.time() - dec_start
        return img_out, embed_list, dec_time

    def decoder_params(self):
        decoder_param = (sum([p.data.nelement() for p in self.parameters()]) - sum([p.data.nelement() for p in self.encoder.parameters()])) /1e6
        return decoder_param

    def cal_params(self, entropy_model=None):
        for m in self.modules():
            if type(m) in [CustomConv2d, CustomLinear]:
                code_w, quant_w, dequant_w = m.weight_quantizer(m.weight)
                m.dequant_w = dequant_w
                if m.bias is not None:
                    code_b, quant_b, dequant_b = m.bias_quantizer(m.bias)
                    m.dequant_b = dequant_b
                if entropy_model is not None:
                    m.bitrate_w_dict.update(entropy_model.cal_bitrate(code_w, quant_w, self.training))
                    if m.bias is not None:
                        m.bitrate_b_dict.update(entropy_model.cal_bitrate(code_b, quant_b, self.training))
                        
    def get_bitrate_sum(self, name="bitrate"):
        sum = 0
        for m in self.modules():
            if type(m) in [CustomConv2d, CustomLinear]:
                sum += m.bitrate_w_dict[name]
                if name in m.bitrate_b_dict.keys():
                    sum += m.bitrate_b_dict[name]
        return sum

    def init_data(self):
        for m in self.modules():
            if type(m) in [CustomConv2d, CustomLinear]:
                m.weight_quantizer.init_data(m.weight)
                if m.bias is not None:
                    m.bias_quantizer.init_data(m.bias)

class HNeRVDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fc_h, self.fc_w = [torch.tensor(x) for x in [model.fc_h, model.fc_w]]
        self.out_bias = model.out_bias
        self.decoder = model.decoder
        self.head_layer = model.head_layer

    def forward(self, img_embed):
        output = self.decoder[0](img_embed)
        n, c, h, w = output.shape
        output = output.view(n, -1, self.fc_h, self.fc_w, h, w).permute(0,1,4,2,5,3).reshape(n,-1,self.fc_h * h, self.fc_w * w)
        for layer in self.decoder[1:]:
            output = layer(output) 
        output = self.head_layer(output)
        return  OutImg(output, self.out_bias)


class HNeRV_Boost(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Encoder LAYERS
        self.embed = args.embed
        ks_enc, ks_dec1, ks_dec2 = [int(x) for x in args.ks.split('_')]
        enc_blks = args.enc_blks        
        
        enc_dim1, enc_dim2 = [int(x) for x in args.enc_dim.split('_')]
        c_in_list, c_out_list = [enc_dim1] * len(args.enc_strds), [enc_dim1] * len(args.enc_strds)
        c_out_list[-1] = enc_dim2
        self.encoder = ConvNeXt(stage_blocks=enc_blks, strds=args.enc_strds, dims=c_out_list, drop_path_rate=0)

        # Decoder LAYERS
        # first part: position embedding for time index
        self.pe_embed_t = PositionEncoding(args.embed, args.lfreq) #PositionEncoding(lbase=args.lbase, levels=args.levels, lfreq=args.lfreq)
        mlp_dim_list = [int(self.pe_embed_t.embed_length)] + [int(args.ch_t*2)] + [args.ch_t]
        self.stem_t = NeRV_MLP(dim_list=mlp_dim_list, bias=True, act=args.act, omega=1, args=args)

        # second part: reconstruction module
        decoder_layers = []        
        ngf = args.fc_dim
        decoder_layer1 = NeRVBlock(dec_block=False, conv_type='conv', ngf=enc_dim2, new_ngf=ngf, ks=0, strd=1, 
                bias=True, norm=args.norm, act=args.act, sft_ngf=args.ch_t, args=args) 
        decoder_layers.append(decoder_layer1)

        for i, strd in enumerate(args.dec_strds):                         
            reduction = sqrt(strd) if args.reduce ==-1 else args.reduce
            new_ngf = int(max(round(ngf / reduction), args.lower_width))
            for j in range(args.dec_blks[i]):
                cur_blk = NeRVBlock(dec_block=True, conv_type=args.conv_type[1], ngf=ngf, new_ngf=new_ngf, 
                    ks=min(ks_dec1+2*i, ks_dec2), strd=1 if j else strd, bias=True, norm=args.norm, act=args.act, sft_ngf=args.ch_t, args=args)
                decoder_layers.append(cur_blk)
                ngf = new_ngf

        self.decoder = nn.ModuleList(decoder_layers)
        self.head_layer = CustomConv2d(ngf, 3, 3, 1, 1, args=args) 
        self.out_bias = args.out_bias
        if args.quant:
            self.embed_quantizer = quant_map[args.quantizer_e](args.quant_embed_bit, signed=False, per_channel=args.per_channel_e)
            self.bitrate_e_dict = {}
        else:
            self.embed_quantizer = None

        self.outf = args.outf

    def forward(self, input, input_embed=None, entropy_model=None, pre_img=None, post_img=None, norm_idx=None,):
        if input_embed != None:
            img_embed = input_embed
        else:
            img_embed = self.encoder(input)

        if self.embed_quantizer is not None:
            self.embed_quantizer.init_data(img_embed)
            code_e, quant_e, img_embed = self.embed_quantizer(img_embed)
            if entropy_model is not None:
                self.bitrate_e_dict.update(entropy_model.cal_bitrate(code_e, quant_e, self.training))

        if pre_img is not None and post_img is not None:
            img_embed = 0.5*(self.encoder(pre_img)+self.encoder(post_img))
        
        embed_list = [img_embed]
        dec_start = time.time()
        t_embed = self.stem_t(self.pe_embed_t(norm_idx[:, None]).float())
        output = self.decoder[0]((img_embed, t_embed))
        embed_list.append(output)
        for layer in self.decoder[1:]:
            output = layer((output, t_embed)) 
            embed_list.append(output)
        img_out = OutImg(self.head_layer(output), self.out_bias)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dec_time = time.time() - dec_start
        return img_out, embed_list, dec_time


    def forward_encoder(self, input):
        img_embed = self.encoder(input)
        return img_embed

    def forward_embed_quant(self, img_embed, entropy_model=None):
        code, quant, img_embed = self.embed_quantizer(img_embed)
        if entropy_model is not None:
            self.bitrate_e_dict.update(entropy_model.cal_bitrate(code, quant, self.training))
        return code, quant, img_embed

    def forward_decoder(self, img_embed, norm_idx):
        embed_list = [img_embed]
        dec_start = time.time()
        t_embed = self.stem_t(self.pe_embed_t(norm_idx[:, None]).float())
        output = self.decoder[0]((img_embed, t_embed))
        embed_list.append(output)
        for layer in self.decoder[1:]:
            output = layer((output, t_embed)) 
            embed_list.append(output)
        img_out = OutImg(self.head_layer(output), self.out_bias)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dec_time = time.time() - dec_start
        return img_out, embed_list, dec_time

    def decoder_params(self):
        decoder_param = (sum([p.data.nelement() for p in self.parameters()]) - sum([p.data.nelement() for p in self.encoder.parameters()])) /1e6
        return decoder_param

    def stage_params(self):
        model_params = self.decoder_params()
        stage0 = sum([p.data.nelement() for p in self.decoder[0].parameters()])/1e6/model_params
        ratio_list = [stage0, 0, 0, 0, 0, 0]
        index = 1
        for i, strd in enumerate(self.dec_strds): 
            for j in range(self.dec_blks[i]):
                ratio_list[i+1] += sum([p.data.nelement() for p in self.decoder[index].parameters()])/1e6/model_params
                index += 1
        return ratio_list
        #print("params distribution:", ratio_list)

    def cal_params(self, entropy_model=None):
        for m in self.modules():
            if type(m) in [CustomConv2d, CustomLinear]:
                code_w, quant_w, dequant_w = m.weight_quantizer(m.weight)
                m.dequant_w = dequant_w
                if m.bias is not None:
                    code_b, quant_b, dequant_b = m.bias_quantizer(m.bias)
                    m.dequant_b = dequant_b
                if entropy_model is not None:
                    m.bitrate_w_dict.update(entropy_model.cal_bitrate(code_w, quant_w, self.training))
                    if m.bias is not None:
                        m.bitrate_b_dict.update(entropy_model.cal_bitrate(code_b, quant_b, self.training))

    def get_bitrate_sum(self, name="bitrate"):
        sum = 0
        for m in self.modules():
            if type(m) in [CustomConv2d, CustomLinear]:
                sum += m.bitrate_w_dict[name]
                if name in m.bitrate_b_dict.keys():
                    sum += m.bitrate_b_dict[name]
        return sum

    def init_data(self):
        for m in self.modules():
            if type(m) in [CustomConv2d, CustomLinear]:
                m.weight_quantizer.init_data(m.weight)
                if m.bias is not None:
                    m.bias_quantizer.init_data(m.bias)