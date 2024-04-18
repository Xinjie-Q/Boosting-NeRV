import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions as dist
from einops import rearrange
from model_blocks import *
import time
from lib.quant_ops import CustomConv2d, CustomLinear

class NeRV_Boost(nn.Module):
    def __init__(self, expansion=1, args=None):
        super().__init__()
        self.encoder = nn.Identity()
        self.pe_t = PositionEncoding(args.embed, args.lfreq)
        self.fc_h, self.fc_w = [int(x) for x in args.fc_hw.split('_')]
        self.fc_dim = args.fc_dim
        mlp_dim_list = [self.pe_t.embed_length] + [256] + [self.fc_h *self.fc_w *self.fc_dim]
        self.stem = NeRV_MLP(dim_list=mlp_dim_list, bias=True, act=args.act, omega=1, args=args)
        self.stem_t = NeRV_MLP(dim_list=[int(self.pe_t.embed_length), int(args.ch_t*2), args.ch_t], bias=True, act=args.act, omega=1, args=args)
    
        # BUILD CONV LAYERS
        self.layers = nn.ModuleList()
        ngf = self.fc_dim
        ks_enc, ks_dec1, ks_dec2 = [int(x) for x in args.ks.split('_')]
        for i, stride in enumerate(args.dec_strds):
            if i == 0:
                # expand channel width at first stage
                new_ngf = int(ngf * expansion)
            else:
                # change the channel width for each stage
                new_ngf = int(max(ngf // (1 if stride == 1 else args.reduce), args.lower_width))

            for j in range(args.dec_blks[i]):
                self.layers.append(NeRVBlock(dec_block=True, conv_type=args.conv_type[1], ngf=ngf, new_ngf=new_ngf, 
                    ks=min(ks_dec1+2*i, ks_dec2), strd=1 if j else stride, bias=True, norm=args.norm, act=args.act, 
                    sft_ngf=args.ch_t, args=args, dump_features=False)
                )
                ngf = new_ngf

        self.head_layer = CustomConv2d(ngf, 3, 1, 1, bias=True, args=args)
        self.out_bias = args.out_bias
        self.outf = args.outf

    def forward(self, input, input_embed=None, norm_idx=None):
        dec_start = time.time()
        t = input[:,None].float()
        t_embed = self.pe_t(t)
        output = self.stem(t_embed) # [B, L, 1, 1]
        output = output.view(output.size(0), self.fc_dim, self.fc_h, self.fc_w)
        t_embed = self.stem_t(t_embed)
        out_list = []
        for layer in self.layers:
            output = layer((output, t_embed))
            out_list.append(output)
        img_out = self.head_layer(output)
        img_out = OutImg(img_out, self.out_bias)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dec_time = time.time() - dec_start
        return img_out, out_list, dec_time

    def decoder_params(self):
        decoder_param = (sum([p.data.nelement() for p in self.parameters()])) /1e6
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
