import os
import time
import torch
import torch.nn as nn
from math import pi, sqrt, ceil
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_, DropPath
import decord
decord.bridge.set_bridge('torch')
from lib.quant_ops import CustomConv2d, CustomLinear, quant_map

###################################  Basic layers like position encoding/ downsample layers/ upscale blocks   ###################################
class NeRVBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        conv = UpConv if kargs['dec_block'] else DownConv
        self.conv = conv(ngf=kargs['ngf'], new_ngf=kargs['new_ngf'], strd=kargs['strd'], ks=kargs['ks'], 
            conv_type=kargs['conv_type'], bias=kargs['bias'], args=kargs['args'])
        self.norm = NormLayer(kargs['norm'], kargs['new_ngf'])
        self.act = ActivationLayer(kargs['act'])
        args = kargs['args']
        self.dec_block = kargs['dec_block'] or len(args.enc_strds)
        if args.sft_block == "res_sft" and kargs["sft_ngf"]!=0:
            if kargs['dec_block'] or len(args.enc_strds):
                sft_ch = kargs['new_ngf']
            else:
                self.fc_h, self.fc_w = [int(x) for x in args.fc_hw.split('_')]
                sft_ch = int(kargs['new_ngf']/(self.fc_h*self.fc_w))
            self.sft_block = ResBlock_SFT(sft_ch, sft_ch, cond_ch=kargs["sft_ngf"], 
                    in_act="relu", out_act="gelu", omega=1, args=kargs['args'])

        
    def forward(self, x):
        if isinstance(x, tuple):
            embed = x[1]
            x0 = self.act(self.norm(self.conv(x[0])))
            if self.dec_block:
                x = self.sft_block((x0, embed))   
            else:
                n, c, h, w = x0.shape
                x = x0.view(n, -1, self.fc_h, self.fc_w, h, w).permute(0,1,4,2,5,3).reshape(n,-1,self.fc_h * h, self.fc_w * w)
                x = self.sft_block((x, embed))
            return x
        else:
            return self.act(self.norm(self.conv(x)))

def Quantize_tensor(img_embed, quant_bit):
    out_min = img_embed.min(dim=1, keepdim=True)[0]
    out_max = img_embed.max(dim=1, keepdim=True)[0]
    scale = (out_max - out_min) / 2 ** quant_bit
    img_embed = ((img_embed - out_min) / scale).round()
    img_embed = out_min + scale * img_embed  
    return img_embed


def OutImg(x, out_bias='tanh'):
    if out_bias == 'sigmoid':
        return torch.sigmoid(x)
    elif out_bias == 'tanh':
        return (torch.tanh(x) * 0.5) + 0.5
    else:
        return x + float(out_bias)


def NeRV_MLP(dim_list, act='relu', bias=True, omega=1., args=None):
    act_fn = ActivationLayer(act)
    fc_list = []
    for i in range(len(dim_list) - 1):
        fc_list += [CustomConv2d(dim_list[i], dim_list[i+1], kernel_size=1, bias=bias, args=args), act_fn]
    return nn.Sequential(*fc_list)


class ResBlock_SFT(nn.Module):
    def __init__(self, in_ch, out_ch, cond_ch, factor=1, in_act="relu", out_act="gelu", omega=1., args=None):
        super().__init__()
        self.sft0 = SFTLayer(cond_ch, in_ch, factor, in_act, omega, args=args)
        self.conv0 = CustomConv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, args=args)
        self.sft1 = SFTLayer(cond_ch, out_ch, factor, in_act, omega, args=args)
        self.conv1 = CustomConv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, args=args)
        self.act = ActivationLayer(act_type=out_act)       

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = self.sft0(x)
        fea = self.act(self.conv0(fea))
        fea = self.sft1((fea, x[1]))
        fea = self.conv1(fea)
        return x[0] + fea #(x[0] + fea, x[1])  # return a tuple containing features and conditions


class SFTLayer(nn.Module):
    def __init__(self, in_ch, out_ch, factor=1, act="relu", omega=1., args=None):
        super().__init__()
        self.SFT_scale_conv0 = CustomConv2d(in_ch, in_ch//factor, 1, args=args )
        self.SFT_scale_conv1 = CustomConv2d(in_ch//factor, out_ch, 1, args=args)
        self.SFT_shift_conv0 = CustomConv2d(in_ch, in_ch//factor, 1, args=args)
        self.SFT_shift_conv1 = CustomConv2d(in_ch//factor, out_ch, 1, args=args)
        self.act = ActivationLayer(act_type=act,)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(self.act(self.SFT_scale_conv0(x[1])))
        shift = self.SFT_shift_conv1(self.act(self.SFT_shift_conv0(x[1])))
        return x[0] * (scale + 1) + shift


class PositionEncoding(nn.Module):
    def __init__(self, pe_embed, lfreq):
        super().__init__()
        self.pe_embed = pe_embed
        if 'pe' in pe_embed:
            lbase, levels = [float(x) for x in pe_embed.split('_')[-2:]]
            if lfreq == "pi":
                self.pe_bases = lbase ** torch.arange(int(levels)) * pi
            else:
                self.pe_bases = lbase ** torch.arange(int(levels)) * float(lfreq)
            self.embed_length = int(2 * levels)

    def forward(self, pos):
        if 'pe' in self.pe_embed:
            value_list = pos * self.pe_bases.to(pos.device)
            pe_embed = torch.cat([torch.sin(value_list), torch.cos(value_list)], dim=-1)
            return pe_embed.view(pos.size(0), -1, 1, 1)
        else:
            return pos


class Sin(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sin, self).__init__()

    def forward(self, input):
        return torch.sin(input)

def ActivationLayer(act_type):
    if act_type == 'relu':
        act_layer = nn.ReLU(True)
    elif act_type == 'leaky':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act_type == 'leaky01':
        act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == 'relu6':
        act_layer = nn.ReLU6(inplace=True)
    elif act_type == 'gelu':
        act_layer = nn.GELU()
    elif act_type == 'sin':
        act_layer = Sin()
    elif act_type == 'swish':
        act_layer = nn.SiLU(inplace=True)
    elif act_type == 'softplus':
        act_layer = nn.Softplus()
    elif act_type == 'hardswish':
        act_layer = nn.Hardswish(inplace=True)
    else:
        raise KeyError(f"Unknown activation function {act_type}.")

    return act_layer


def NormLayer(norm_type, ch_width):    
    if norm_type == 'none':
        norm_layer = nn.Identity()
    elif norm_type == 'bn':
        norm_layer = nn.BatchNorm2d(num_features=ch_width)
    elif norm_type == 'in':
        norm_layer = nn.InstanceNorm2d(num_features=ch_width)
    else:
        raise NotImplementedError

    return norm_layer


class DownConv(nn.Module):
    def __init__(self, **kargs):
        super(DownConv, self).__init__()
        ks, ngf, new_ngf, strd = kargs['ks'], kargs['ngf'], kargs['new_ngf'], kargs['strd']
        args = kargs['args']
        if kargs['conv_type'] == 'pshuffel':
            self.downconv = nn.Sequential(
                nn.PixelUnshuffle(strd) if strd !=1 else nn.Identity(),
                CustomConv2d(ngf * strd**2, new_ngf, ks, 1, ceil((ks - 1) // 2), bias=kargs['bias'], args=args)
            )
        elif kargs['conv_type'] == 'conv':
            self.downconv = CustomConv2d(ngf, new_ngf, ks+strd, strd, ceil(ks / 2), bias=kargs['bias'], args=args)
        elif kargs['conv_type'] == 'interpolate':
            self.downconv = nn.Sequential(
                nn.Upsample(scale_factor=1. / strd, mode='bilinear',),
                CustomConv2d(ngf, new_ngf, ks+strd, 1, ceil((ks + strd -1) / 2), bias=kargs['bias'], args=args)
            )
        
    def forward(self, x):
        return self.downconv(x)


class UpConv(nn.Module):
    def __init__(self, **kargs):
        super(UpConv, self).__init__()
        ks, ngf, new_ngf, strd = kargs['ks'], kargs['ngf'], kargs['new_ngf'], kargs['strd']
        args = kargs['args']
        if  kargs['conv_type']  == 'pshuffel':
            self.upconv = nn.Sequential(
                CustomConv2d(ngf, new_ngf * strd * strd, ks, 1, ceil((ks - 1) // 2), bias=kargs['bias'], args=args),
                nn.PixelShuffle(strd) if strd !=1 else nn.Identity(),
            )
        elif  kargs['conv_type']  == 'conv':
            self.upconv = nn.ConvTranspose2d(ngf, new_ngf, ks+strd, strd, ceil(ks / 2))
        elif  kargs['conv_type']  == 'interpolate':
            self.upconv = nn.Sequential(
                nn.Upsample(scale_factor=strd, mode='bilinear',),
                CustomConv2d(ngf, new_ngf, strd + ks, 1, ceil((ks + strd -1) / 2), bias=kargs['bias'], args=args)
            )
        elif kargs['conv_type']  == 'pshuffel_3x3':
            ks = 3 if ks>3 else ks
            self.upconv = nn.Sequential(
                CustomConv2d(ngf, new_ngf * strd * strd, ks, 1, ceil((ks - 1) // 2), bias=kargs['bias'], args=args),
                nn.PixelShuffle(strd) if strd !=1 else nn.Identity(),
            )
    def forward(self, x):
        return self.upconv(x)

###################################  Code for ConvNeXt   ###################################
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, stage_blocks=0, strds=[2,2,2,2], dims=[96, 192, 384, 768], 
            in_chans=3, drop_path_rate=0., layer_scale_init_value=1e-6,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stage_num = len(dims)
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, stage_blocks*self.stage_num)] 
        cur = 0
        for i in range(self.stage_num):
            # Build downsample layers
            if i > 0:
                downsample_layer = nn.Sequential(
                        LayerNorm(dims[i-1], eps=1e-6, data_format="channels_first"),
                        nn.Conv2d(dims[i-1], dims[i], kernel_size=strds[i], stride=strds[i]),
                )
            else:
                downsample_layer = nn.Sequential(
                    nn.Conv2d(in_chans, dims[0], kernel_size=strds[i], stride=strds[i]),
                    LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
                )                
            self.downsample_layers.append(downsample_layer)

            # Build more blocks
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(stage_blocks)]
            )
            self.stages.append(stage)
            cur += stage_blocks

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_list = []
        for i in range(self.stage_num):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            out_list.append(x)
        return out_list[-1]


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
