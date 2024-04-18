from .transform_ops import *
import torch.nn as nn
import torch.nn.functional as F
import math

quant_map = {
    "edgescale": EdgeScale_T,
    "scale": Scale_T,
    "scalebeta":ScaleBeta_T,
    "multiscale": MS_T,
    "log": Log_T,
    "exp": Exp_T,
    "lsq": LSQ,
    "lsqv2": LSQV2,
    "dq":DirectQuant,
}

class CustomConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, **kargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        args = kargs["args"]
        if args.quant:
            bits_w, bits_b = args.quant_model_bit, args.quant_bias_bit
            per_channel_w, per_channel_b = args.per_channel_w, args.per_channel_b
            quantizer_w, quantizer_b = args.quantizer_w, args.quantizer_b
            self.weight_quantizer = quant_map[quantizer_w](bits_w, signed=True, per_channel=per_channel_w)
            self.weight_quantizer.init_form(self.weight)
            if bias:
                self.bias_quantizer = quant_map[quantizer_b](bits_b, signed=True, per_channel=per_channel_b)
                self.bias_quantizer.init_form(self.bias)
            else:
                self.bias_quantizer = None
            self.bitrate_w_dict = {}
            self.bitrate_b_dict = {}
        self.dequant_w = None
        self.dequant_b = None
        self.quant = args.quant

    def forward(self, x):
        return F.conv2d(x, self.weight if self.dequant_w is None else self.dequant_w, self.bias if self.dequant_b is None else self.dequant_b, 
            self.stride, self.padding, self.dilation, self.groups)

class CustomLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kargs):
        super().__init__(in_features, out_features, bias=bias)
        args = kargs["args"]
        if args.quant:
            bits_w, bits_b = args.quant_model_bit, args.quant_bias_bit
            per_channel_w, per_channel_b = args.per_channel_w, args.per_channel_b
            quantizer_w, quantizer_b = args.quantizer_w, args.quantizer_b
            self.weight_quantizer = quant_map[quantizer_w](bits_w, signed=True, per_channel=per_channel_w)
            self.weight_quantizer.init_form(self.weight)
            if bias:
                self.bias_quantizer = quant_map[quantizer_b](bits_b, signed=True, per_channel=per_channel_b)
                self.bias_quantizer.init_form(self.bias)
            else:
                self.bias_quantizer = None
            self.bitrate_w_dict = {}
            self.bitrate_b_dict = {}
        self.dequant_w = None
        self.dequant_b = None
        self.quant = args.quant

    def forward(self, x):
        return F.linear(x, self.weight if self.dequant_w is None else self.dequant_w, self.bias if self.dequant_b is None else self.dequant_b)