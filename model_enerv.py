import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions as dist
from einops import rearrange
from model_blocks import *
import time
from lib.quant_ops import CustomConv2d, CustomLinear

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., args=None):
        super().__init__()
        self.net = nn.Sequential(
            CustomLinear(dim, hidden_dim, args=args),
            nn.GELU(),
            nn.Dropout(dropout),
            CustomLinear(hidden_dim, dim, args=args),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., args=None):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not(heads==1 and dim_head==dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = CustomLinear(dim, inner_dim * 3, bias=False, args=args)

        self.to_out = nn.Sequential(
            CustomLinear(inner_dim, dim, args=args),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0., prenorm=False, args=None):
        super(TransformerBlock, self).__init__()
        if prenorm:
            self.attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, args=args))
            self.ffn = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout, args=args))
        else:
            self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, args=args)
            self.ffn = FeedForward(dim, mlp_dim, dropout=dropout, args=args)
    def forward(self, x):
        x = self.attn(x) + x
        x = self.ffn(x) + x
        return x

class Conv_Up_Block(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        ngf = kargs['ngf']
        new_ngf = kargs['new_ngf']
        args=kargs['args']

        if ngf <= new_ngf:
            factor = 4
            self.conv1 = UpConv(ngf=ngf, new_ngf=ngf // factor, ks=kargs['ks'], strd=kargs['stride'], bias=kargs['bias'], conv_type=kargs['conv_type'], args=args)
            self.conv2 = CustomConv2d(ngf // factor, new_ngf, 3, 1, 1, bias=kargs['bias'], args=args)
        else:
            self.conv1 = CustomConv2d(ngf, new_ngf, 3, 1, 1, bias=kargs['bias'], args=args)
            self.conv2 = UpConv(ngf=new_ngf, new_ngf=new_ngf, ks=kargs['ks'], strd=kargs['stride'], bias=kargs['bias'], conv_type=kargs['conv_type'], args=args)
        self.norm = NormLayer(kargs['norm'], kargs['new_ngf'])
        self.act = ActivationLayer(kargs['act'])
        args = kargs['args']
        self.use_sft = ("sft" in args.sft_block)
        if args.sft_block == "res_sft":
            self.sft_block = ResBlock_SFT(kargs['new_ngf'], kargs['new_ngf'], cond_ch=kargs["sft_ngf"], 
                    in_act="relu", out_act="gelu", omega=1, args=args)

    def forward(self, x):
        if isinstance(x, tuple):
            embed = x[1]
            x = self.act(self.norm(self.conv2(self.conv1(x[0]))))
            x = self.sft_block((x, embed))   
            return x
        else:
            return self.act(self.norm(self.conv2(self.conv1(x))))

class ENeRV(nn.Module):
    def __init__(self, expansion=3, args=None):
        super().__init__()
        self.encoder = nn.Identity()
        # t mapping
        self.pe_t = PositionEncoding(args.embed, args.lfreq)
        self.fc_h, self.fc_w = [int(x) for x in args.fc_hw.split('_')]
        self.fc_dim = args.fc_dim

        self.block_dim = args.block_dim
        mlp_dim = args.block_dim//2
        
        mlp_dim_list = [self.pe_t.embed_length] + [self.block_dim*2] + [self.block_dim]
        self.stem_t = NeRV_MLP(dim_list=mlp_dim_list, act=args.act, args=args)
        self.pe_t_manipulate = PositionEncoding(args.embed, args.lfreq)
        self.t_branch = NeRV_MLP(dim_list=[self.pe_t_manipulate.embed_length, 128, 128], act=args.act, args=args)

        self.pe_xy = PositionEncoding(args.embed, args.lfreq)
        self.stem_xy = NeRV_MLP(dim_list=[2 * self.pe_xy.embed_length, self.block_dim], act=args.act, args=args)
        self.trans1 = TransformerBlock(
            dim=self.block_dim, heads=1, dim_head=64, mlp_dim=mlp_dim, dropout=0., prenorm=False, args=args
        )
        self.trans2 = TransformerBlock(
            dim=self.block_dim, heads=8, dim_head=64, mlp_dim=mlp_dim, dropout=0., prenorm=False, args=args
        )
        if self.block_dim == self.fc_dim:
            self.toconv = nn.Identity()
        else:
            self.toconv = NeRV_MLP(dim_list=[self.block_dim, self.fc_dim], act=args.act, args=args)
        
        # BUILD CONV LAYERS
        self.layers, self.t_layers, self.norm_layers = [nn.ModuleList() for _ in range(3)]
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
                self.t_layers.append(NeRV_MLP(dim_list=[128, 2*ngf], act=args.act, args=args))
                self.norm_layers.append(nn.InstanceNorm2d(ngf, affine=False))
            
                if i == 0:
                    self.layers.append(Conv_Up_Block(ngf=ngf, new_ngf=new_ngf, ks=min(ks_dec1+2*i, ks_dec2), stride=1 if j else stride, 
                        bias=True, norm=args.norm, act=args.act, conv_type=args.conv_type[1], sft_ngf=args.ch_t, args=args))
                else:
                    self.layers.append(NeRVBlock(dec_block=True, conv_type=args.conv_type[1], ngf=ngf, new_ngf=new_ngf, 
                        ks=min(ks_dec1+2*i, ks_dec2), strd=1 if j else stride, bias=True, norm=args.norm, act=args.act, sft_ngf=args.ch_t, args=args)
                    )
                ngf = new_ngf

        # build head classifier, upscale feature layer, upscale img layer 
        self.head_layer = CustomConv2d(ngf, 3, 1, 1, bias=True, args=args)
        self.out_bias = args.out_bias

    def fuse_t(self, x, t):
        # x: [B, C, H, W], normalized among C
        # t: [B, 2* C]
        f_dim = t.shape[-1] // 2
        gamma = t[:, :f_dim]
        beta = t[:, f_dim:]

        gamma = gamma[..., None, None]
        beta = beta[..., None, None]
        out = x * gamma + beta
        return out

    def forward(self, input, input_embed=None, norm_idx=None):
        device = next(self.parameters()).device
        xy_coord = torch.stack(torch.meshgrid(torch.arange(self.fc_h) / self.fc_h, 
            torch.arange(self.fc_w) / self.fc_w), dim=0).flatten(1, 2).to(device)  # [2, h*w]

        dec_start = time.time()
        batchsize = input.size(0)
        t = input[:,None].float()
        t_emb = self.stem_t(self.pe_t(t)).view(batchsize,  -1) # [B, L, 1, 1]
        t_manipulate = self.t_branch(self.pe_t_manipulate(t)) # [B, 128, 1, 1]

        x_coord = self.pe_xy(xy_coord[0][:, None])    # [h*w, C, 1, 1]
        y_coord = self.pe_xy(xy_coord[1][:, None])    # [h*w, C, 1, 1]
        xy_emb = torch.cat([x_coord, y_coord], dim=1) #[h*w, 2C, 1, 1]
        xy_emb = self.stem_xy(xy_emb).view(1, int(self.fc_h*self.fc_w), -1).expand(batchsize, -1, -1)  # [B, h*w, L]

        xy_emb = self.trans1(xy_emb)
        # fuse t into xy map
        t_emb_list = [t_emb for i in range(xy_emb.shape[1])]
        t_emb_map = torch.stack(t_emb_list, dim=1)  # [B, h*w, L]
        emb = xy_emb * t_emb_map
        emb = self.trans2(emb) 
        emb = emb.reshape(emb.shape[0], self.fc_h, self.fc_w, emb.shape[-1])
        emb = emb.permute(0, 3, 1, 2)
        emb = self.toconv(emb) #[B, fc_dim, h, w]
        output = emb

        out_list = []
        for layer, t_layer, norm_layer in zip(self.layers, self.t_layers, self.norm_layers):
            # t_manipulate
            output = norm_layer(output)
            t_feat = t_layer(t_manipulate).view(batchsize, -1)
            output = self.fuse_t(output, t_feat)
            # conv
            output = layer(output)
            out_list.append(output)
            
        img_out = self.head_layer(output)
        # normalize the final output iwth sigmoid or tanh function
        img_out = OutImg(img_out, self.out_bias) #torch.sigmoid(img_out) if self.sigmoid else (torch.tanh(img_out) + 1) * 0.5
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

class ENeRV_Boost(ENeRV):
    def __init__(self, expansion=3, args=None):
        super().__init__(expansion, args)
        self.t_branch = NeRV_MLP(dim_list=[self.pe_t_manipulate.embed_length, args.ch_t * 2, args.ch_t], act=args.act, args=args)
        self.t_layers, self.norm_layers = None, None
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
                if i == 0:
                    self.layers.append(Conv_Up_Block(ngf=ngf, new_ngf=new_ngf, ks=min(ks_dec1+2*i, ks_dec2), stride=1 if j else stride, 
                        bias=True, norm=args.norm, act=args.act, conv_type=args.conv_type[1], sft_ngf=args.ch_t, args=args))
                else:
                    self.layers.append(NeRVBlock(dec_block=True, conv_type=args.conv_type[1], ngf=ngf, new_ngf=new_ngf, 
                        ks=min(ks_dec1+2*i, ks_dec2), strd=1 if j else stride, bias=True, norm=args.norm, act=args.act, sft_ngf=args.ch_t, args=args)
                    )
                ngf = new_ngf

            
    def forward(self, input, input_embed=None, norm_idx=False):
        device = next(self.parameters()).device
        xy_coord = torch.stack(torch.meshgrid(torch.arange(self.fc_h) / self.fc_h, 
            torch.arange(self.fc_w) / self.fc_w), dim=0).flatten(1, 2).to(device)  # [2, h*w]

        dec_start = time.time()
        batchsize = input.size(0)
        t = input[:,None].float()
        t_emb = self.stem_t(self.pe_t(t)).view(batchsize,  -1) # [B, L, 1, 1]
        t_manipulate = self.t_branch(self.pe_t_manipulate(t)) # [B, ch_t, 1, 1]

        x_coord = self.pe_xy(xy_coord[0][:, None])    # [h*w, C, 1, 1]
        y_coord = self.pe_xy(xy_coord[1][:, None])    # [h*w, C, 1, 1]
        xy_emb = torch.cat([x_coord, y_coord], dim=1) #[h*w, 2C, 1, 1]
        xy_emb = self.stem_xy(xy_emb).view(1, int(self.fc_h*self.fc_w), -1).expand(batchsize, -1, -1)  # [B, h*w, L]

        xy_emb = self.trans1(xy_emb)
        # fuse t into xy map
        t_emb_list = [t_emb for i in range(xy_emb.shape[1])]
        t_emb_map = torch.stack(t_emb_list, dim=1)  # [B, h*w, L]
        emb = xy_emb * t_emb_map
        emb = self.trans2(emb) 
        emb = emb.reshape(emb.shape[0], self.fc_h, self.fc_w, emb.shape[-1])
        emb = emb.permute(0, 3, 1, 2)
        emb = self.toconv(emb) #[B, fc_dim, h, w]
        output = emb

        out_list = [t_manipulate]
        for layer in self.layers:
            output = layer((output, t_manipulate))
            out_list.append(output)
            
        img_out = self.head_layer(output)
        # normalize the final output iwth sigmoid or tanh function
        img_out = OutImg(img_out, self.out_bias) #torch.sigmoid(img_out) if self.sigmoid else (torch.tanh(img_out) + 1) * 0.5
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dec_time = time.time() - dec_start
        return img_out, out_list, dec_time

