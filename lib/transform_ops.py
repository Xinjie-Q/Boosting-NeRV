import torch
import torch.nn as nn


def grad_scale(x, scale):
    return (x - x * scale).detach() + x * scale

def ste(x):
    return (x.round() - x).detach() + x

def myabs(x):
    return torch.where(x==0, x, torch.abs(x))

def mysign(x):
    return torch.where(x == 0, torch.ones_like(x), torch.sign(x))

class LSQV2(nn.Module):
    def __init__(self, bits, signed=False, per_channel=False):
        super().__init__()
        if signed:
            self.qmin = -2**(bits - 1)
            self.qmax = 2 ** (bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** bits - 1

        self.scale = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.init = False
        self.per_channel = per_channel
        

    def init_form(self, tensor):
        if self.per_channel:   
            self.scale = nn.Parameter(torch.ones(tensor.size(1)), requires_grad=True)
            self.beta = nn.Parameter(torch.ones(tensor.size(1)), requires_grad=True)

    def init_data(self, tensor):
        if not self.init:
            device = tensor.device
            t_min, t_max = tensor.min(), tensor.max()
            scale = (t_max - t_min) / (self.qmax-self.qmin)
            self.beta.data = torch.Tensor([t_min]).to(device)
            self.scale.data = torch.Tensor([scale]).to(device)
        self.init = True

    def forward(self, x):
        grad = 1.0 / ((self.qmax * x.numel()) ** 0.5)
        scale = self.scale
        beta = self.beta
        s_scale = grad_scale(scale, grad)
        beta_scale = grad_scale(beta, grad)
        code = ((x - beta_scale) / s_scale).clamp(self.qmin, self.qmax)
        quant = ste(code)
        dequant = quant * s_scale + beta_scale
        return code, quant, dequant

class LSQ(nn.Module):
    def __init__(self, bits, signed=False, per_channel=False,):
        super().__init__()
        if signed:
            self.qmin = -2**(bits - 1)
            self.qmax = 2 ** (bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** bits - 1

        self.scale = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.per_channel = per_channel
        self.init = False

    def init_form(self, tensor):
        if self.per_channel:   
            self.scale = nn.Parameter(torch.ones(tensor.size(0)), requires_grad=True)

    def init_data(self, tensor):
        if not self.init:
            device = tensor.device
            if self.per_channel:
                if len(tensor.shape)>1:
                    t_min, t_max = tensor.min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0], tensor.max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
                    scale = (t_max - t_min) / (self.qmax-self.qmin)
                    self.scale.data = scale.to(device)
                else:
                    t_min, t_max = tensor.min(), tensor.max()
                    scale = (t_max - t_min) / (self.qmax-self.qmin)
                    self.scale.data = torch.ones(tensor.size(0)).to(device)*scale.to(device)
            else:
                t_min, t_max = tensor.min(), tensor.max()
                scale = (t_max - t_min) / (self.qmax-self.qmin)
                self.scale.data = torch.Tensor([scale]).to(device)
            self.init = True

    def forward(self, x):
        grad = 1.0 / ((self.qmax * x.numel()) ** 0.5)
        s_scale = grad_scale(self.scale, grad)
        if self.per_channel and len(x.shape)>1:
            s_scale = s_scale.view(-1, 1, 1, 1)
        code = (x / s_scale).clamp(self.qmin, self.qmax)
        quant = ste(code)
        dequant = quant * s_scale
        return code, quant, dequant


class DirectQuant(nn.Module):
    def __init__(self, bits, signed=False, per_channel=False,):
        super().__init__()
        self.init = False

    def init_form(self, tensor):
        if not self.init:
            self.init = True

    def init_data(self, tensor):
        if not self.init:
            self.init = True

    def forward(self, x):
        code = x
        quant = ste(code)
        dequant = quant
        return code, quant, dequant


class EdgeScale_T(nn.Module):
    def __init__(self, bits, signed=False, per_channel=False):
        super().__init__()
        self.scale = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.thresold = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.init = False
        self.signed = signed
        self.per_channel = per_channel
        if self.signed:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.qmin = - 2 ** (bits - 1)
            self.qmax = 2 ** (bits - 1) - 1
        else:
            # unsigned activation is quantized to [0, 2^b-1]
            self.qmin = 0
            self.qmax = 2 ** bits - 1

    def init_form(self, tensor):
        if self.per_channel:   
            self.scale = nn.Parameter(torch.ones(tensor.size(0)), requires_grad=True)
            self.thresold = nn.Parameter(torch.ones(tensor.size(0)), requires_grad=True)

    def init_data(self, tensor):
        if not self.init:
            device = tensor.device
            if self.per_channel:
                if len(tensor.shape)>1:
                    t_min, t_max = tensor.min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0], tensor.max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
                    scale = (t_max - t_min) / (self.qmax -self.qmin)
                    self.scale.data = scale.to(device)
                    self.thresold.data = scale.to(device)
                else:
                    t_min, t_max = tensor.min(), tensor.max()
                    scale = (t_max - t_min) / (self.qmax -self.qmin)
                    self.scale.data = torch.ones(tensor.size(0)).to(device)*scale.to(device)
                    self.thresold.data = torch.ones(tensor.size(0)).to(device)*scale.to(device)
            else:
                t_min, t_max = tensor.min(), tensor.max()
                scale = (t_max - t_min) / (self.qmax -self.qmin)
                self.scale.data = torch.Tensor([scale]).to(device) 
                self.thresold.data = torch.Tensor([scale]).to(device)
            self.init = True

    def encode(self, x):
        param_sign = torch.sign(x)
        if self.per_channel and len(x.shape) > 1:
            thresold = self.thresold.view(-1, 1, 1, 1)
            scale = self.scale.view(-1, 1, 1, 1)
        else:
            thresold = self.thresold
            scale = self.scale
        reserve_mask = torch.abs(x) > torch.abs(thresold)
        sparse = (x / (2 * torch.abs(thresold)))
        reserve = (param_sign * (0.5 + (torch.abs(x) - torch.abs(thresold)) / torch.abs(scale)))
        return torch.where(reserve_mask, reserve, sparse)

    def decode(self, x):
        if self.per_channel and len(x.shape) > 1:
            thresold = self.thresold.view(-1, 1, 1, 1)
            scale = self.scale.view(-1, 1, 1, 1)
        else:
            thresold = self.thresold
            scale = self.scale
        code_sign = torch.sign(x)
        reserve_mask = torch.abs(x) > 0.5
        sparse = (x * (2 * torch.abs(thresold)))
        reserve = (code_sign * (torch.abs(thresold) + (torch.abs(x) - 0.5) * torch.abs(scale)))
        return torch.where(reserve_mask, reserve, sparse) 

    def forward(self, x):
        code = self.encode(x)
        quant = (code.round() - code).detach() + code
        dequant = self.decode(quant)
        return code, quant, dequant

class Scale_T(nn.Module):
    def __init__(self, bits, signed=False, per_channel=False):
        super().__init__()
        self.scale = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.init = False
        self.signed = signed
        self.per_channel = per_channel
        if self.signed:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.qmin = - 2 ** (bits - 1)
            self.qmax = 2 ** (bits - 1) - 1
        else:
            # unsigned activation is quantized to [0, 2^b-1]
            self.qmin = 0
            self.qmax = 2 ** bits - 1
        

    def init_form(self, tensor):
        if self.per_channel:   
            self.scale = nn.Parameter(torch.ones(tensor.size(0)), requires_grad=True)

    def init_data(self, tensor):
        if not self.init:
            device = tensor.device
            if self.per_channel:
                if len(tensor.shape)>1:
                    t_min, t_max = tensor.min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0], tensor.max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
                    scale = (t_max - t_min) / (self.qmax -self.qmin)
                    self.scale.data = scale.to(device)
                else:
                    t_min, t_max = tensor.min(), tensor.max()
                    scale = (t_max - t_min) / (self.qmax -self.qmin)
                    self.scale.data = torch.ones(tensor.size(0)).to(device)*scale.to(device),
            else:
                t_min, t_max = tensor.min(), tensor.max()
                scale = (t_max - t_min) / (self.qmax -self.qmin)
                self.scale.data = torch.Tensor([scale]).to(device)
            self.init = True

    def encode(self, x):
        scale = self.scale
        return x/scale

    def decode(self, x):
        scale = self.scale
        return x*scale

    def forward(self, x):
        code = self.encode(x)
        quant = (code.round() - code).detach() + code
        dequant = self.decode(quant)
        return code, quant, dequant

class ScaleBeta_T(nn.Module):
    def __init__(self, bits, signed=False, per_channel=False):
        super().__init__()
        if signed:
            self.qmin = -2**(bits - 1)
            self.qmax = 2 ** (bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** bits - 1

        self.scale = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.init = False
        self.per_channel = per_channel

    def init_form(self, tensor):
        if self.per_channel:   
            self.scale = nn.Parameter(torch.ones(tensor.size(1)), requires_grad=True)
            self.beta = nn.Parameter(torch.ones(tensor.size(1)), requires_grad=True)

    def init_data(self, tensor):
        if not self.init:
            device = tensor.device
            t_min, t_max = tensor.min(), tensor.max()
            scale = (t_max - t_min) / (self.qmax-self.qmin)
            self.beta.data = torch.Tensor([t_min]).to(device)
            self.scale.data = torch.Tensor([scale]).to(device)
        self.init = True

    def forward(self, x):
        code = ((x - self.beta) / self.scale)
        quant = ste(code)
        dequant = quant * self.scale + self.beta
        return code, quant, dequant

class Exp_T(nn.Module):
    def __init__(self, bits, signed=False, per_channel=False):
        super().__init__()
        self.scale = nn.Parameter(torch.Tensor([1.0/64]), requires_grad=True)
        self.shift = nn.Parameter(torch.Tensor([-1.0]), requires_grad=True)
        self.inner_scale = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
   
        self.init = False
        self.signed = signed
        self.per_channel = per_channel
        if self.signed:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.qmin = - 2 ** (bits - 1)
            self.qmax = 2 ** (bits - 1) - 1
        else:
            # unsigned activation is quantized to [0, 2^b-1]
            self.qmin = 0
            self.qmax = 2 ** bits - 1

    def init_form(self, tensor):
        if not self.init:
            self.inner_scale.data = torch.Tensor([tensor.abs().max() / 0.69314718056])
            self.init = True

    def encode(self, x):
        return mysign(x) * (torch.exp(myabs(x) / self.inner_scale) + self.shift) / self.scale

    def decode(self, x):
        return mysign(x) * torch.log(myabs(x) * self.scale - self.shift) * self.inner_scale

    def forward(self, x):
        code = self.encode(x)
        quant = (code.round() - code).detach() + code
        dequant = self.decode(quant)
        return code, quant, dequant

class Log_T(nn.Module):
    def __init__(self, bits, signed=False, per_channel=False):
        super().__init__()
        self.scale = nn.Parameter(torch.Tensor([1.0/64]), requires_grad=True)
        self.shift = nn.Parameter(torch.Tensor([-1.0]), requires_grad=True)
        self.inner_scale = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
   
        self.init = False
        self.signed = signed
        self.per_channel = per_channel
        if self.signed:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.qmin = - 2 ** (bits - 1)
            self.qmax = 2 ** (bits - 1) - 1
        else:
            # unsigned activation is quantized to [0, 2^b-1]
            self.qmin = 0
            self.qmax = 2 ** bits - 1

    def init_form(self, tensor):
        if not self.init:
            self.inner_scale.data = torch.Tensor([tensor.abs().max() / 1.718281828459045])
            self.init = True

    def encode(self, x):
        return mysign(x) * torch.log(self.shift + myabs(x) / self.inner_scale) / self.scale

    def decode(self, x):
        return mysign(x) * (torch.exp(myabs(x) * self.scale) - self.shift) * self.inner_scale

    def forward(self, x):
        code = self.encode(x)
        quant = (code.round() - code).detach() + code
        dequant = self.decode(quant)
        return code, quant, dequant

class MS_T(nn.Module):
    def __init__(self, bits, signed=False, per_channel=False):
        super().__init__()
        self.scale = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

        self.init = False
        self.signed = signed
        self.per_channel = per_channel
        if self.signed:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.qmin = - 2 ** (bits - 1)
            self.qmax = 2 ** (bits - 1) - 1
        else:
            # unsigned activation is quantized to [0, 2^b-1]
            self.qmin = 0
            self.qmax = 2 ** bits - 1

    def init_form(self, tensor):
        if not self.init:
            NUM_LIN = 5
            self.scale = nn.Parameter(torch.full((NUM_LIN,), (tensor.max() - tensor.min()) / 256))
            self.param_range = (torch.arange(1, NUM_LIN, device=tensor.device, dtype=torch.float32) * (tensor.abs().max() / NUM_LIN)).detach().requires_grad_(False)
            self.init = True

    def encode(self, x):
        assert self.param_range.shape[0] + 1 == self.scale.shape[0]
        param_sign = mysign(x)
        res = torch.zeros_like(x)
        filled = torch.zeros_like(x).bool()
        base_last = 0
        range_last = 0
        for i in range(len(self.param_range)):
            mask = (myabs(x) < self.param_range[i]) & (~filled)
            res[mask] = (base_last + (myabs(x) - range_last) / myabs(self.scale[i]))[mask]
            filled = filled | mask
            base_last += ((self.param_range[i] - range_last) / myabs(self.scale[i]))
            range_last = self.param_range[i]
        res[~filled] = (base_last + (myabs(x) - range_last) / myabs(self.scale[-1]))[~filled]
        return res * param_sign

    def decode(self, x):
        assert self.param_range.shape[0] + 1 == self.scale.shape[0]
        code_sign = mysign(x)
        res = torch.zeros_like(x)
        filled = torch.zeros_like(x).bool()
        base_last = 0
        range_last = 0
        for i in range(len(self.param_range)):
            base_now = (base_last + (self.param_range[i] - range_last) / self.scale[i])
            mask = (myabs(x) < base_now) & (~filled)
            res[mask] = (range_last + (myabs(x) - base_last) * self.scale[i])[mask]
            filled = filled | mask
            base_last = base_now
            range_last = self.param_range[i]

        res[~filled] = (range_last + (myabs(x) - base_last) * self.scale[-1])[~filled]
        return res * code_sign

    def forward(self, x):
        code = self.encode(x)
        quant = (code.round() - code).detach() + code
        dequant = self.decode(quant)
        return code, quant, dequant

