import math
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim
from torchvision.transforms.functional import center_crop, resize
from torchvision import transforms
from PIL import Image
from torch.nn.functional import interpolate
from torch.utils.data import Dataset
import os
################# Dataset ##################
class VideoDataSet(Dataset):
    def __init__(self, args):

        self.samples = [os.path.join(args.data_path, x) for x in sorted(os.listdir(args.data_path))]
        if args.interpolation:
            if len(self.samples) % 2 == 0:
                self.samples.pop()
        self.transform = transforms.ToTensor()
        self.crop_h, self.crop_w = [int(x) for x in args.crop_list.split('_')[:2]]
        first_frame = Image.open(self.samples[0]).convert("RGB")
        h, w = first_frame.height, first_frame.width
        if h>=self.crop_h and w>=self.crop_w: 
            first_frame = self.transform(center_crop(first_frame, (self.crop_h, self.crop_w)))
            self.crop = True
        else:
            first_frame = self.transform(interpolate(first_frame, (self.crop_h, self.crop_w), 'bicubic'))
            self.crop = False
        self.final_size = first_frame.size(-2) * first_frame.size(-1)
        self.embed_inter = args.embed_inter and args.interpolation

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.crop:
            img = center_crop(img, (self.crop_h, self.crop_w))
        else:
            img = interpolate(img, (self.crop_h, self.crop_w), 'bicubic')
        img = self.transform(img)
        norm_idx =  float(idx+1) / len(self.samples)
        if self.embed_inter:
            if idx %2 == 0:
                pre_img, post_img = img, img
            else:
                pre_img = self.transform(center_crop(Image.open(self.samples[idx-1]).convert("RGB"), (self.crop_h, self.crop_w)))
                post_img = self.transform(center_crop(Image.open(self.samples[idx+1]).convert("RGB"), (self.crop_h, self.crop_w)))
            return {'img': img, 'idx': idx, 'norm_idx': norm_idx, 'pre_img':pre_img, 'post_img':post_img}
        else:
            return {'img': img, 'idx': idx, 'norm_idx': norm_idx}

###################################  Tranform input for inpainting   ###################################
class TransformInput(nn.Module):
    def __init__(self, args):
        super(TransformInput, self).__init__()
        self.inpanting = args.inpanting
        if 'inpanting_fixed' in self.inpanting:
            self.inpaint_size = int(self.inpanting.split('_')[-1]) // 2

    def forward(self, img, idx):
        inpaint_mask = torch.ones_like(img)
        if 'inpanting' in self.inpanting:
            gt = img.clone()
            h,w = img.shape[-2:]
            inpaint_mask = torch.ones((h,w)).to(img.device)
            if 'center' in self.inpanting:
                inpaint_h, inpaint_w = h//8, w//8
                ctr_x, ctr_y = int(0.5 * h), int(0.5 * w)
                inpaint_mask[ctr_x - inpaint_h: ctr_x + inpaint_h, ctr_y - inpaint_w: ctr_y + inpaint_w] = 0
            elif 'fixed' in self.inpanting: #fixed
                for ctr_x, ctr_y in [(1/2, 1/2), (1/4, 1/4), (1/4, 3/4), (3/4, 1/4), (3/4, 3/4)]:
                    ctr_x, ctr_y = int(ctr_x * h), int(ctr_y * w)
                    inpaint_mask[ctr_x - self.inpaint_size: ctr_x + self.inpaint_size, ctr_y - self.inpaint_size: ctr_y + self.inpaint_size] = 0
            input = (img * inpaint_mask).clamp(min=0,max=1)
        else:
            input, gt = img, img

        return input, gt, inpaint_mask.detach()

################## split one video into seen/unseen frames ##################
def data_split(img_list, split_num_list, shuffle_data, rand_num=0):
    valid_train_length, total_train_length, total_data_length = split_num_list
    # assert total_train_length < total_data_length
    temp_train_list, temp_val_list = [], []
    if shuffle_data:
        random.Random(rand_num).shuffle(img_list)
    for cur_i, frame_id in enumerate(img_list):
        if (cur_i % total_data_length) < valid_train_length:
            temp_train_list.append(frame_id)
        elif (cur_i % total_data_length) >= total_train_length:
            temp_val_list.append(frame_id)
    return temp_train_list, temp_val_list

################# Tensor quantization and dequantization #################
def quant_tensor(t, bits=8):
    tmin_scale_list = []
    # quantize over the whole tensor, or along each dimenstion
    t_min, t_max = t.min(), t.max()
    scale = (t_max - t_min) / (2**bits-1)
    tmin_scale_list.append([t_min, scale])
    for axis in range(t.dim()):
        t_min, t_max = t.min(axis, keepdim=True)[0], t.max(axis, keepdim=True)[0]
        if t_min.nelement() / t.nelement() < 0.02:
            scale = (t_max - t_min) / (2**bits-1)
            # tmin_scale_list.append([t_min, scale]) 
            tmin_scale_list.append([t_min.to(torch.float16), scale.to(torch.float16)]) 
    # import pdb; pdb.set_trace; from IPython import embed; embed() 
     
    quant_t_list, new_t_list, err_t_list = [], [], []
    for t_min, scale in tmin_scale_list:
        t_min, scale = t_min.expand_as(t), scale.expand_as(t)
        quant_t = ((t - t_min) / (scale)).round().clamp(0, 2**bits-1)
        new_t = t_min + scale * quant_t
        err_t = (t - new_t).abs().mean()
        quant_t_list.append(quant_t)
        new_t_list.append(new_t)
        err_t_list.append(err_t)   

    # choose the best quantization 
    best_err_t = min(err_t_list)
    best_quant_idx = err_t_list.index(best_err_t)
    best_new_t = new_t_list[best_quant_idx]
    best_quant_t = quant_t_list[best_quant_idx].to(torch.uint8)
    best_tmin = tmin_scale_list[best_quant_idx][0]
    best_scale = tmin_scale_list[best_quant_idx][1]
    quant_t = {'quant': best_quant_t, 'min': best_tmin, 'scale': best_scale}

    return quant_t, best_new_t             

def quantize_per_tensor(t, bits=8, axis=-1):
    if axis == -1:
        t_valid = t!=0
        t_min, t_max =  t[t_valid].min(), t[t_valid].max()
        scale = (t_max - t_min) / (2**bits-1)
    elif axis == 0:
        min_max_list = []
        for i in range(t.size(0)):
            t_valid = t[i]!=0
            if t_valid.sum():
                min_max_list.append([t[i][t_valid].min(), t[i][t_valid].max()])
            else:
                min_max_list.append([0, 0])
        min_max_tf = torch.tensor(min_max_list).to(t.device)        
        scale = (min_max_tf[:,1] - min_max_tf[:,0]) / (2**bits-1)
        if t.dim() == 4:
            scale = scale[:,None,None,None]
            t_min = min_max_tf[:,0,None,None,None]
        elif t.dim() == 2:
            scale = scale[:,None]
            t_min = min_max_tf[:,0,None]
        else:
            t_min = min_max_tf[:,0]

    elif axis == 1:
        min_max_list = []
        for i in range(t.size(1)):
            t_valid = t[:,i]!=0
            if t_valid.sum():
                min_max_list.append([t[:,i][t_valid].min(), t[:,i][t_valid].max()])
            else:
                min_max_list.append([0, 0])
        min_max_tf = torch.tensor(min_max_list).to(t.device)             
        scale = (min_max_tf[:,1] - min_max_tf[:,0]) / (2**bits-1)
        if t.dim() == 4:
            scale = scale[None,:,None,None]
            t_min = min_max_tf[None,:,0,None,None]
        elif t.dim() == 2:
            scale = scale[None,:]
            t_min = min_max_tf[None,:,0]            
    # import pdb; pdb.set_trace; from IPython import embed; embed()  
    t_min, scale = t_min.to(torch.float16), scale.to(torch.float16)     
    quant_t = ((t - t_min) / scale).round()
    new_t = t_min + scale * quant_t
    #quant_t_dict = {'quant': quant_t, 'min': t_min, 'scale': scale}
    return quant_t, new_t, t_min, scale
     


def dequant_tensor(quant_t):
    quant_t, tmin, scale = quant_t['quant'], quant_t['min'], quant_t['scale']
    new_t = tmin.expand_as(quant_t) + scale.expand_as(quant_t) * quant_t
    return new_t

################# Function used in distributed training #################
def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


def all_reduce(tensors, average=True):
    """
    All reduce the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all reduce across all processes in
        all machines.
        average (bool): scales the reduced tensor by the number of overall
        processes across all machines.
    """

    for tensor in tensors:
        dist.all_reduce(tensor, async_op=False)
    if average:
        world_size = dist.get_world_size()
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()
    
def reduce_dict(input_dict, average=True):
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []

        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def worker_init_fn(worker_id):
    """
    Re-seed each worker process to preserve reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return


def RoundTensor(x, num=2, group_str=False):
    if group_str:
        str_list = []
        for i in range(x.size(0)):
            x_row =  [str(round(ele, num)) for ele in x[i].tolist()]
            str_list.append(','.join(x_row))
        out_str = '/'.join(str_list)
    else:
        str_list = [str(round(ele, num)) for ele in x.flatten().tolist()]
        out_str = ','.join(str_list)
    return out_str


def adjust_lr(optimizer, cur_epoch, cur_iter, args):
    # cur_epoch = (cur_epoch + cur_iter) / args.epochs
    if 'hybrid' in args.lr_type:
        up_ratio, up_pow, down_pow, min_lr, final_lr = [float(x) for x in args.lr_type.split('_')[1:]]
        if cur_epoch < up_ratio:
            lr_mult = min_lr + (1. - min_lr) * (cur_epoch / up_ratio)** up_pow
        else:
            lr_mult = 1 - (1 - final_lr) * ((cur_epoch - up_ratio) / (1. - up_ratio))**down_pow
    elif 'cosine' in args.lr_type:
        up_ratio, up_pow, min_lr = [float(x) for x in args.lr_type.split('_')[1:]]
        if cur_epoch < up_ratio:
            lr_mult = min_lr + (1. - min_lr) * (cur_epoch / up_ratio)** up_pow
        else:
            lr_mult = 0.5 * (math.cos(math.pi * (cur_epoch - up_ratio)/ (1 - up_ratio)) + 1.0)
    elif 'enerv_sch' in args.lr_type:
        all_iter = args.epochs * args.full_data_length
        now_iter = cur_epoch * args.full_data_length + cur_iter
        if now_iter < all_iter * 0.2:
            lr_mult = 0.1 + 0.9 * now_iter / (all_iter * 0.2)
        else:
            whole = all_iter - all_iter * 0.2
            cur = now_iter - all_iter * 0.2
            lr_mult = 0.5 * (math.cos(math.pi * cur / whole) + 1.0)

    else:
        raise NotImplementedError

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = args.lr * lr_mult

    return args.lr * lr_mult


############################ Function for loss compuation and evaluate metrics ############################

def psnr2(img1, img2):
    mse = (img1 - img2) ** 2
    PIXEL_MAX = 1
    psnr = -10 * torch.log10(mse)
    psnr = torch.clamp(psnr, min=0, max=50)
    return psnr


def loss_fn(pred, target, loss_type='L2', batch_average=True):
    target = target.detach()

    if loss_type == 'L2':
        loss = F.mse_loss(pred, target, reduction='none').flatten(1).mean(1)
    elif loss_type == 'L1':
        loss = F.l1_loss(pred, target, reduction='none').flatten(1).mean(1)
    elif loss_type == 'SSIM':
        loss = 1 - ssim(pred, target, data_range=1, size_average=False)
    elif loss_type == 'Fusion1':
        loss = 0.3 * F.mse_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.7 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion2':
        loss = 0.3 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.7 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion3':
        loss = 0.5 * F.mse_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.5 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion4':
        loss = 0.5 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.5 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion5':
        loss = 0.7 * F.mse_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion6':
        loss = 0.7 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion7':
        loss = 0.7 * F.mse_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.3 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1)
    elif loss_type == 'Fusion8':
        loss = 0.5 * F.mse_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.5 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1)
    elif loss_type == 'Fusion9':
        loss = 0.9 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.1 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion10':
        loss = 0.7 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.3 * (1 - ms_ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion11':
        loss = 0.9 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.1 * (1 - ms_ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion12':
        loss = 0.8 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.2 * (1 - ms_ssim(pred, target, data_range=1, size_average=False))

    elif loss_type == 'Fusion10_freq':
        loss = 0.7 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.3 * (1 - ms_ssim(pred, target, data_range=1, size_average=False))
        pred_freq = torch.fft.fft2(pred, dim=(-2, -1))
        pred_freq = torch.stack([pred_freq.real, pred_freq.imag], -1)
        target_freq = torch.fft.fft2(target, dim=(-2, -1))
        target_freq = torch.stack([target_freq.real, target_freq.imag], -1)
        freq_loss = F.l1_loss(pred_freq, target_freq, reduction='none').flatten(1).mean(1)
        loss = 60 * loss + freq_loss

    elif loss_type == 'L1_freq':
        loss = F.l1_loss(pred, target, reduction='none').flatten(1).mean(1)
        pred_freq = torch.fft.fft2(pred, dim=(-2, -1))
        pred_freq = torch.stack([pred_freq.real, pred_freq.imag], -1)
        target_freq = torch.fft.fft2(target, dim=(-2, -1))
        target_freq = torch.stack([target_freq.real, target_freq.imag], -1)
        freq_loss = F.l1_loss(pred_freq, target_freq, reduction='none').flatten(1).mean(1)
        loss = 60 * loss + freq_loss

    elif loss_type == 'L1_ssim_freq':
        l1_loss = F.l1_loss(pred, target, reduction='none').flatten(1).mean(1)
        pred_freq = torch.fft.fft2(pred, dim=(-2, -1))
        pred_freq = torch.stack([pred_freq.real, pred_freq.imag], -1)
        target_freq = torch.fft.fft2(target, dim=(-2, -1))
        target_freq = torch.stack([target_freq.real, target_freq.imag], -1)
        freq_loss = F.l1_loss(pred_freq, target_freq, reduction='none').flatten(1).mean(1)
        ssim_loss = 1 - ssim(pred, target, data_range=1, size_average=False)
        loss = 60 * (0.7*l1_loss+0.3*ssim_loss) + freq_loss

    return loss.mean() if batch_average else loss


def psnr_fn_single(output, gt):
    l2_loss = F.mse_loss(output.detach(), gt.detach(),  reduction='none')
    psnr = -10 * torch.log10(l2_loss.flatten(start_dim=1).mean(1) + 1e-9)
    return psnr.cpu()

def psnr_fn_batch(output_list, gt):
    psnr_list = [psnr_fn_single(output.detach(), gt.detach()) for output in output_list]
    return torch.stack(psnr_list, 0).cpu()


def msssim_fn_single(output, gt):
    msssim = ms_ssim(output.float().detach(), gt.detach(), data_range=1, size_average=False)
    return msssim.cpu()

def msssim_fn_batch(output_list, gt):
    msssim_list = [msssim_fn_single(output.detach(), gt.detach()) for output in output_list]
    # for output in output_list:
    #     msssim = ms_ssim(output.float().detach(), gt.detach(), data_range=1, size_average=False)
    #     msssim_list.append(msssim)
    return torch.stack(msssim_list, 0).cpu()


def psnr_fn(output_list, target_list):
    psnr_list = []
    for output, target in zip(output_list, target_list):
        l2_loss = F.mse_loss(output.detach(), target.detach(), reduction='mean')
        psnr = -10 * torch.log10(l2_loss + 1e-9)
        psnr = psnr.view(1, 1).expand(output.size(0), -1)
        psnr_list.append(psnr)
    psnr = torch.cat(psnr_list, dim=1) #(batchsize, num_stage)
    return psnr


def msssim_fn(output_list, target_list):
    msssim_list = []
    for output, target in zip(output_list, target_list):
        if output.size(-2) >= 160:
            msssim = ms_ssim(output.float().detach(), target.detach(), data_range=1, size_average=True)
        else:
            msssim = torch.tensor(0).to(output.device)
        msssim_list.append(msssim.view(1))
    msssim = torch.cat(msssim_list, dim=0) #(num_stage)
    msssim = msssim.view(1, -1).expand(output_list[-1].size(0), -1) #(batchsize, num_stage)
    return msssim



def eval_quantize_per_tensor(t, bit=8):
    tmin_scale_list = []
    # quantize on the full tensor
    tmin, t_max = t.min().expand_as(t), t.max().expand_as(t)
    scale = (t_max - t_min) / 2**bit
    tmin_scale_list.append([t_min, scale])

    # quantize on axis 0
    min_max_list = []
    for i in range(t.size(0)):
        t_valid = t[i]!=0
        if t_valid.sum():
            min_max_list.append([t[i][t_valid].min(), t[i][t_valid].max()])
        else:
            min_max_list.append([0, 0])
    min_max_tf = torch.tensor(min_max_list).to(t.device)        
    scale = (min_max_tf[:,1] - min_max_tf[:,0]) / 2**bit
    if t.dim() == 4:
        scale = scale[:,None,None,None]
        t_min = min_max_tf[:,0,None,None,None]
    elif t.dim() == 2:
        scale = scale[:,None]
        t_min = min_max_tf[:,0,None]
    tmin_scale_list.append([t_min, scale])

    # quantize on axis 1
    min_max_list = []
    for i in range(t.size(1)):
        t_valid = t[:,i]!=0
        if t_valid.sum():
            min_max_list.append([t[:,i][t_valid].min(), t[:,i][t_valid].max()])
        else:
            min_max_list.append([0, 0])
    min_max_tf = torch.tensor(min_max_list).to(t.device)             
    scale = (min_max_tf[:,1] - min_max_tf[:,0]) / 2**bit
    if t.dim() == 4:
        scale = scale[None,:,None,None]
        t_min = min_max_tf[None,:,0,None,None]
    elif t.dim() == 2:
        scale = scale[None,:]
        t_min = min_max_tf[None,:,0]    
    tmin_scale_list.append([t_min, scale])

    # import pdb; pdb.set_trace; from IPython import embed; embed()  
    quant_t_list, new_t_list, err_t_list = [], [], []
    for tmin, scale in tmin_scale_list:
        quant_t = ((t - tmin) / (scale + 1e-19)).round()
        new_t = tmin + scale * quant_t
        quant_t_list.append(quant_t)
        new_t_list.append(new_t)
        err_t_list.append((t - new_t).abs().mean())   

    # choose the best quantization way
    best_err_t = min(err_t_list)
    best_quant_idx = err_t_list.index(best_err_t)
    best_quant_t = quant_t_list[best_quant_idx]
    best_new_t = new_t_list[best_quant_idx]

    return best_quant_t, best_new_t             

