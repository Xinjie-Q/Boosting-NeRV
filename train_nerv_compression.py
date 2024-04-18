from curses import meta
from email.policy import strict
import imageio
import argparse
import os
import random
import shutil
from datetime import datetime
import numpy as np
import csv
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from model_hnerv import HNeRV, HNeRVDecoder, HNeRV_Boost
from model_enerv import ENeRV_Boost
from model_nerv import NeRV_Boost
from hnerv_utils import *
from torch.utils.data import Subset
from copy import deepcopy
from torchvision.utils import save_image
import pandas as pd
import yaml
import constriction
from lib.entropy_model import *
from lib.quant_ops import CustomConv2d, CustomLinear

def main():
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='', help='data path for vid')
    parser.add_argument('--vid', type=str, default='k400_train0', help='video id',)
    parser.add_argument('--shuffle_data', action='store_true', help='randomly shuffle the frame idx')
    parser.add_argument('--data_split', type=str, default='1_1_1', 
        help='Valid_train/total_train/all data split, e.g., 18_19_20 means for every 20 samples, the first 19 samples is full train set, and the first 18 samples is chose currently')
    parser.add_argument('--crop_list', type=str, default='640_1280', help='video crop size',)
    parser.add_argument('--resize_list', type=str, default='-1', help='video resize size',)

    # NERV architecture parameters
    # Embedding and encoding parameters
    parser.add_argument('--model', type=str, default='', help='model name')
    parser.add_argument('--embed', type=str, default='', help='empty string for HNeRV, and base value/embed_length for NeRV position encoding')
    parser.add_argument('--ks', type=str, default='0_3_3', help='kernel size for encoder and decoder')
    parser.add_argument('--enc_blks', type=int, default=1, help='the number of encoder blocks')
    parser.add_argument('--enc_strds', type=int, nargs='+', default=[], help='stride list for encoder')
    parser.add_argument('--enc_dim', type=str, default='64_16', help='enc latent dim and embedding ratio')
    parser.add_argument('--modelsize', type=float,  default=1.5, help='model parameters size: model size + embedding parameters')
    parser.add_argument('--saturate_stages', type=int, default=-1, help='saturate stages for model size computation')

    # Decoding parameters: FC + Conv
    parser.add_argument('--lfreq', type=str, default="pi", help='out size (h,w) for mlp')
    parser.add_argument('--fc_dim', type=int, default=None, help='out size (h,w) for mlp')
    parser.add_argument('--fc_hw', type=str, default='9_16', help='out size (h,w) for mlp')
    parser.add_argument('--reduce', type=float, default=1.2, help='chanel reduction for next stage')
    parser.add_argument('--lower_width', type=int, default=32, help='lowest channel width for output feature maps')
    parser.add_argument('--dec_strds', type=int, nargs='+', default=[5, 3, 2, 2, 2], help='strides list for decoder')
    parser.add_argument('--dec_blks', type=int, nargs='+',  default=[1, 1, 1, 1, 1], help='block number for decoder')
    #parser.add_argument('--num_blks', type=str, default='1_1', help='block number for encoder and decoder')
    parser.add_argument("--conv_type", default=['convnext', 'pshuffel',], type=str, nargs="+",
        help='conv type for encoder/decoder', choices=['pshuffel', 'conv', 'convnext', 'interpolate', 'pshuffel_3x3'])
    parser.add_argument('--norm', default='none', type=str, help='norm layer for generator', choices=['none', 'bn', 'in'])
    parser.add_argument('--act', type=str, default='gelu', help='activation to use', 
        choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish', 'sin', 'ressin'])
    parser.add_argument('--sft_block', type=str, default='none', help='activation to use')
    parser.add_argument('--ch_t', type=int, default=32, help='sft in channels')
    parser.add_argument('--block_dim', type=int, default=128, help='transformer dims in the enerv model')

    # General training setups
    parser.add_argument('-j', '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('-b', '--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--start_epoch', type=int, default=-1, help='starting epoch')
    parser.add_argument('--not_resume', action='store_true', help='not resume from latest checkpoint')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='Epoch number')
    parser.add_argument('--block_params', type=str, default='1_1', help='residual blocks and percentile to save')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
    parser.add_argument('--lr_type', type=str, default='cosine_0.1_1_0.1', help='learning rate type, default=cosine')
    parser.add_argument('--loss', type=str, default='Fusion6', help='loss type, default=L2')
    parser.add_argument('--out_bias', default='tanh', type=str, help='using sigmoid/tanh/0.5 for output prediction')
    parser.add_argument('--optim_type', default='adan', type=str, help='using adan optimizer')
    parser.add_argument('--clip_max_norm', default=0., type=float, help='clip_max_norm')
    parser.add_argument('--inpanting', default='none', type=str, help='do inpanting')
    parser.add_argument('--interpolation', action='store_true', default=False, help='do interpolation')
    parser.add_argument('--embed_inter', action='store_true', default=False, help='do interpolation')

    #quantization parameters
    parser.add_argument('--quant', action='store_true', default=False, help='enable quantization')
    parser.add_argument('--quant_model_bit', type=int, default=8, help='bit length for weight quantization')
    parser.add_argument('--quant_bias_bit', type=int, default=8, help='bit length for bias quantization')
    parser.add_argument('--quant_embed_bit', type=int, default=6, help='bit length for embedding quantization')
    parser.add_argument('--per_channel_w', action='store_true', default=False, help='per channel weight')
    parser.add_argument('--per_channel_b', action='store_true', default=False, help='per channel bias')
    parser.add_argument('--per_channel_e', action='store_true', default=False, help='per channel embedding')
    parser.add_argument('--quantizer_w', type=str, default='lsq', help='quantizer weight')    
    parser.add_argument('--quantizer_b', type=str, default='lsq', help='quantizer bias')
    parser.add_argument('--quantizer_e', type=str, default='lsqv2', help='quantizer embedding')
    parser.add_argument('--embed_entropy', action='store_true', default=False, help='use entropy model for embedding')
    parser.add_argument('--target_bit', type=float, default=5, help='entropy model type')
    parser.add_argument('--quant_axis', type=int, default=0, help='quantization axis (-1 means per tensor)')
    parser.add_argument('--lambda_rate', default=0.2, type=float, help='lambda_rate')

    # evaluation parameters
    parser.add_argument('--eval_only', action='store_true', default=False, help='do evaluation only')
    parser.add_argument('--eval_freq', type=int, default=10, help='evaluation frequency,  added to suffix!!!!')
    parser.add_argument('--dump_images', action='store_true', default=False, help='dump the prediction images')
    parser.add_argument('--dump_videos', action='store_true', default=False, help='concat the prediction images into video')
    parser.add_argument('--eval_fps', action='store_true', default=False, help='fwd multiple times to test the fps ')
    parser.add_argument('--encoder_file',  default='', type=str, help='specify the embedding file')
    parser.add_argument('--dump_values', action='store_true', default=False, help='concat the prediction images into video')
    parser.add_argument('--dump_features', action='store_true', default=False, help='dump the prediction images')

    # distribute learning parameters
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
    parser.add_argument('-d', '--distributed', action='store_true', default=False, help='distributed training,  added to suffix!!!!')

    # logging, output directory, 
    parser.add_argument('--debug', action='store_true', help='defbug status, earlier for train/eval')  
    parser.add_argument('-p', '--print-freq', default=50, type=int,)
    parser.add_argument('--weight', default='None', type=str, help='pretrained weights for ininitialization')
    parser.add_argument('--overwrite', action='store_true', help='overwrite the output dir if already exists')
    parser.add_argument('--outf', default='unify', help='folder to output images and model checkpoints')
    parser.add_argument('--suffix', default='', help="suffix str for outf")


    args = parser.parse_args()
    torch.set_printoptions(precision=4) 
    if args.debug:
        args.eval_freq = 1
        args.outf = 'output/debug'
    else:
        args.outf = os.path.join('output', args.outf)

    args.enc_strd_str, args.dec_strd_str = ','.join([str(x) for x in args.enc_strds]), ','.join([str(x) for x in args.dec_strds])
    extra_str = 'Size{}_ENC_{}_{}_DEC_{}_{}_{}{}{}'.format(args.modelsize, args.conv_type[0], args.enc_strd_str, 
        args.conv_type[1], args.dec_strd_str, '' if args.norm == 'none' else f'_{args.norm}', 
        '_dist' if args.distributed else '', '_shuffle_data' if args.shuffle_data else '',)
    args.quant_str = f'quant_M{args.quant_model_bit}_E{args.quant_embed_bit}'
    embed_str = f'{args.embed}_Dim{args.enc_dim}'
    args.exp_id = exp_id = f'{args.vid}/Size{args.modelsize}'

    args.outf = os.path.join(args.outf, exp_id)
    if args.overwrite and os.path.isdir(args.outf):
        print('Will overwrite the existing output dir!')
        shutil.rmtree(args.outf)

    if not os.path.isdir(args.outf):
        os.makedirs(args.outf)

    port = hash(args.exp_id) % 20000 + 10000
    args.init_method =  f'tcp://127.0.0.1:{port}'
    print(f'init_method: {args.init_method}', flush=True)

    torch.set_printoptions(precision=2) 
    args.ngpus_per_node = torch.cuda.device_count()
    if args.distributed and args.ngpus_per_node > 1:
        mp.spawn(train, nprocs=args.ngpus_per_node, args=(args,))
    else:
        train(None, args)

def data_to_gpu(x, device):
    return x.to(device)

def train(local_rank, args):
    cudnn.benchmark = True
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    if args.distributed and args.ngpus_per_node > 1:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=args.init_method,
            world_size=args.ngpus_per_node,
            rank=local_rank,
        )
        torch.cuda.set_device(local_rank)
        assert torch.distributed.is_initialized()        
        args.batchSize = int(args.batchSize / args.ngpus_per_node)

    args.metric_names = ['pred_seen_psnr', 'pred_seen_ssim', 'pred_unseen_psnr', 'pred_unseen_ssim',
        'quant_seen_psnr', 'quant_seen_ssim', 'quant_unseen_psnr', 'quant_unseen_ssim']
    best_metric_list = [torch.tensor(0) for _ in range(len(args.metric_names))]

    # setup dataloader    
    full_dataset = VideoDataSet(args)
    sampler = torch.utils.data.distributed.DistributedSampler(full_dataset) if args.distributed else None
    full_dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=args.batchSize, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=sampler, drop_last=False, worker_init_fn=worker_init_fn)
    args.final_size = full_dataset.final_size
    args.full_data_length = len(full_dataset)
    split_num_list = [int(x) for x in args.data_split.split('_')]
    train_ind_list, args.val_ind_list = data_split(list(range(args.full_data_length)), split_num_list, args.shuffle_data, 0)
    # print("train:", train_ind_list)
    # print("val:", args.val_ind_list)
    args.dump_vis = (args.dump_images or args.dump_videos)

    #  Make sure the testing dataset is fixed for every run
    train_dataset =  Subset(full_dataset, train_ind_list)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=(train_sampler is None),
         num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, worker_init_fn=worker_init_fn)

    # Compute the parameter number
    if ('pe' in args.embed or 'le' in args.embed) and "HNeRV_Boost" not in args.model:
        embed_param = 0
        embed_dim = int(args.embed.split('_')[-1]) * 2
        fc_param = np.prod([int(x) for x in args.fc_hw.split('_')])
    else:
        total_enc_strds = np.prod(args.enc_strds)
        embed_hw = args.final_size / total_enc_strds**2
        enc_dim1, embed_ratio = [float(x) for x in args.enc_dim.split('_')]
        embed_dim = int(embed_ratio * args.modelsize * 1e6 / args.full_data_length / embed_hw) if embed_ratio < 1 else int(embed_ratio) 
        embed_param = float(embed_dim) / total_enc_strds**2 * args.final_size * args.full_data_length
        args.enc_dim = f'{int(enc_dim1)}_{embed_dim}' 
        fc_param = (np.prod(args.enc_strds) // np.prod(args.dec_strds))**2 * 9

    decoder_size = args.modelsize * 1e6 - embed_param
    ch_reduce = 1. / args.reduce
    dec_ks1, dec_ks2 = [int(x) for x in args.ks.split('_')[1:]]
    fix_ch_stages = len(args.dec_strds) if args.saturate_stages == -1 else args.saturate_stages
    a =  ch_reduce * sum([ch_reduce**(2*i) * s**2 * min((2*i + dec_ks1), dec_ks2)**2 for i,s in enumerate(args.dec_strds[:fix_ch_stages])])
    b =  embed_dim * fc_param 
    c =  args.lower_width **2 * sum([s**2 * min(2*(fix_ch_stages + i) + dec_ks1, dec_ks2)  **2 for i, s in enumerate(args.dec_strds[fix_ch_stages:])])
    if args.fc_dim is None:
        args.fc_dim = int(np.roots([a,b,c - decoder_size]).max())

    # Building model
    if args.model == "NeRV_Boost":
        expansion = 1
        model = NeRV_Boost(expansion, args=args)
        args.expansion = expansion
    elif args.model == "ENeRV_Boost":
        expansion = 3
        model = ENeRV_Boost(expansion, args=args)
        args.expansion = expansion     
    elif args.model == "HNeRV_Boost":
        model = HNeRV_Boost(args)
    elif args.model == "HNeRV":
        model = HNeRV(args)

    entropy_model = DiffEntropyModel(distribution="gaussian")

    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    with open(os.path.join(args.outf, 'args.yaml'), 'w') as f:
        f.write(args_text)
    ##### get model params and flops #####
    if local_rank in [0, None]:
        encoder_param = (sum([p.data.nelement() for p in model.encoder.parameters()]) / 1e6) 
        decoder_param = model.decoder_params()
        total_param = decoder_param + embed_param / 1e6
        args.encoder_param, args.decoder_param, args.total_param = encoder_param, decoder_param, total_param
        args.target_bpp = args.target_bit * args.total_param * 1e6 / args.final_size / args.full_data_length
        param_str = f'Encoder_{round(encoder_param, 2)}M_Decoder_{round(decoder_param, 4)}M_Total_{round(total_param, 4)}M'
        print(f'{args}\n {param_str}', flush=True)
        with open('{}/rank0.txt'.format(args.outf), 'a') as f:
            f.write(str(model) + '\n' + f'{param_str}\n')
        writer = SummaryWriter(os.path.join(args.outf, param_str, 'tensorboard'))
    else:
        writer = None


    # distrite model to gpu or parallel
    print("Use GPU: {} for training".format(local_rank))
    if args.distributed and args.ngpus_per_node > 1:
        model = torch.nn.parallel.DistributedDataParallel(model.to(local_rank), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    elif args.ngpus_per_node > 1:
        model = torch.nn.DataParallel(model)
    elif torch.cuda.is_available():
        model = model.cuda()

    if args.optim_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim_type == "Adan":
        from optimizer import Adan
        optimizer = Adan(model.parameters(), lr=args.lr)
    args.transform_func = TransformInput(args)

    # resume from args.weight
    checkpoint = None
    loc = 'cuda:{}'.format(local_rank if local_rank is not None else 0)
    if args.weight != 'None':
        print("=> loading checkpoint '{}'".format(args.weight))
        checkpoint_path = args.weight
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        orig_ckt = checkpoint['state_dict']
        #new_ckt={k.replace('blocks.0.',''):v for k,v in orig_ckt.items()} 
        if 'module' in list(orig_ckt.keys())[0] and not hasattr(model, 'module'):
            new_ckt={k.replace('module.',''):v for k,v in new_ckt.items()}
            model.load_state_dict(new_ckt, strict=False)
        elif 'module' not in list(orig_ckt.keys())[0] and hasattr(model, 'module'):
            model.module.load_state_dict(orig_ckt, strict=False)
        else:
            model.load_state_dict(orig_ckt, strict=False)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.weight, checkpoint['epoch']))        

    # resume from model_latest
    if not args.not_resume:
        checkpoint_path = os.path.join(args.outf, 'model_latest.pth')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("=> Auto resume loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            print("=> No resume checkpoint found at '{}'".format(checkpoint_path))

    if args.start_epoch < 0:
        if checkpoint is not None and not args.not_resume:
            args.start_epoch = checkpoint['epoch'] 
        args.start_epoch = max(args.start_epoch, 0)

    if args.eval_only:
        print_str = 'Evaluation ... \n {} Results for checkpoint: {}\n'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), args.weight)
        results_list, hw = evaluate(model, full_dataloader, local_rank, args, args.dump_vis, coding=True, entropy_model=entropy_model)
        print_str = f'PSNR for output {hw} for quant {args.quant_str}: '
        for i, (metric_name, best_metric_value, metric_value) in enumerate(zip(args.metric_names, best_metric_list, results_list)):
            best_metric_value = best_metric_value if best_metric_value > metric_value.max() else metric_value.max()
            cur_v = RoundTensor(best_metric_value, 4 if 'psnr' in metric_name else 4)
            print_str += f'best_{metric_name}: {cur_v} | '
            best_metric_list[i] = best_metric_value
        if local_rank in [0, None]:
            print(print_str, flush=True)
            with open('{}/eval.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n\n')        
            args.train_time, args.cur_epoch = 0, args.epochs
            Dump2CSV(args, best_metric_list, results_list, [torch.tensor(0)], 'eval.csv')
        return

    # Training
    start = datetime.now()
    
    psnr_list = []
    model.init_data()
    for epoch in range(args.start_epoch, args.epochs):
        model.train()       
        epoch_start_time = datetime.now()
        pred_psnr_list = []
        # iterate over dataloader
        device = next(model.parameters()).device
        for i, sample in enumerate(train_dataloader):
            img_data, norm_idx, img_idx = data_to_gpu(sample['img'], device), data_to_gpu(sample['norm_idx'], device), data_to_gpu(sample['idx'], device)
            if i > 10 and args.debug:
                break

            # forward and backward
            img_data, img_gt, inpaint_mask = args.transform_func(img_data, img_idx)
            if 'pe' not in args.embed or "HNeRV_Boost" in args.model:
                cur_input = img_data
            else:
                cur_input = norm_idx

            cur_epoch = (epoch + float(i) / len(train_dataloader)) / args.epochs
            lr = adjust_lr(optimizer, cur_epoch, i, args)
            model.cal_params(entropy_model)
            if args.embed_entropy:
                img_out, _, _ = model(cur_input, entropy_model=entropy_model, norm_idx=norm_idx)
                bit_embed = model.bitrate_e_dict["bitrate"] * args.full_data_length
                bpp = (model.get_bitrate_sum(name="bitrate")+bit_embed) / args.final_size
            else:
                img_out, _, _ = model(cur_input, norm_idx=norm_idx)
                bpp = model.get_bitrate_sum(name="bitrate") / args.final_size

            out_loss = loss_fn(img_out*inpaint_mask, img_gt*inpaint_mask, args.loss)    
            if bpp/args.full_data_length > args.target_bpp:
                final_loss = out_loss + args.lambda_rate * bpp
            else:
                final_loss = out_loss
            optimizer.zero_grad()
            final_loss.backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            optimizer.step()

            pred_psnr_list.append(psnr_fn_single(img_out.detach(), img_gt)) 
            if i % args.print_freq == 0 or i == len(train_dataloader) - 1:
                pred_psnr = torch.cat(pred_psnr_list).mean()
                print_str = '[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e} pred_PSNR: {}, loss:{}, bpp:{}'.format(
                    datetime.now().strftime("%Y/%m/%d %H:%M:%S"), local_rank, epoch+1, args.epochs, i+1, len(train_dataloader), lr, 
                    RoundTensor(pred_psnr, 2), RoundTensor(final_loss, 4), RoundTensor(bpp/args.full_data_length, 6))
                print(print_str, flush=True)
                if local_rank in [0, None]:
                    with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                        f.write(print_str + '\n')

        # collect numbers from other gpus
        if args.distributed and args.ngpus_per_node > 1:
            pred_psnr = all_reduce([pred_psnr.to(local_rank)])

        # ADD train_PSNR TO TENSORBOARD
        if local_rank in [0, None]:
            h, w = img_out.shape[-2:]
            writer.add_scalar(f'Train/pred_PSNR_{h}X{w}', pred_psnr, epoch+1)
            writer.add_scalar('Train/lr', lr, epoch+1)
            epoch_end_time = datetime.now()
            print("Time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}".format( (epoch_end_time - epoch_start_time).total_seconds(), \
                    (epoch_end_time - start).total_seconds() / (epoch + 1 - args.start_epoch) ))

        # evaluation
        if (epoch + 1) % args.eval_freq == 0 or (args.epochs - epoch) in [1, 3, 5]:
            results_list, hw = evaluate(model, full_dataloader, local_rank, args, 
                args.dump_vis if epoch == args.epochs - 1 else False, 
                True if epoch == args.epochs - 1 else False, entropy_model)            
            if local_rank in [0, None]:
                # ADD val_PSNR TO TENSORBOARD
                print_str = f'Eval at epoch {epoch+1} for {hw}: '
                for i, (metric_name, best_metric_value, metric_value) in enumerate(zip(args.metric_names, best_metric_list, results_list)):
                    best_metric_value = best_metric_value if best_metric_value > metric_value.max() else metric_value.max()
                    if 'psnr' in metric_name:
                        writer.add_scalar(f'Val/{metric_name}_{hw}', metric_value.max(), epoch+1)
                        writer.add_scalar(f'Val/best_{metric_name}_{hw}', best_metric_value, epoch+1)
                        if metric_name == 'pred_seen_psnr':
                            psnr_list.append(metric_value.max())
                    print_str += f'{metric_name}: {RoundTensor(metric_value, 4)} | '
                    best_metric_list[i] = best_metric_value
                print(print_str, flush=True)
                with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                    f.write(print_str + '\n')

        state_dict = model.state_dict()
        save_checkpoint = {
            'epoch': epoch+1,
            'state_dict': state_dict,
            'optimizer': optimizer.state_dict(),   
        }    
        if local_rank in [0, None]:
            torch.save(save_checkpoint, '{}/model_latest.pth'.format(args.outf))
            if (epoch + 1) % args.epochs == 0:
                args.cur_epoch = epoch + 1
                args.train_time = str(datetime.now() - start)
                Dump2CSV(args, best_metric_list, results_list, psnr_list, f'epoch{epoch+1}.csv')
                torch.save(save_checkpoint, f'{args.outf}/epoch{epoch+1}.pth')
                if best_metric_list[0]==results_list[0]:
                    torch.save(save_checkpoint, f'{args.outf}/model_best.pth')

    if local_rank in [0, None]:
        print(f"Training complete in: {str(datetime.now() - start)}")


# Writing final results in CSV file
def Dump2CSV(args, best_results_list, results_list, psnr_list, filename='results.csv'):
    result_dict = {'Vid':args.vid, 'CurEpoch':args.cur_epoch, 'Time':args.train_time, 
        'FPS':args.fps, 'Split':args.data_split, 'Embed':args.embed, 'Crop': args.crop_list,
        'Resize':args.resize_list, 'Lr_type':args.lr_type, 'LR (E-3)': args.lr*1e3, 'Batch':args.batchSize,
        'Size (M)': f'{round(args.encoder_param, 2)}_{round(args.decoder_param, 2)}_{round(args.total_param, 2)}', 
        'ModelSize': args.modelsize, 'Epoch':args.epochs, 'Loss':args.loss, 'Act':args.act, 'Norm':args.norm,
        'FC':args.fc_hw, 'Reduce':args.reduce, 'ENC_type':args.conv_type[0], 'ENC_strds':args.enc_strd_str, 'KS':args.ks,
        'enc_dim':args.enc_dim, 'DEC':args.conv_type[1], 'DEC_strds':args.dec_strd_str, 'lower_width':args.lower_width,
        'Quant':args.quant_str, 'bits/pixel':args.total_bpp, f'PSNR_list_{args.eval_freq}':','.join([RoundTensor(v, 2) for v in psnr_list]),}
    result_dict.update({f'best_{k}':RoundTensor(v, 4) for k,v in zip(args.metric_names, best_results_list)})
    result_dict.update({f'{k}':RoundTensor(v, 4) for k,v in zip(args.metric_names, results_list)})
    csv_path = os.path.join(args.outf, filename)
    print(f'results dumped to {csv_path}')
    pd.DataFrame(result_dict,index=[0]).to_csv(csv_path)


@torch.no_grad()
def evaluate(model, full_dataloader, local_rank, args, 
    dump_vis=False, coding=False, entropy_model=None):
    
    metric_list = [[] for _ in range(len(args.metric_names))]
    time_list = []
    model.eval()
    device = next(model.parameters()).device
    quant_params, trans_params = [], []
    entropy_params = []
    pred_embed, quant_embed, dequant_embed = [], [], []

    for name,m in model.named_modules():
        if type(m) in [CustomConv2d, CustomLinear]:
            code_w, quant_w, dequant_w = m.weight_quantizer(m.weight)
            m.dequant_w = dequant_w
            quant_params.extend(quant_w.int().flatten().tolist())
            for p in m.weight_quantizer.parameters():
                trans_params.extend(p.flatten().tolist())

            if m.bias is not None:
                code_b, quant_b, dequant_b = m.bias_quantizer(m.bias)
                m.dequant_b = dequant_b
                quant_params.extend(quant_b.int().flatten().tolist())
                for p in m.bias_quantizer.parameters():
                    trans_params.extend(p.flatten().tolist())

            if entropy_model is not None:
                m.bitrate_w_dict.update(entropy_model.cal_bitrate(code_w, quant_w, False))
                entropy_params.extend(m.bitrate_w_dict["mean"].flatten().tolist())
                entropy_params.extend(m.bitrate_w_dict["std"].flatten().tolist())
                if m.bias is not None:
                    m.bitrate_b_dict.update(entropy_model.cal_bitrate(code_b, quant_b, False))
                    entropy_params.extend(m.bitrate_b_dict["mean"].flatten().tolist())
                    entropy_params.extend(m.bitrate_b_dict["std"].flatten().tolist())

    if "HNeRV" in args.model:
        for p in model.embed_quantizer.parameters():
            trans_params.extend(p.flatten().tolist())

    bitrate_e_dict = {"bitrate": 0, "mean":[], "std":[], "real_bitrate":0}
    for i, sample in enumerate(full_dataloader):
        img_data, norm_idx, img_idx = data_to_gpu(sample['img'], device), data_to_gpu(sample['norm_idx'], device), data_to_gpu(sample['idx'], device)
        img_data, img_gt, inpaint_mask = args.transform_func(img_data, img_idx)
        if 'pe' not in args.embed or "HNeRV_Boost" in args.model:
            cur_input = img_data
        else:
            cur_input = norm_idx
        if "HNeRV" in args.model:
            img_embed = model.forward_encoder(cur_input)
            code_e, quant_e, dequant_e = model.forward_embed_quant(img_embed)
            pred_embed.append(img_embed)
            quant_embed.append(quant_e)
            dequant_embed.append(dequant_e)
            quant_params.extend(quant_e.int().flatten().tolist())
            if args.embed_entropy:
                model.bitrate_e_dict.update(entropy_model.cal_bitrate(code_e, quant_e, False))
                bitrate_e_dict["mean"].extend(model.bitrate_e_dict["mean"].flatten().tolist())
                bitrate_e_dict["std"].extend(model.bitrate_e_dict["std"].flatten().tolist())
                bitrate_e_dict["bitrate"] += model.bitrate_e_dict["bitrate"]
                bitrate_e_dict["real_bitrate"] += model.bitrate_e_dict["real_bitrate"]
            img_out, embed_list, dec_time = model.forward_decoder(dequant_e, norm_idx)
        else:
            img_out, embed_list, dec_time = model(cur_input)

        # collect decoding fps
        time_list.append(dec_time)
        if args.eval_fps:
            time_list.pop()
            for _ in range(100):
                _, _, dec_time = model(cur_input, embed_list[0])
                time_list.append(dec_time)

        # compute psnr and ms-ssim
        pred_psnr, pred_ssim = psnr_fn_batch([img_out], img_gt), msssim_fn_batch([img_out], img_gt)
        for metric_idx, cur_v in  enumerate([pred_psnr, pred_ssim]):
            for batch_i, cur_img_idx in enumerate(img_idx):
                metric_idx_start = 2 if cur_img_idx in args.val_ind_list else 0
                metric_list[metric_idx_start+metric_idx+4].append(cur_v[:,batch_i])

        # print eval results and add to log txt
        if i % args.print_freq == 0 or i == len(full_dataloader) - 1:
            avg_time = sum(time_list) / len(time_list)
            fps = args.batchSize / avg_time
            print_str = '[{}] Rank:{}, Eval at Step [{}/{}] , FPS {}, '.format(
                datetime.now().strftime("%Y/%m/%d %H:%M:%S"), local_rank, i+1, len(full_dataloader), round(fps, 2))
            for v_name, v_list in zip(args.metric_names, metric_list):
                cur_value = torch.stack(v_list, dim=-1).mean(-1) if len(v_list) else torch.zeros(1)
                print_str += f'{v_name}: {RoundTensor(cur_value, 4)} | '
                
            if local_rank in [0, None]:
                print(print_str, flush=True)
                with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                    f.write(print_str + '\n')
        
    # Collect results from 
    results_list = [torch.stack(v_list, dim=1).mean(1).cpu() if len(v_list) else torch.zeros(1) for v_list in metric_list]
    args.fps = fps
    h,w = img_data.shape[-2:]
    model.train()  
    if args.distributed and args.ngpus_per_node > 1:
        for cur_v in results_list:
            cur_v = all_reduce([cur_v.to(local_rank)])

    # dump quantized checkpoint, and decoder
    if local_rank in [0, None]:
        # Entropy coding
        if coding:
            total_pixels = args.final_size * args.full_data_length
            
            #Network-free Gaussian Entropy model
            trans_params_len = len(trans_params)
            estimate_bits = model.get_bitrate_sum(name="bitrate")
            data_bits = model.get_bitrate_sum(name="real_bitrate")
            meta_bits = len(entropy_params) * 32
            meta_bits += trans_params_len * 32
            if "HNeRV" in args.model:
                estimate_bits += bitrate_e_dict["bitrate"]
                data_bits += bitrate_e_dict["real_bitrate"]
                meta_bits = meta_bits + (len(bitrate_e_dict["mean"])+len(bitrate_e_dict["std"])) * 32       

            estimate_total_bits = meta_bits + estimate_bits.item()
            total_bits = data_bits + meta_bits
            args.total_bpp = total_bits / total_pixels
            args.estimate_bpp = estimate_total_bits / total_pixels
            print_str = f'Gaussian Entropy Model real bpp: {round(args.total_bpp, 6)}, estimated bpp:{round(args.estimate_bpp, 6)}, target_bpp:{round(args.target_bpp,6)} \n'          
            print(print_str, flush=True)
            with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n')
    return results_list, (h,w)


if __name__ == '__main__':
    main()
