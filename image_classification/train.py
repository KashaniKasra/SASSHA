# Acknowledgement: This code is based on https://github.com/pytorch/examples/blob/main/imagenet/main.py

import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import math
import sys
import re

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Subset
import contextlib
from utils import *

# [NEW]
import json
import csv
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.func import functional_call

# load additional models
from models import get_model

# load learning rate scheduler
from LRScheduler import CosineAnnealingWarmupRestarts

# load optimizers
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('optimizers'))))
from optimizers import get_optimizer

# load hessain power scheduler
from optimizers.hessian_scheduler import ConstantScheduler, ProportionScheduler, LinearScheduler, CosineScheduler

# import wandb
wandb_log = False

# for sam
from bypass_bn import enable_running_stats, disable_running_stats

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# additional models
model_names.append('vit_s_32')
model_names.append('vit_s_16')
model_names.append('resnet32')
model_names.append('resnet20')
model_names.append('wideresnet_28_10')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', type=str, default='imagenet',
                    help='choose dataset imagenet, cifar10/100')
parser.add_argument('path', metavar='DIR', nargs='?', default='your imagenet folder path',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument('--optimizer', type=str, default='sassha',help='choose optim')
parser.add_argument("--offline", action="store_true", default=False, help="wandb offline")


# Learning rate scheduler
parser.add_argument('--LRScheduler', type=str, default='multi_step', help="choose LRScheduler 1. 'multi_step', 2. 'cosine', 3. 'plateau' ")
parser.add_argument('--lr-decay-epoch', type=int, nargs='+', default=[80, 120],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='learning rate ratio')
parser.add_argument("--warmup_epochs", default=8, type=int, help="number of epochs for the warm up step")
parser.add_argument("--grad_clip_norm", default=0.0, type=float, help="gradient clipping for AdamW")
parser.add_argument('--min_lr', type=float, default=0.0, help="the minimum value of learning rate")

# Second-order optimization settings
parser.add_argument("--n_samples", default=1, type=int, help="the number of sampling")
parser.add_argument('--betas', type=float, nargs='*', default=[0.9, 0.999], help='betas')
parser.add_argument("--eps", default=1e-4, type=float, help="add a small number for stability")
parser.add_argument("--lazy_hessian", default=10, type=int, help="Delayed hessian update.")
parser.add_argument("--clip_threshold", default=0.05, type=float, help="Clipping threshold.")

# Hessian power scheduler
parser.add_argument('--hessian_power_scheduler', type=str, default='constant', help="choose Hessian power scheduler 1. 'constant', 2. 'proportion', 3. 'linear', 4. 'cosine'")
parser.add_argument('--max_hessian_power', type=float, default=1)
parser.add_argument('--min_hessian_power', type=float, default=0.5)

# Sharpness minimization settings
parser.add_argument("--rho", default=0.05, type=float, help="Rho parameter for SAM.")
parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
parser.add_argument('--project_name', type=str, default='project_name', help="project_name")

# [NEW]
parser.add_argument('--out-dir', type=str, default='runs', help='output directory for logs/plots')
parser.add_argument('--run-tag', type=str, default='', help='optional tag to add to run folder name')
parser.add_argument('--log-steps', action='store_true', help='log step-level train/test metrics (batch-wise)')
parser.add_argument("--hvp_every", default=1, type=int, help="Compute HVP correction every k steps (k>=1).")
parser.add_argument("--hvp_mode", default="skip", type=str, choices=["skip", "reuse"], help="On non-HVP steps: skip or reuse cached projector term.")

best_acc1 = 0

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
        
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = get_model(args)

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # [NEW if branch]
        if args.optimizer in ['samsgd', 'samadamw']:
            model = model.to(device)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features)
                model.cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()


    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_loader, val_loader, train_sampler, val_sampler = getData(
            name=args.dataset,
            path=args.path,
            train_bs=args.batch_size,
            test_bs=args.batch_size,
            num_workers=args.workers,
            distributed=args.distributed)
    
    # [NEW]
    is_main = _is_main_process(args, ngpus_per_node)
    run_name_parts = [
        args.dataset, args.arch, args.optimizer,
        f"lr{args.lr}", f"wd{args.weight_decay}", f"rho{args.rho}", f"seed{args.seed}"
    ]
    if args.run_tag:
        run_name_parts.append(args.run_tag)
    run_name_parts.append(datetime.now().strftime("%Y%m%d-%H%M%S"))
    run_name = "_".join(run_name_parts)
    run_dir = os.path.join(args.out_dir, run_name)
    if is_main:
        _ensure_dir(run_dir)
        _save_json(os.path.join(run_dir, "args.json"), vars(args))
    args.train_steps_per_epoch = len(train_loader)
    
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    
    # get an optimizer
    optimizer, create_graph, two_steps = get_optimizer(model, args)
    
    # select a learning rate scheduler
    if args.LRScheduler == 'multi_step':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            args.lr_decay_epoch,
            gamma=args.lr_decay,
            last_epoch=-1)
            
    elif args.LRScheduler == 'cosine':
        num_images = 1281167  # for imagenet
        num_miniB = (num_images // (args.batch_size*ngpus_per_node)) + 1
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=(num_miniB)*args.epochs, cycle_mult=1.0, max_lr=args.lr, min_lr=0.0, warmup_steps=num_miniB*args.warmup_epochs)
    
    elif args.LRScheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    # select a hessian power scheduler
    if args.optimizer == 'sassha':
        if args.hessian_power_scheduler == 'constant':
            hessian_power_scheduler = ConstantScheduler(
                T_max=args.epochs*len(train_loader), 
                max_value=0.5,
                min_value=0.5)
        
        elif args.hessian_power_scheduler == 'proportion':
            hessian_power_scheduler = ProportionScheduler(
                pytorch_lr_scheduler=scheduler,
                max_lr=args.lr,
                min_lr=args.min_lr,
                max_value=args.max_hessian_power,
                min_value=args.min_hessian_power)
        
        elif args.hessian_power_scheduler == 'linear':
            hessian_power_scheduler = LinearScheduler(
                T_max=args.epochs*len(train_loader), 
                max_value=args.max_hessian_power,
                min_value=args.min_hessian_power)
        
        elif args.hessian_power_scheduler == 'cosine':
            hessian_power_scheduler = CosineScheduler(
                T_max=args.epochs*len(train_loader), 
                max_value=args.max_hessian_power,
                min_value=args.min_hessian_power)
        
        optimizer.hessian_power_scheduler = hessian_power_scheduler

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            if args.optimizer == 'sassha':
                for p in optimizer.get_params():
                    optimizer.state[p]["hessian step"] = 0
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    # import and init wandb
    if wandb_log:
        import wandb
        os.environ["WANDB__SERVICE_WAIT"] = "300"

        wandb_project = args.project_name
        wandb_run_name = f'{args.optimizer}-{args.arch}-{args.lr}-{args.weight_decay}-{args.rho}'
        wandb.init(project=wandb_project, name=wandb_run_name)
        wandb.config.update(args)

    if args.LRScheduler == 'cosine':
        scheduler.step()

    # Training loop
    # scheduler.step()  # [CHANGED]
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        # [CHANGED]
        train_acc, train_loss, tr_step_acc, tr_step_loss = train(train_loader, model, criterion, optimizer, epoch, device, args, scheduler, create_graph, two_steps)

        # evaluate on validation set
        # [CHANGED]
        args.current_epoch_for_logging = epoch  # [NEW]
        val_acc, val_loss, va_step_acc, va_step_loss = validate(val_loader, model, criterion, args)

        if wandb_log:
            wandb.log({
                "train/acc": train_acc,
                "train/loss": train_loss,
                "val/acc": val_acc,
                "val/loss": val_loss,
                'lr': optimizer.param_groups[0]['lr'],
                "hessian_power": optimizer.hessian_power_t if args.optimizer == 'sassha' else 0,
            }, step=epoch)

        # [NEW]
        if is_main:
            row = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "test_loss": float(val_loss),
                "test_acc": float(val_acc),
                "lr": float(optimizer.param_groups[0]["lr"])
            }

            csv_path = os.path.join(run_dir, "metrics_epoch.csv")
            _append_csv(csv_path, fieldnames=list(row.keys()), row_dict=row)

            hist_path = os.path.join(run_dir, "metrics_epoch.json")
            if os.path.isfile(hist_path):
                with open(hist_path, "r") as f:
                    hist = json.load(f)
            else:
                hist = []
            hist.append(row)
            _save_json(hist_path, hist)

            steps_npz = os.path.join(run_dir, "metrics_steps.npz")
            if os.path.isfile(steps_npz):
                old = np.load(steps_npz, allow_pickle=True)
                train_steps_acc = old["train_steps_acc"].tolist()
                test_steps_acc = old["test_steps_acc"].tolist()
                train_steps_loss = old["train_steps_loss"].tolist()
                test_steps_loss = old["test_steps_loss"].tolist()
            else:
                train_steps_acc, test_steps_acc = [], []
                train_steps_loss, test_steps_loss = [], []

            train_steps_acc += tr_step_acc
            test_steps_acc += va_step_acc
            train_steps_loss += tr_step_loss
            test_steps_loss += va_step_loss

            np.savez(
                steps_npz,
                train_steps_acc=np.array(train_steps_acc, dtype=object),
                test_steps_acc=np.array(test_steps_acc, dtype=object),
                train_steps_loss=np.array(train_steps_loss, dtype=object),
                test_steps_loss=np.array(test_steps_loss, dtype=object)
            )

            _plot_acc_steps(
                train_steps=train_steps_acc,
                test_steps=test_steps_acc,
                out_path=os.path.join(run_dir, "acc_steps.png")
            )

            epochs = [h["epoch"] for h in hist]
            gap_acc = [h["train_acc"] - h["test_acc"] for h in hist]
            gap_loss = [h["train_loss"] - h["test_loss"] for h in hist]

            _plot_gap(
                epochs, gap_acc,
                out_path=os.path.join(run_dir, "gap_acc.png"),
                title="Gap (train_acc - test_acc)",
                ylabel="Acc Gap (pp)"
            )

            _plot_gap(
                epochs, gap_loss,
                out_path=os.path.join(run_dir, "gap_loss.png"),
                title="Gap (train_loss - test_loss)",
                ylabel="Loss Gap"
            )

            _plot_epoch_curves(
                epochs,
                [h["train_acc"] for h in hist],
                [h["test_acc"] for h in hist],
                [h["train_loss"] for h in hist],
                [h["test_loss"] for h in hist],
                out_path_prefix=os.path.join(run_dir, "curves_epoch")
            )

        if args.LRScheduler == 'multi_step':
            scheduler.step()
        
        elif args.LRScheduler == 'plateau':
            scheduler.step(val_acc)
        
        # remember best acc@1 and save checkpoint
        is_best = val_acc > best_acc1
        best_acc1 = max(val_acc, best_acc1)

        file_name = f'{args.arch}-{args.optimizer}.pth.tar'

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best, filename=file_name)


def train(train_loader, model, criterion, optimizer, epoch, device, args, scheduler, create_graph, two_steps):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    
    # [NEW]
    step_acc = []
    step_loss = []

    # switch to train mode
    model.train()

    if args.optimizer == 'msassha':
        optimizer.move_up_to_momentumAscent()
    for i, (images, target) in enumerate(train_loader):
       
        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # [NEW if branch] Seperate SAM optimizers
        if args.optimizer in ['samsgd', 'samadamw']:
            cache = {}

            def closure(param_overrides=None):
                sync_ctx = contextlib.nullcontext()
                if (param_overrides is None) and torch.distributed.is_initialized() and hasattr(model, "no_sync"):
                    sync_ctx = model.no_sync()

                with sync_ctx:
                    if param_overrides is None:
                        enable_running_stats(model)
                        out = model(images)
                    else:
                        disable_running_stats(model)
                        out = functional_call(model, param_overrides, (images,))

                    l = criterion(out, target)

                if param_overrides is None:
                    cache["output"] = out.detach()
                    cache["loss"] = l.detach()

                return l


            optimizer.step(closure=closure)

            output = cache["output"]
            loss = cache["loss"]

        else:
            if two_steps:
                with maybe_no_sync(model):
                    enable_running_stats(model)
                    output = model(images)
                    loss = criterion(output, target)
                    loss.backward()

                    if args.optimizer == 'sassha':
                        optimizer.perturb_weights(zero_grad=True)
                        
                    elif args.optimizer in ['samsgd', 'samadamw']:
                        optimizer.first_step(zero_grad=True)

                    disable_running_stats(model)
                    criterion(model(images), target).backward(create_graph=create_graph)

                    if args.optimizer == 'sassha':
                        optimizer.unperturb()
                
                if args.optimizer == 'sassha':
                    optimizer.step()
                    optimizer.zero_grad()
                    
                elif args.optimizer in ['samsgd', 'samadamw']:
                    if args.grad_clip_norm != 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    optimizer.second_step(zero_grad=True)
                    
            else:
                output = model(images)
                loss = criterion(output, target)
                loss.backward(create_graph=create_graph)

                if args.grad_clip_norm != 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

                optimizer.step()
                optimizer.zero_grad()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # [NEW]
        if args.log_steps:
            global_step = epoch * len(train_loader) + i
            step_acc.append((global_step, _to_float(acc1[0])))
            step_loss.append((global_step, _to_float(loss)))

        # Edit
        if args.LRScheduler == 'cosine': 
            scheduler.step()

        if i % args.print_freq == 0:
            progress.display(i + 1)
    
    if args.optimizer == 'msassha':
        optimizer.move_back_from_momentumAscent()

    # [CHANGED]
    return (_to_float(top1.avg), _to_float(losses.avg), step_acc, step_loss)

    
def validate(val_loader, model, criterion, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # [NEW]
                if args.log_steps:
                    offset = (args.current_epoch_for_logging + 1) * args.train_steps_per_epoch
                    global_step = offset + (i - base_progress)
                    step_acc.append((global_step, _to_float(acc1[0])))
                    step_loss.append((global_step, _to_float(loss)))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    
    # [NEW]
    step_acc = []
    step_loss = []

    # switch to evaluate mode
    model.eval()
    
    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    # [CHANGED]
    return (_to_float(top1.avg), _to_float(losses.avg), step_acc, step_loss)

def maybe_no_sync(model):
    if torch.distributed.is_initialized():
        return model.no_sync()
    else:
        return contextlib.ExitStack()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# [NEW]
def _to_float(x):
    if hasattr(x, "item"):
        return float(x.item())
    
    return float(x)

# [NEW]
def _is_main_process(args, ngpus_per_node):
    if not args.multiprocessing_distributed:
        return True
    
    return hasattr(args, "rank") and (args.rank % ngpus_per_node == 0)

# [NEW]
def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# [NEW]
def _append_csv(path, fieldnames, row_dict):
    file_exists = os.path.isfile(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row_dict)

# [NEW]
def _save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# [NEW]
def _plot_acc_steps(train_steps, test_steps, out_path):
    plt.figure()

    if len(train_steps) > 0:
        tx, ta = zip(*train_steps)
        plt.plot(tx, ta)

    if len(test_steps) > 0:
        vx, va = zip(*test_steps)
        plt.plot(vx, va)

    plt.xlabel("global step")
    plt.ylabel("Acc@1 (%)")
    plt.title("Step-wise Accuracy (train/test)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# [NEW]
def _plot_gap(epochs, gaps, out_path, title, ylabel):
    plt.figure()

    plt.plot(epochs, gaps)

    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# [NEW]
def _plot_epoch_curves(epochs, tr_acc, va_acc, tr_loss, va_loss, out_path_prefix):
    plt.figure()

    plt.plot(epochs, tr_acc)
    plt.plot(epochs, va_acc)

    plt.xlabel("epoch")
    plt.ylabel("Acc@1 (%)")
    plt.title("Epoch Accuracy (train/test)")

    plt.tight_layout()
    plt.savefig(out_path_prefix + "_acc.png", dpi=200)
    plt.close()

    plt.figure()

    plt.plot(epochs, tr_loss)
    plt.plot(epochs, va_loss)

    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.title("Epoch Loss (train/test)")

    plt.tight_layout()
    plt.savefig(out_path_prefix + "_loss.png", dpi=200)
    plt.close()


if __name__ == '__main__':
    main()
