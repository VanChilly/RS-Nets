'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
Modified by VanChilly 
'''
from __future__ import print_function

import os, sys, argparse
import warnings, random, shutil, time

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import models.imagenet as customized_models
from models.drs.ocr_drs import ResolutionSelector as DRS
from tools.gumbelsoftmax import GumbelSoftmax, gumbel_softmax
from tools.flops_table import flops_table, get_flops_loss
from utils import AverageMeter, accuracy, mkdir_p
from utils.dataloaders import *
from tensorboardX import SummaryWriter

warnings.filterwarnings('ignore')

default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='parallel_resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: parallel_resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
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
parser.add_argument('--resume_all', default='', type=str, metavar='PATH_all',
                    help='path to drs and backbone')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-i', '--inference', dest='inference', action='store_true',
                    help='model inference on validation set')
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
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--lr-decay', type=str, default='schedule',
                    help='mode for learning rate decay')
parser.add_argument('--step', type=int, default=30,
                    help='interval for learning rate decay in step mode')
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60, 85, 95, 105],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--warmup', action='store_true',
                    help='set lower initial learning rate to warm up the training')

parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')

parser.add_argument('--cardinality', type=int, default=32, help='ResNeXt model cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNeXt model base width (number of channels in each group).')
parser.add_argument('--groups', type=int, default=3, help='ShuffleNet model groups')
parser.add_argument('--extent', type=int, default=0, help='GENet model spatial extent ratio')
parser.add_argument('--theta', dest='theta', action='store_true', help='GENet model parameterising the gather function')
parser.add_argument('--excite', dest='excite', action='store_true', help='GENet model combining the excite operator')

parser.add_argument('--sizes', type=int, nargs='+', default=[224, 192, 160, 128, 96],
                    help='input resolutions.')
parser.add_argument('--kd', action='store_true',
                    help='build losses of knowledge distillation across resolutions')
parser.add_argument('-t', '--kd-type', metavar='KD_TYPE', default='ens_topdown',
                    choices=['ens_topdown', 'ens'])  # stand for full-version and  vanilla-version MRED, respectively

# training mode setting
parser.add_argument('--train_mode', default=1, type=int, dest='train_mode', 
                    help='1: train backbone, 2: freeze backbone and train drs,'
                    '3: train backbone and drs')
parser.add_argument('--flops_loss', default='DRNet', type=str, help='flops loss type')

# drs settings
parser.add_argument('--eta', default=1., type=float, help='eta in DRNet')
parser.add_argument('--alpha', default=0.03, type=float, help='alpha in DRNet')

args = parser.parse_args()
n_sizes = len(args.sizes)
best_acc0, best_acc = 0, 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if device == torch.device("cuda"):
    print(f"cuda version: {torch.version.cuda}, cuda is available")
else:
    print("cpu is available")

def main():
    global best_acc0, best_acc
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting from checkpoints.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch.startswith('resnext'):
            model = models.__dict__[args.arch](
                    baseWidth=args.base_width,
                    cardinality=args.cardinality,
                )
        elif args.arch.startswith('shufflenetv1'):
            model = models.__dict__[args.arch](
                    groups=args.groups
                )
        elif args.arch.startswith('ge_resnet'):
            model = models.__dict__[args.arch](
                    extent=args.extent,
                    theta=args.theta,
                    excite=args.excite
                )
        elif args.arch.startswith('parallel') or args.arch.startswith('meta'):
            # for 32
            model = models.__dict__[args.arch](num_parallel=n_sizes, num_classes=10)

            # for 224
            # model = models.__dict__[args.arch](num_parallel=n_sizes, num_classes=10)
        else:
            model = models.__dict__[args.arch]()

    if args.arch.endswith("resnet18"):
        flops_list = flops_table['resnet18'][:len(args.sizes)]
    elif args.arch.endswith("resnet20"):
        flops_list = flops_table['resnet20'][:len(args.sizes)]
    else:
        raise NotImplementedError(f"Don't have flops table of model:{args.arch}")
    # if 'resnet' in args.arch:
    #     args.epochs = 120
    if 'mobilenetv2' in args.arch:
        args.epochs = 150
        args.lr = 0.05
        args.weight_decay = 4e-5

    alpha = nn.Parameter(torch.ones(n_sizes, requires_grad=True))
    model.register_parameter('alpha', alpha)

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.to(device)
        else:
            model = torch.nn.DataParallel(model).to(device)
    else:
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model)

    print('# parameters:', sum(param.numel() for param in model.parameters()))

    # gs = GumbelSoftmax(hard=False) # GumbelSoftmax
    gs = gumbel_softmax
    drs = DRS(scale_factor=args.sizes) # Dynamic Resolution Selector

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer_drs = torch.optim.SGD(
        [
            {'params': drs.parameters()}, 
            {'params': model.module.fc.parameters()}
        ], 
        args.lr * 0.1, 
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(
        model.parameters(), 
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    title = 'ImageNet-' + args.arch
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)


    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch']
            best_acc0 = checkpoint['best_acc0'] if args.train_mode == 1 else 0
            best_acc = checkpoint['best_acc'] if args.train_mode == 1 else 0
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'drs' in checkpoint:
                drs.load_state_dict(checkpoint['drs'])
                print(f"=> loaded drs {checkpoint['drs_name']}")
            else:
                print(f"=> [Warning]: not load drs checkpoint")

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            args.checkpoint = os.path.dirname(args.resume)
            logger = open(os.path.join(args.checkpoint, 'log.txt'), 'a+')
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return
    else:
        logger = open(os.path.join(args.checkpoint, 'log.txt'), 'w+')

    print(sys.argv)

    # save sys.argv to cmd.txt
    # with open(os.path.join(args.checkpoint, 'cmd.txt'), 'w') as f:
    #     print(sys.argv, file=f)

    # script_path = os.path.join(args.checkpoint, 'scripts')
    # mkdir_p(script_path)
    # os.system('cp *py %s' % script_path)
    # if args.arch.startswith('parallel_resnet'):
    #     arch_file = 'parallel_resnet'
    # else:
    #     arch_file = args.arch
    # os.system('cp models/imagenet/%s.py %s' % (arch_file, script_path))

    cudnn.benchmark = True

    get_train_loader, get_val_loader = get_pytorch_train_loader, get_pytorch_val_loader
    train_loader, train_loader_len = get_train_loader(args.data, args.batch_size, args.sizes, workers=args.workers)
    if args.inference:
        # we set batch size = 1 here for inference
        val_loader, val_loader_len = get_val_loader(args.data, 1, args.sizes, workers=args.workers)
    else:
        val_loader, val_loader_len = get_val_loader(args.data, args.batch_size, args.sizes, workers=args.workers)
    val_loader_in, val_loader_in_len = get_val_loader(args.data, 1, args.sizes, workers=args.workers)

    if drs.__class__.__name__ == 'ResolutionSelector':
        drs_name = 'ResolutionSelector'
    else:
        raise NotImplementedError(f"Don't know DRS -> {drs.__class__.__name__}!")

    # gs = gs.to(device)
    drs = drs.to(device)
    # visualization
    writer = SummaryWriter(os.path.join(
        args.checkpoint, 'logs', str(args.eta), str(args.alpha)
        ))

    # inference
    if args.inference:
        inference(val_loader, val_loader_len, model, logger, writer, gs, drs, flops_list)
        logger.close()
        return
    # eval
    if args.evaluate:
        validate(val_loader, val_loader_len, model, criterion, logger, alpha)
        logger.close()
        return

    if args.train_mode == 2:
        args.start_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        print('\nEpoch: %d | %d' % (epoch + 1, args.epochs))
        logger.write('\nEpoch: %d | %d\n' % (epoch + 1, args.epochs))

        # train for one epoch
        if args.train_mode == 1:
            train(
                train_loader, train_loader_len, model, criterion, optimizer, 
                epoch, logger
                )
            acc1, acc5 = validate(
                val_loader, val_loader_len, model, criterion,logger
                )
        elif args.train_mode == 2:
            train_drs(
                train_loader, train_loader_len, model, criterion, 
                optimizer_drs, epoch, logger, writer, gs, drs, flops_list
                )
        else:
            raise NotImplementedError("Error, Unknow train mode")

    # evaluate on validation set
    if args.train_mode == 2:
        acc1, acc5 = inference(
            val_loader_in, val_loader_in_len, model, logger, 
            writer, gs, drs, flops_list
            )
    
    acc = acc1 + acc5
    lr = optimizer.param_groups[0]['lr']

    # tensorboardX
    # writer.add_scalar('learning rate', lr, epoch + 1)
    # for j in range(n_sizes):
    #     writer.add_scalars('accuracy', {'validation accuracy (' + str(args.sizes[j]) + ')': acc1[j]}, epoch + 1)

    is_best = acc[0] > best_acc0
    is_best = is_best or (acc[0] == best_acc0 and sum(acc) / len(acc) > best_acc)
    best_acc0 = max(acc[0], best_acc0)
    best_acc = max(sum(acc) / len(acc), best_acc)
    # print(f"Best_acc {args.sizes[0]}: {best_acc0:.3f} Best_acc ens: {best_acc:.3f}")
    if args.mode == 1:
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc0': best_acc0,
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)
    elif args.train_mode == 2:
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'drs_name': drs_name,
            'drs': drs.state_dict(),
            'best_acc0': best_acc0,
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)


    logger.close()
    writer.close()

def train_drs(train_loader, train_loader_len, model, criterion, optimizer, 
                epoch, logger, writer, gs, drs, flops_list):
    """Train drs model

    Parameters
    ----------
    train_loader : Dataloader
        data_loader
    train_loader_len : int
        length of train_loader
    model : nn.Module
        backbone model
    criterion : Loss
        Loss criterion
    optimizer : nn.optim
        optimizer for drs
    epoch : int
        training epoch
    logger : logger
        logger
    gs : function
        gumbel_softmax function
    drs : nn.Module
        Dynamic Resolution Selector
    """
    flops_losses = AverageMeter()
    cls_losses = AverageMeter()
    model.eval()
    drs.train()
    tau = 1
    # tau = adjust_tau(tau, epoch, None)
    for i, (input, target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, i, train_loader_len, 'step')
        if device == torch.device("cuda"):
            target = target.cuda(non_blocking=True)
        else:
            target = target.to(device)
        
        reso_decision = drs(input[-1].to(device))
        gumbel_tensor = gs(training=True, x=reso_decision, tau=tau, hard=False)

        output = model(input)
        loss = 0

        flops_loss = 0
        loss_gamma = args.eta
        flops_loss = get_flops_loss(gumbel_tensor, flops_list, alpha=args.alpha, loss_type=args.flops_loss)
        gamma_mul_flops_loss = flops_loss * loss_gamma
        loss += gamma_mul_flops_loss
        output_ens = 0
        alpha_soft = [gumbel_tensor[:, i:i + 1] 
                        for i in range(gumbel_tensor.shape[1])]
        for j in range(n_sizes):
            output_ens += alpha_soft[j] * output[j].detach()
        cls_loss = criterion(output_ens, target)
        loss += cls_loss

        flops_losses.update(gamma_mul_flops_loss.item(), input[0].size(0))
        cls_losses.update(cls_loss.item(), input[0].size(0))

        # TODO kd
        if i % 100 == 0:
            print(f"[{i + 1}/{train_loader_len}]")
            print(
                f"\ngamma_mul_flops loss: {gamma_mul_flops_loss.item():.5f}"
                f" cls loss: {cls_loss.item():.5f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    writer.add_scalar(f'FLops Loss:', flops_losses.avg, epoch)
    writer.add_scalar(f'Cls Loss:', cls_losses.avg, epoch)
    return


def train(train_loader, train_loader_len, model, criterion, optimizer,  
            epoch, logger):
    # switch to train mode
    model.train() 
    for i, (input, target) in tqdm(enumerate(train_loader), total=train_loader_len):
        adjust_learning_rate(optimizer, epoch, i, train_loader_len, args.lr_decay)
        if device == torch.device("cuda"):
            target = target.cuda(non_blocking=True)
        else:
            target = target.to(device)

        # compute output
        output = model(input)
        loss = 0
        # all sizes losses
        for j in range(n_sizes):
            loss += criterion(output[j], target)

        output_ens = 0

        # KLDivLoss
        # To avoid underflow issues when computing this quantity, this loss expects the argument input in the log-space.
        # The argument target may also be provided in the log-space if log_target= True.
        if args.kd:
            loss_kd = 0
            for j in range(n_sizes):
                loss_kd += nn.KLDivLoss(reduction='batchmean')(
                    nn.LogSoftmax(dim=1)(output[j]),
                    nn.Softmax(dim=1)(output_ens.detach()))
                # ensemble distillation and top-down dense distillation
                if args.kd_type == 'ens_topdown':
                    for k in range(j):
                        loss_kd += nn.KLDivLoss(reduction='batchmean')(
                            nn.LogSoftmax(dim=1)(output[j]),
                            nn.Softmax(dim=1)(output[k].detach()))
            if args.kd_type == 'ens_topdown':
                loss += loss_kd.to(device) * 2 / (n_sizes + 1)
            else:
                loss += loss_kd.to(device)

        # compute gradient and do SGD step
        # print(f"\nloss: {loss.item():.5f}")
        if i % 100 == 0 and i != 0:
            print(f"all loss: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return


def validate(val_loader, val_loader_len, model, criterion, logger):
    top1 = [AverageMeter() for _ in range(n_sizes)]
    top5 = [AverageMeter() for _ in range(n_sizes)]
    if n_sizes > 1:
        top1_ens, top5_ens = AverageMeter(), AverageMeter()

    # switch to evaluate mode
    model.eval()
    for i, (input, target) in tqdm(enumerate(val_loader), total=val_loader_len):
        if device == torch.device("cuda"):
            target = target.cuda(non_blocking=True)
        else:
            target = target.to(device)

        with torch.no_grad():
            output = model(input)
            for j in range(n_sizes):
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output[j], target, topk=(1, 5))
                top1[j].update(acc1.item(), input[0].size(0))
                top5[j].update(acc5.item(), input[0].size(0))

    print(f"All test cases: 10000")
    for j, size in enumerate(args.sizes):
        top1_avg, top5_avg = top1[j].avg, top5[j].avg
        print('\nsize%03d: top1 %.2f, top5 %.2f' % (size, top1_avg, top5_avg))
        logger.write('size%03d: top1 %.3f, top5 %.3f\n' % (size, top1_avg, top5_avg))
    if n_sizes > 1:
        top1_ens_avg, top5_ens_avg = round(top1_ens.avg, 1), round(top5_ens.avg, 1)
        print('\nensemble: top1 %.2f, top5 %.2f' % (top1_ens_avg, top5_ens_avg))
        logger.write('ensemble top1 %.3f, top5 %.3f\n' % (top1_ens_avg, top5_ens_avg))

    return [round(t.avg, 1) for t in top1], [round(t.avg, 1) for t in top5]

def inference(val_loader, val_loader_len, model, logger, writer, gs, drs, flops_list):
    """Inference model on val dataloader via batch size = 1

    Parameters
    ----------
    val_loader : nn.DataLoader
        validation loader
    val_loader_len : int
        length if val_loader
    model : nn.Module
        the model
    logger : unknown
        logger tool
    gs : function
        gumbel softmax function
    drs : nn.Module
        DynamicResolutionSelector
    """
    data = np.zeros((6))
    def get_stat(acc):
        # acc_h acc_m, acc_l 
        # 1: acc_h > acc_m > acc_l
        # 2: acc_h > acc_l > acc_m
        # 3: acc_l > acc_h > acc_m
        # 4: acc_l > acc_m > acc_h
        # 5: acc_m > acc_h > acc_l
        # 6: acc_m > acc_l > acc_h
        h, m, l = acc[0], acc[1], acc[2]
        if h >= m >= l:
            data[0] += 1
        elif h >= l >= m:
            data[1] += 1
        elif l >= h >= m:
            data[2] += 1
        elif l >= m >= h:
            data[3] += 1
        elif m >= h >= l:
            data[4] += 1
        elif m >= l >= h:
            data[5] += 1
        else:
            raise NotImplementedError(f"Not considered situation: h:{h}, m:{m}, l:{l}")
            
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    drs.eval()
    resolution_log = [0 for _ in range(n_sizes)]
    for i, (input, target) in tqdm(enumerate(val_loader), total=val_loader_len):
        # NOTE batch size = 1 here
        if device == torch.device("cuda"):
            target = target.cuda(non_blocking=True)
        else:
            target = target.to(device)
        with torch.no_grad():
            reso_decision = drs(input[-1].to(device))
            gumbel_tensor = gs(training=False, x=reso_decision, tau=1.0, hard=True)
            # now we got [N=1, 5] tensor
            _, reso_choice = torch.max(gumbel_tensor, dim=1)
            reso_choice = reso_choice.item()
            resolution_log[reso_choice] += 1
            output = model(input)
            accs = []
            for size in range(n_sizes):
                acc1 = accuracy(output[size], target, topk=(1,))
                accs += [acc1]
            get_stat(accs)
            # we just record the chosen one's info
            acc1, acc5 = accuracy(output[reso_choice], target, topk=(1, 5))
            top1.update(acc1.item(), input[reso_choice].size(0))
            top5.update(acc5.item(), input[reso_choice].size(0))

    print(f"All test cases: 10000")
    flops = 0
    print(f'\ntop1: {top1.avg:.3f} top5: {top5.avg}')    
    for j, size in enumerate(args.sizes):
        print(f"#size{size}: {resolution_log[j]}")
        flops += flops_list[j] * resolution_log[j]
    print(f"Average Costs: {flops / 10000:.5f} GFLOPs")
    print(
        f"State: \n"
        f"1.: {data[0]}\n"
        f"2.: {data[1]}\n"
        f"3.: {data[2]}\n"
        f"4.: {data[3]}\n"
        f"5.: {data[4]}\n"
        f"6.: {data[5]}\n"
    )

    return [round(top1.avg, 1)], [round(top5.avg, 1)]

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        new_path = os.path.join(checkpoint, 'model_best.pth.tar')
        shutil.copyfile(filepath, new_path)
        print(f"Checkpoint: {new_path} saved!")
    else:
        print(f"Checkpoint: {filepath} saved!")
    


from math import cos, pi
def adjust_learning_rate(optimizer, epoch, iteration, num_iter, lr_decay):
    lr = optimizer.param_groups[0]['lr']

    warmup_epoch = 5 if args.warmup else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter

    if lr_decay == 'step':
        lr = args.lr * (args.gamma ** ((current_iter - warmup_iter) // (max_iter - warmup_iter)))
    elif lr_decay == 'cos':
        lr = args.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif lr_decay == 'linear':
        lr = args.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif lr_decay == 'schedule':
        count = sum([1 for s in args.schedule if s <= epoch])
        lr = args.lr * pow(args.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(lr_decay))

    if epoch < warmup_epoch:
        lr = args.lr * current_iter / warmup_iter

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_tau(tau, epoch, itr):
    tau = tau * 0.1 ** (epoch + 1 % 5)
    return tau

if __name__ == '__main__':
    main()
    print("Done!")
