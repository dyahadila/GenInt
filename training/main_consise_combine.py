import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from data.autoaugment import ImageNetPolicy
from data.dataset import RandomLoader

import numpy as np


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=64, type=int, metavar='N',
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
parser.add_argument('--batch_size_2', default=256, type=int,
                    help='GPU id to use.')
parser.add_argument('--batch_size_3', default=32, type=int,
                    help='GPU id to use.')
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
parser.add_argument('--prefix_name', default='', type=str, metavar='PATH',
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
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--main_gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--lambda_1', default=1, type=float, metavar='M',
                    help='gan_lambda')
parser.add_argument('--lambda_2', default=0.05, type=float, metavar='M',
                    help='gan_lambda')
parser.add_argument('--datanum', default=1000000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--decrease', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--small_epo', default=-1, type=int, metavar='N',
                    help='tune smaller epoch')
parser.add_argument('--rotate',         action='store_true')
parser.add_argument('--intgenGAN',         action='store_true')
parser.add_argument('--Img2Imgnetstyle',         action='store_true')
parser.add_argument('--Stylized_original',         action='store_true')
parser.add_argument('--intGAN',         action='store_true')  # only image+intGAN
parser.add_argument('--transfer_3',         action='store_true')
parser.add_argument('--cat_transfer_3',         action='store_true')
parser.add_argument('--finetune',         action='store_true')
parser.add_argument('--saveall',         action='store_true')
parser.add_argument('--mixup',         action='store_true')
parser.add_argument('--autoaug',         action='store_true')
parser.add_argument('--augganonly2',         action='store_true')  # only augment the second loader
parser.add_argument('--ori_gan_2',         action='store_true') # only use orignal data and GAn data without the styletransfer
parser.add_argument('--observation_gan',         action='store_true')
parser.add_argument('--adam',         action='store_true')
parser.add_argument('--backup_output_dir',          type=str,       default='/your/path/to/save',  help='')


best_acc1 = 0
experiment_backup_folder = ""
writer = None
eval_writer = None

def main():
    args = parser.parse_args()

    for k, v in args.__dict__.items(): # Prints arguments and contents of config file
        print(k, ':', v)

    backup_output_dir = args.backup_output_dir
    os.makedirs(backup_output_dir, exist_ok=True)
    global experiment_backup_folder, writer, eval_writer
    if os.path.exists(backup_output_dir):
        import uuid
        import datetime
        unique_str = str(uuid.uuid4())[:8]
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
        experiment_name = args.prefix_name + timestamp + "_" + unique_str
        experiment_backup_folder = os.path.join(backup_output_dir, experiment_name)
        print("experiment folder", experiment_backup_folder)
        shutil.copytree('.', experiment_backup_folder)  #

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
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

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, experiment_backup_folder))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, experiment_backup_folder)


def main_worker(gpu, ngpus_per_node, args, experiment_backup_folder):
    # global experiment_backup_folder, writer, eval_writer

    # print(experiment_backup_folder)
    # exit()

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
        model = models.__dict__[args.arch]()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    is_main_gpu=False
    writer=None
    if args.main_gpu == gpu or not args.multiprocessing_distributed:
        is_main_gpu=True

        log_dir = os.path.join(experiment_backup_folder, "runs_{}".format(gpu))

        # os.makedirs(log_dir, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)
        eval_writer = SummaryWriter(log_dir=log_dir + '/validate_runs/')


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            if not args.adam:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    # Data loading code
    # traindir = args.data
    transfered_dir = None
    import socket
    if socket.gethostname()=='cv10':
        valdir = os.path.join("ImageNet-Data", 'val')
        imgnetdir='ImageNet-Data/train'
        if args.observation_gan:
            transfered_dir = 'path to /GANdata/BigGANgentr02'
        elif args.transfer_3:
            transfered_dir = 'path to transferred intervention/transfer_3_styfromint'
        elif args.intGAN:
            # transfered_dir = '/local/vondrick/chengzhi/GANdata/x-z_tr1.0-imgnet-rand1000-fix5time'
            transfered_dir = 'path to interventional GAN data/GANdata/x-z_tr1.0-imgnet-rand1000-int5time'

        int_gan_dir = '/local/vondrick/chengzhi/GANdata/x-z_tr1.0-imgnet-rand1000-int5time'

    print("transfered_dir", transfered_dir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.autoaug:
        print("using auto Aug")
        composed_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(), ImageNetPolicy(),
            transforms.ToTensor(),
            normalize,
        ])

    elif args.rotate:
        composed_transforms = transforms.Compose([
               # transforms.RandomRotation(90, resample=2, expand=True),
               transforms.RandomRotation((0,360), resample=2, expand=True),
               transforms.RandomResizedCrop(224),
               transforms.RandomHorizontalFlip(),
               transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
               transforms.RandomVerticalFlip(),
               transforms.ToTensor(),
               normalize,
           ])
    else:
        composed_transforms = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])


    train_dataset = datasets.ImageFolder(
        imgnetdir,
        composed_transforms
    )

    if args.augganonly2:
        composed_transforms_2 = transforms.Compose([
            # transforms.RandomRotation(90, resample=2, expand=True),
            transforms.RandomRotation((0, 360), resample=2, expand=True),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        print('aug gan only')
        if args.intGAN:
            train_transfered_dataset = RandomLoader(
                transfered_dir,
                composed_transforms_2)
        else:
            train_transfered_dataset = datasets.ImageFolder(
                transfered_dir,
                composed_transforms_2)
    else:
        if args.intGAN:
            train_transfered_dataset = RandomLoader(
                transfered_dir,
                composed_transforms)
        else:
            train_transfered_dataset = datasets.ImageFolder(
            transfered_dir,
            composed_transforms)

    if args.intgenGAN:
        from data.concat_loader import LoaderConcat_split, Loader_Random
        train_dataset_gan = datasets.ImageFolder(
            int_gan_dir, composed_transforms)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    transfered_loader = torch.utils.data.DataLoader(
        train_transfered_dataset, batch_size=args.batch_size_2, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    if args.intgenGAN:
        intGAN_loader = torch.utils.data.DataLoader(
            train_dataset_gan, batch_size=args.batch_size_3, shuffle=(train_sampler is None),
            num_workers=args.workers//4, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if args.finetune:
            adjust_finetune_learning_rate(optimizer, epoch, args)
        else:
            adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        if args.intgenGAN:
            train_3([train_loader, transfered_loader, intGAN_loader], model, criterion, optimizer, epoch, args, is_main_gpu, writer)
        else:
            train([train_loader, transfered_loader], model, criterion, optimizer, epoch, args, is_main_gpu, writer)



        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            # evaluate on validation set
            acc1, acc5 = validate(val_loader, model, criterion, args)

            eval_writer.add_scalar('Test/top1 acc', acc1, epoch * len(train_loader))
            eval_writer.add_scalar('Test/top5 acc', acc5, epoch * len(train_loader))

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            save=False
            if args.saveall:
                save=True

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename='checkpoint.pth.tar', experiment_backup_folder=experiment_backup_folder,
                epoch=epoch, save=save)


def train(train_loader_list, model, criterion, optimizer, epoch, args, is_main_gpu, writer):

    for k, v in args.__dict__.items(): # Prints arguments and contents of config file
        print(k, ':', v)


    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_2 = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top1_2 = AverageMeter('Acc@1', ':6.2f')
    top5_2 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader_list[0]),
        [batch_time, data_time, losses,losses_2, top1, top5, top1_2, top5_2],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    criterion_1 = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_2 = nn.CrossEntropyLoss().cuda(args.gpu)

    for i, datas in enumerate(zip(train_loader_list[0], train_loader_list[1])):
        # if i>10:
        #     break
        if i>args.datanum//args.batch_size and not args.arch=='resnet152':
            break
            # Simulate random sampling
        e1, e2 = datas

        images_1, target_1 = e1
        images_2, target_2 = e2

        bs = images_1.size(0)

        images_1 = images_1.cuda(args.gpu, non_blocking=True)
        images_2 = images_2.cuda(args.gpu, non_blocking=True)
        target_1 = target_1.cuda(args.gpu, non_blocking=True)
        target_2 = target_2.cuda(args.gpu, non_blocking=True)
        images = torch.cat((images_1, images_2), dim=0)
        targets = torch.cat((target_1, target_2), dim=0)

        # measure data loading time
        data_time.update(time.time() - end)

        # if args.gpu is not None:
        #     images = images.cuda(args.gpu, non_blocking=True)
        # if torch.cuda.is_available():
        #     target = target.cuda(args.gpu, non_blocking=True)

        # compute output

        if args.mixup:
            images, target_a, target_b, lam = mixup_data(images, targets)

            output = model(images)

            # loss = criterion(output, target)
            def mixup_criterion(y_a, y_b, lam):
                return lam * criterion(output, y_a) + (1 - lam) * criterion(output, y_b)

            loss = mixup_criterion(target_a, target_b, lam)

            # just for log
            out_1 = output[:bs]
            out_2 = output[bs:]
            loss_1 = criterion_1(out_1, target_1)
            loss_2 = criterion_2(out_2, target_2)


        else:
            output = model(images)
            out_1 = output[:bs]
            out_2 = output[bs:]

            loss_1 = criterion_1(out_1, target_1)
            loss_2 = criterion_2(out_2, target_2)
            loss = loss_1 + loss_2 * args.lambda_1

        # compute gradient and do SGD step

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(out_1, target_1, topk=(1, 5))
        losses.update(loss_1.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        acc1, acc5 = accuracy(out_2, target_2, topk=(1, 5))
        losses_2.update(loss_2.item(), images.size(0))
        top1_2.update(acc1[0], images.size(0))
        top5_2.update(acc5[0], images.size(0))



        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        if i % 1000==1 and is_main_gpu:
            writer.add_scalar('Train/top1 acc', top1.avg, epoch*len(train_loader_list[0]) + i)
            writer.add_scalar('Train/top5 acc', top5.avg, epoch*len(train_loader_list[0]) + i)

            writer.add_scalar('Train/top1  2 acc', top1_2.avg, epoch * len(train_loader_list[0]) + i)
            writer.add_scalar('Train/top5 2  acc', top5_2.avg, epoch * len(train_loader_list[0]) + i)
            writer.add_scalar('xent loss', losses.avg, epoch*len(train_loader_list[0]) + i)
            writer.add_scalar('xent loss 2', losses_2.avg, epoch*len(train_loader_list[0]) + i)


def train_3(train_loader_list, model, criterion, optimizer, epoch, args, is_main_gpu, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_2 = AverageMeter('Loss', ':.4e')
    losses_3 = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top1_2 = AverageMeter('Acc@1', ':6.2f')
    top5_2 = AverageMeter('Acc@5', ':6.2f')
    top1_3 = AverageMeter('Acc@1', ':6.2f')
    top5_3 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader_list[0]),
        [batch_time, data_time, losses,losses_2, losses_3, top1, top5, top1_2, top5_2, top1_3, top5_3],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    for k, v in args.__dict__.items(): # Prints arguments and contents of config file
        print(k, ':', v)

    end = time.time()
    criterion_1 = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_2 = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_3 = nn.CrossEntropyLoss().cuda(args.gpu)

    for i, datas in enumerate(zip(train_loader_list[0], train_loader_list[1], train_loader_list[2])):
        # if i>10:
        #     break
        if i>1000*1000//args.batch_size and not args.arch=='resnet152':
            break
            # Simulate random sampling
        if args.small_epo>0:
            if i==args.small_epo:
                break
        e1, e2, e3 = datas

        images_1, target_1 = e1
        images_2, target_2 = e2
        images_3, target_3 = e3

        bs = images_1.size(0)
        bs2 = images_2.size(0)

        images_1 = images_1.cuda(args.gpu, non_blocking=True)
        images_2 = images_2.cuda(args.gpu, non_blocking=True)
        images_3 = images_3.cuda(args.gpu, non_blocking=True)
        target_1 = target_1.cuda(args.gpu, non_blocking=True)
        target_2 = target_2.cuda(args.gpu, non_blocking=True)
        target_3 = target_3.cuda(args.gpu, non_blocking=True)
        images = torch.cat((images_1, images_2, images_3), dim=0)
        targets = torch.cat((target_1, target_2, target_3), dim=0)

        # measure data loading time
        data_time.update(time.time() - end)

        # if args.gpu is not None:
        #     images = images.cuda(args.gpu, non_blocking=True)
        # if torch.cuda.is_available():
        #     target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        if args.mixup:
            images, target_a, target_b, lam = mixup_data(images, targets)

            output = model(images)

            # loss = criterion(output, target)
            def mixup_criterion(out, y_a, y_b, lam):
                return lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)

            out_1 = output[:bs]
            out_2 = output[bs:bs + bs2]
            out_3 = output[bs + bs2:]

            target_a_1 = target_a[:bs]
            target_a_2 = target_a[bs:bs + bs2]
            target_a_3 = target_a[bs + bs2:]

            target_b_1 = target_b[:bs]
            target_b_2 = target_b[bs:bs + bs2]
            target_b_3 = target_b[bs + bs2:]

            loss_1 = mixup_criterion(out_1, target_a_1, target_b_1, lam)
            loss_2 = mixup_criterion(out_2, target_a_2, target_b_2, lam)
            loss_3 = mixup_criterion(out_3, target_a_3, target_b_3, lam)


        else:
            output = model(images)
            out_1 = output[:bs]
            out_2 = output[bs:bs + bs2]
            out_3 = output[bs + bs2:]

            loss_1 = criterion_1(out_1, target_1)
            loss_2 = criterion_2(out_2, target_2)
            loss_3 = criterion_3(out_3, target_3)

        optimizer.zero_grad()

        # compute gradient and do SGD step
        loss = loss_1 + loss_2*args.lambda_1 + loss_3+args.lambda_2
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(out_1, target_1, topk=(1, 5))
        losses.update(loss_1.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        acc1, acc5 = accuracy(out_2, target_2, topk=(1, 5))
        losses_2.update(loss_2.item(), images.size(0))
        top1_2.update(acc1[0], images.size(0))
        top5_2.update(acc5[0], images.size(0))

        acc1, acc5 = accuracy(out_3, target_3, topk=(1, 5))
        losses_3.update(loss_3.item(), images.size(0))
        top1_3.update(acc1[0], images.size(0))
        top5_3.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        if i % 1000==1 and is_main_gpu:
            writer.add_scalar('Train/top1 acc', top1.avg, epoch*len(train_loader_list[0]) + i)
            writer.add_scalar('Train/top5 acc', top5.avg, epoch*len(train_loader_list[0]) + i)

            writer.add_scalar('Train/top1  2 acc', top1_2.avg, epoch * len(train_loader_list[0]) + i)
            writer.add_scalar('Train/top5 2  acc', top5_2.avg, epoch * len(train_loader_list[0]) + i)

            writer.add_scalar('Train/top1  3 acc', top1_3.avg, epoch * len(train_loader_list[0]) + i)
            writer.add_scalar('Train/top5 3  acc', top5_3.avg, epoch * len(train_loader_list[0]) + i)

            writer.add_scalar('xent loss', losses.avg, epoch*len(train_loader_list[0]) + i)
            writer.add_scalar('xent loss 2', losses_2.avg, epoch*len(train_loader_list[0]) + i)
            writer.add_scalar('xent loss 3', losses_3.avg, epoch*len(train_loader_list[0]) + i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            # if i > 10:
            #     break

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
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

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg

#
# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'model_best.pth.tar')
from utils import save_checkpoint

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
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

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # if epoch<3:
    #     lr = 0.01
    # else:
    lr = args.lr * (0.1 ** (epoch // 30))
    print("current learning rate={}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_finetune_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # if epoch<3:
    #     lr = 0.01
    # else:
    lr = args.lr
    if epoch > 120:
        lr = args.lr * 0.1
    elif epoch>150:
        lr = args.lr * 0.01

    if args.arch=='resnet152' and epoch>args.decrease:
        lr = args.lr * 0.1

    print("current learning rate={}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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


def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam



if __name__ == '__main__':
    main()
