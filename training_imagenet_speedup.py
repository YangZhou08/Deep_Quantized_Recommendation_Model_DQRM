import argparse
import os
import random
import shutil
import time
import logging
import warnings

import numpy as np
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

from pyhessian import hessian
from pyhessian import get_params_grad

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
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
parser.add_argument('--test-batch-per-gpu', default=64, type=int,
                    help='batch size of one GPUs during testing when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0, type=float, metavar='M',
                    help='momentum, (default: 0.9)')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
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
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--checkpoint-dir', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--metric', default='averageL1', type=str,
                    help='metric used to conduct structured communications')
parser.add_argument('--hessian-batch-size', default=256, type=int,
                    help='batch size to calculate Hessian trace')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--top-k', default=20, type=int,
                    help='top-k to communicate')

best_acc1 = 0

args = parser.parse_args()
if not os.path.exists(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', filename=args.checkpoint_dir+'log.log')
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

# large parallel experiments, we assume hessian batch size per node is the same as training batch size
# assert(args.hessian_batch_size == args.batch_size)


# def pre_average(g):
#     g[0, 0, 0, :] *= 0
#     # model.conv1.weight.requires_gradient = False
#     print(f'Pre-DDP hook ({g.device}): {g[0, 0, 0, :]}')
#     # print(model.conv1.weight.requires_gradient)


# def post_average(g):
#     print(f'Post-DDP hook ({g.device}): {g[0, 0, 0, :]}')


def average_gradients_update(model, args, indicator, top_k, communication, trace=None):

    global tmp_list_all
    # global trace

    tmp_list = torch.zeros((1), requires_grad=False).cuda(args.gpu)
    # tmp_list = []

    if indicator % args.world_size == 0:
        # print("calculate metrics")
        counter = 0
        for name, param in model.named_parameters():
            if len(param.size()) >= 2:
                l = param.grad.data
                # l = param.grad.data.clone()
                tmp_l = l.view(l.size(0), -1)

                if not args.metric == 'hessian':
                    tmp_norm = tmp_l.norm(dim=1) ** 2 / tmp_l.size(1)
                else:
                    tmp_norm = (tmp_l.norm(dim=1) ** 2) * trace[counter] / tmp_l.size(1)
                # tmp_norm = tmp_l.norm(dim=1)
                # tmp_norm = tmp_l.norm(p=1, dim=1) / tmp_l.size(1)

                tmp_list = torch.cat((tmp_list, tmp_norm), 0)
                # numpy version tmp_list
                # tmp_list += list(tmp_norm.cpu().numpy())
                counter += 1

        _, index = torch.topk(tmp_list, top_k, largest=True, sorted=False, out=None)
        # numpy version tmp_list
        # tmp_list = np.array(tmp_list)
        # index = tmp_list.argsort()[-top_k:][::-1]

        tmp_list *= 0
        tmp_list[index] = 1

        if indicator == 0:
            tmp_list_all = torch.zeros((int(args.world_size), tmp_list.shape[0]),
                                       dtype=torch.uint8, requires_grad=False).cuda(args.gpu)
        else:
            tmp_list_all *= 0
        # tmp_list_all = torch.zeros((int(device_amount), tmp_list.size), dtype=torch.uint8).cuda(args.rank)
        tmp_list_all[args.rank] = tmp_list.to(torch.uint8)
        # tmp_list_all[args.rank] = torch.from_numpy(tmp_list)

        # tmp_list_all = torch.new_tensor(tmp_list_all, requires_grad=False)
        dist.all_reduce(tmp_list_all, op=dist.ReduceOp.SUM)

    pointer = 0

    for name, param in model.named_parameters():
        if len(param.size()) >= 2:
            pointer_next = pointer + param.size(0)
            all_reduce_index = tmp_list_all[int(indicator % args.world_size), pointer: pointer_next].to(torch.bool)
            pointer = pointer_next

            num_communicate = torch.sum(all_reduce_index)
            if num_communicate == 0:
                continue

            communication += torch.numel(param) * num_communicate / (param.size(0) * 1000000)

            # print(all_reduce_index)
            update = param.grad.data[all_reduce_index]
            # print(update.size())
            dist.all_reduce(update, op=dist.ReduceOp.SUM)
            update = update / args.world_size

            if args.weight_decay != 0:
                update.add_(args.weight_decay, param.data[all_reduce_index])

            # momentum correction
            # if momentum != 0:
            #     param_state = self.state[param]
            #     if 'momentum_buffer' not in param_state:
            #         buf = param_state['momentum_buffer'] = torch.clone(update).detach()
            #     else:
            #         buf = param_state['momentum_buffer']
            #         buf.mul_(momentum).add_(1, update)
            #     update = buf

            param.data[all_reduce_index].add_(-args.lr, update)
            param.grad.data[all_reduce_index] *= 0

        elif len(param.size()) == 1:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            update = param.grad.data / args.world_size

            if args.weight_decay != 0:
                update.add_(args.weight_decay, param.data)

            # if momentum != 0:
            #     param_state = self.state[param]
            #     if 'momentum_buffer' not in param_state:
            #         buf = param_state['momentum_buffer'] = torch.clone(update).detach()
            #     else:
            #         buf = param_state['momentum_buffer']
            #         buf.mul_(momentum).add_(1, update)
            #     update = buf

            param.data.add_(-args.lr, update)
            param.grad.data *= 0

    # if args.rank == 1:
    #     logging.info(torch.sum(tmp_list_all[int(indicator % args.world_size), 0: 10000]))
    # print(pointer)
    # return tmp_list_all


def average_gradients(model):
    world_size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= world_size


def get_hessian_data(dataloader, args, ngpus_per_node):
    hessian_data = []
    # The Hessian batch size is per node, args.batch_size here is per GPU
    tmp_number = args.hessian_batch_size / (args.batch_size * ngpus_per_node)
    # print(tmp_number)
    for i, (data, target) in enumerate(dataloader):
        if i < tmp_number:
            hessian_data.append((data, target))
        else:
            break

    return hessian_data


def get_k_value(k, epoch, total_epoch, dataset):

    dynanmic_k = 0

    if dataset == 'imagenet':
        if epoch > 60:
            dynanmic_k = 4 * k
        elif epoch > 30:
            dynanmic_k = 2 * k
        else:
            dynanmic_k = k
    elif dataset == 'cifar10':
        if epoch > 150:
            dynanmic_k = 8 * k
        elif epoch > 120:
            dynanmic_k = 4 * k
        elif epoch > 60:
            dynanmic_k = 2 * k
        else:
            dynanmic_k = k

    return dynanmic_k


def main(args):

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
        mp.spawn(main_worker, args=(ngpus_per_node, args), nprocs=ngpus_per_node)
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        logging.info("Use GPU: {} for training".format(args.gpu))

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
        logging.info("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        logging.info("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.distributed:
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
            # print(args.gpu)
            # print(model)
            # model.conv1.weight.register_hook(pre_average)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            # model.module.conv1.weight.register_hook(post_average)
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

    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
    #                             args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))
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
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = False

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch_per_gpu, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    total_communication = torch.zeros((1), requires_grad=False).cuda(args.gpu)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        communication = torch.zeros((1), requires_grad=False).cuda(args.gpu)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, ngpus_per_node, communication)

        total_communication += communication
        logging.info("Total Communication {} M".format(epoch, total_communication.item()))

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint_dir=args.checkpoint_dir)


def train(train_loader, model, criterion, optimizer, epoch, args, ngpus_per_node, communication):

    if args.metric == 'hessian':
        logging.info('***************** Trace Computation ******************')
        hessian_start = time.time()
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        # model.eval()
        hessian_data = get_hessian_data(train_loader, args, ngpus_per_node)
        # remember we always assume Hessian batch size is the same as training batch size
        hessian_comp = hessian(model, criterion, data=hessian_data, cuda=True,
                               gpu=args.gpu, ngpus=args.world_size)
        # Compute trace
        trace = []
        hessian_trace = hessian_comp.trace()

        hessian_time = time.time() - hessian_start
        logging.info('Hessian Computation time: ')
        logging.info(hessian_time)

        for vhvi in hessian_trace:
            if len(vhvi.shape) >= 2:
                trace.append(torch.squeeze(vhvi))

        counter = 0
        for p in model.parameters():
            if len(p.shape) >= 2:
                trace[counter] /= (2 * torch.numel(p) / p.shape[0])
                trace[counter] += 1
                counter += 1

        # if args.gpu == 0:
        #     for vhvi in hessian_trace:
        #         print(vhvi[0: 10, 0, 0, 0])

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    indicator = 0
    top_k = args.top_k

    end = time.time()
    start_cuda = torch.cuda.Event(enable_timing=True)
    end_cuda = torch.cuda.Event(enable_timing=True)
    epoch_begin = time.time()
    for i, (images, target) in enumerate(train_loader):

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        data_time.update(time.time() - end)

        output = model(images)
        loss = criterion(output, target)

        torch.cuda.synchronize()
        dist.barrier()
        start_cuda.record()

        optimizer.zero_grad()
        loss.backward()

        # print(f'Final ({images.device}): {model.module.conv1.weight.grad[0, 0, 0, :]}')
        if indicator == 0:
            tmp_list_all = None

        # torch.cuda.synchronize()
        # dist.barrier()
        # start_cuda.record()

        # if not args.metric == 'hessian':
        #     average_gradients_update(model, args, indicator, top_k, communication)
        # else:
        #     average_gradients_update(model, args, indicator, top_k, communication, trace)
        # # tmp_list_all = average_gradients_update(model, args.gpu, indicator, tmp_list_all)

        indicator += 1

        # average_gradients(model)

        optimizer.step()

        # measure elapsed time
        dist.barrier()
        end_cuda.record()
        # batch_time.update(time.time() - end)
        end = time.time()
        # dist.barrier()
        torch.cuda.synchronize()
        batch_time.update(start_cuda.elapsed_time(end_cuda))

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        if i % args.print_freq == 0:
            progress.display(i)
    epoch_end = time.time()
    logging.info("Epoch{} takes time {}".format(epoch, epoch_end-epoch_begin))
    logging.info("Epoch{} takes communication {} M".format(epoch, communication.item()))


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
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
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
        logging.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, checkpoint_dir=None, filename='checkpoint.pth.tar'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(state, checkpoint_dir+filename)
    if is_best:
        shutil.copyfile(checkpoint_dir+filename, checkpoint_dir+'model_best.pth.tar')


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
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main(args)
