#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Copyright (c) Yiming Li, 2020
'''

from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import time
from model import *
from tools import *
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from torchvision import utils as vutils



parser = argparse.ArgumentParser(description='PyTorch CIFAR-10')

parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--manualSeed', type=int, default=666, help='manual seed')

parser.add_argument('--model-path', default='', help='trained model path')
parser.add_argument('--model', default='resnet', type=str,
                    help='model structure (resnet or vgg)')
parser.add_argument('--trigger', default='square', type=str,
                    help='trigger type (line or square)')
parser.add_argument('--y-target', default=1, type=int, help='target Label')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

assert args.model == 'resnet' or args.model == 'vgg', 'model structure can only be resnet or vgg'
assert args.trigger == 'line' or args.trigger == 'square', 'trigger can only be line or square'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy



# Trigger Initialize
print('==> Loading the Trigger')
if args.trigger =='square':
    from PIL import Image
    trigger = Image.open('./Trigger_default1.png')
    trigger = transforms.ToTensor()(trigger)
    print("White Square is adopted as the trigger.")
else:
    from PIL import Image
    trigger = Image.open('./Trigger_default2.png')
    trigger = transforms.ToTensor()(trigger)
    print("Black Line is adopted as the trigger.")



# alpha Initialize
print('==> Loading the Alpha')
if args.trigger =='square':
    from PIL import Image
    alpha_1 = Image.open('./Alpha_default1.png')
    alpha_1 = transforms.ToTensor()(alpha_1)

    alpha_09 = alpha_1.clone().detach() * 0.9
    alpha_08 = alpha_1.clone().detach() * 0.8
    alpha_07 = alpha_1.clone().detach() * 0.7
    alpha_06 = alpha_1.clone().detach() * 0.6
    alpha_05 = alpha_1.clone().detach() * 0.5
    alpha_04 = alpha_1.clone().detach() * 0.4
    alpha_03 = alpha_1.clone().detach() * 0.3
    alpha_02 = alpha_1.clone().detach() * 0.2
    alpha_01 = alpha_1.clone().detach() * 0.1
else:
    from PIL import Image
    alpha_1 = Image.open('./Alpha_default2.png')
    alpha_1 = transforms.ToTensor()(alpha_1)

    alpha_09 = alpha_1.clone().detach() * 0.9
    alpha_08 = alpha_1.clone().detach() * 0.8
    alpha_07 = alpha_1.clone().detach() * 0.7
    alpha_06 = alpha_1.clone().detach() * 0.6
    alpha_05 = alpha_1.clone().detach() * 0.5
    alpha_04 = alpha_1.clone().detach() * 0.4
    alpha_03 = alpha_1.clone().detach() * 0.3
    alpha_02 = alpha_1.clone().detach() * 0.2
    alpha_01 = alpha_1.clone().detach() * 0.1


def main():
    # Dataset preprocessing
    title = 'CIFAR-10 Evaluation'

    # Create Datasets

    transform_01 = transforms.Compose([
        TriggerAppending(trigger=trigger, alpha=alpha_01),
        transforms.ToTensor(),
    ])

    transform_02 = transforms.Compose([
        TriggerAppending(trigger=trigger, alpha=alpha_02),
        transforms.ToTensor(),
    ])


    transform_03 = transforms.Compose([
        TriggerAppending(trigger=trigger, alpha=alpha_03),
        transforms.ToTensor(),
    ])

    transform_04 = transforms.Compose([
        TriggerAppending(trigger=trigger, alpha=alpha_04),
        transforms.ToTensor(),
    ])


    transform_05 = transforms.Compose([
        TriggerAppending(trigger=trigger, alpha=alpha_05),
        transforms.ToTensor(),
    ])

    transform_06 = transforms.Compose([
        TriggerAppending(trigger=trigger, alpha=alpha_06),
        transforms.ToTensor(),
    ])


    transform_07 = transforms.Compose([
        TriggerAppending(trigger=trigger, alpha=alpha_07),
        transforms.ToTensor(),
    ])

    transform_08 = transforms.Compose([
        TriggerAppending(trigger=trigger, alpha=alpha_08),
        transforms.ToTensor(),
    ])

    transform_09 = transforms.Compose([
        TriggerAppending(trigger=trigger, alpha=alpha_09),
        transforms.ToTensor(),
    ])

    transform_1 = transforms.Compose([
        TriggerAppending(trigger=trigger, alpha=alpha_1),
        transforms.ToTensor(),
    ])

    print('==> Loading the dataset')

    dataloader = datasets.CIFAR10

    testset_01 = dataloader(root='./data', train=False, download=True, transform=transform_01)
    testset_02 = dataloader(root='./data', train=False, download=True, transform=transform_02)
    testset_03 = dataloader(root='./data', train=False, download=True, transform=transform_03)
    testset_04 = dataloader(root='./data', train=False, download=True, transform=transform_04)
    testset_05 = dataloader(root='./data', train=False, download=True, transform=transform_05)
    testset_06 = dataloader(root='./data', train=False, download=True, transform=transform_06)
    testset_07 = dataloader(root='./data', train=False, download=True, transform=transform_07)
    testset_08 = dataloader(root='./data', train=False, download=True, transform=transform_08)
    testset_09 = dataloader(root='./data', train=False, download=True, transform=transform_09)
    testset_1 = dataloader(root='./data', train=False, download=True, transform=transform_1)


    poisoned_target = [args.y_target] * len(testset_01.data)  # Reassign their label to the target label
    testset_01.targets = poisoned_target
    testset_02.targets = poisoned_target
    testset_03.targets = poisoned_target
    testset_04.targets = poisoned_target
    testset_05.targets = poisoned_target
    testset_06.targets = poisoned_target
    testset_07.targets = poisoned_target
    testset_08.targets = poisoned_target
    testset_09.targets = poisoned_target
    testset_1.targets = poisoned_target

    testloader_01 = torch.utils.data.DataLoader(testset_01, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    testloader_02 = torch.utils.data.DataLoader(testset_02, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    testloader_03 = torch.utils.data.DataLoader(testset_03, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    testloader_04 = torch.utils.data.DataLoader(testset_04, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    testloader_05 = torch.utils.data.DataLoader(testset_05, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    testloader_06 = torch.utils.data.DataLoader(testset_06, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    testloader_07 = torch.utils.data.DataLoader(testset_07, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    testloader_08 = torch.utils.data.DataLoader(testset_08, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    testloader_09 = torch.utils.data.DataLoader(testset_09, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    testloader_1 = torch.utils.data.DataLoader(testset_1, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)


    # Model
    print('==> Loading the model')
    if args.model == 'resnet':
        model = ResNet18()
        print("ResNet is adopted")
    else:
        model = vgg19_bn()
        print("VGG is adopted")

    assert os.path.isfile(args.model_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.model_path)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()

    test_loss_01, test_acc_01 = test(testloader_01, model, criterion, use_cuda)
    test_loss_02, test_acc_02 = test(testloader_02, model, criterion, use_cuda)
    test_loss_03, test_acc_03 = test(testloader_03, model, criterion, use_cuda)
    test_loss_04, test_acc_04 = test(testloader_04, model, criterion, use_cuda)
    test_loss_05, test_acc_05 = test(testloader_05, model, criterion, use_cuda)
    test_loss_06, test_acc_06 = test(testloader_06, model, criterion, use_cuda)
    test_loss_07, test_acc_07 = test(testloader_07, model, criterion, use_cuda)
    test_loss_08, test_acc_08 = test(testloader_08, model, criterion, use_cuda)
    test_loss_09, test_acc_09 = test(testloader_09, model, criterion, use_cuda)
    test_loss_1, test_acc_1 = test(testloader_1, model, criterion, use_cuda)






def test(testloader, model, criterion, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record standard loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)



if __name__ == '__main__':
    main()
