#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This is the implement of Paired T testing on CIFAR-10 dataset.

Copyright (c) Yiming Li, Ziqi Zhang,  2020
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
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from model import *
from tools import *
from scipy.stats import ttest_rel
import gtsrb_dataset as dataset
import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch GTSRB')

parser.add_argument('--num-img', default=100, type=int, metavar='N',
                    help='number of images for testing (default: 100)')

parser.add_argument('--num-test', default=100, type=int,
                    help='number of T-test')

parser.add_argument('--select-class', default=2, type=int,
                    help='class from 0 to 9 (default: 2)')
parser.add_argument('--target-label', default=1, type=int,
                    help='the class chosen to be attacked (default: 1)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--model-path', default='', help='trained model path')
parser.add_argument('--model', default='resnet', type=str,
                    help='model structure (resnet or vgg)')
parser.add_argument('--trigger', help='Trigger (image size)')
parser.add_argument('--alpha', help='(1-Alpha)*Image + Alpha*Trigger')
parser.add_argument('--margin', default=0.2, type=float, help='the margin in the pairwise T-test')




args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

assert args.model == 'resnet' or args.model == 'vgg', 'model structure can only be resnet or vgg'


# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy



# Trigger Initialize
print('==> Loading the Trigger')

from PIL import Image
args.trigger = Image.open(args.trigger)
args.trigger = transforms.ToTensor()(args.trigger)
assert (torch.max(args.trigger) < 1.001)

args.alpha = Image.open(args.alpha)
args.alpha = transforms.ToTensor()(args.alpha)
assert (torch.max(args.alpha) < 1.001)

def main():
    # Dataset preprocessing
    title = 'GTSRB pairwise T testing'

    # Load model
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

    # Create 2 Dataloaders
    transform_test_watermarked = transforms.Compose([
        transforms.Resize((32, 32)),
        TriggerAppending(trigger=args.trigger, alpha=args.alpha),
        transforms.ToTensor(),
    ])

    transform_test_standard = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])


    print('==> Loading the dataset')


    Stats = [-1]*args.num_test
    p_value = [-1]*args.num_test

    for iters in range(args.num_test):
        # Random seed
        random.seed(random.randint(1, 10000))

        #Construct watermarked dataset

        testset_watermarked_new = dataset.GTSRB_Testset(root_dir='./data', train=False, transform=transform_test_watermarked, select_class=args.select_class, num_img=args.num_img)

        testset_standard_new = dataset.GTSRB_Testset(root_dir='./data', train=False, transform=transform_test_standard, select_class=args.select_class, num_img=args.num_img)

        watermarked_loader = torch.utils.data.DataLoader(testset_watermarked_new, batch_size=args.test_batch,
                                                         shuffle=False, num_workers=args.workers)
        standard_loader = torch.utils.data.DataLoader(testset_standard_new, batch_size=args.test_batch,
                                                         shuffle=False, num_workers=args.workers)

        output_watermarked = test(watermarked_loader, model, use_cuda)
        output_standard = test(standard_loader, model, use_cuda)


        # export the target label
        target_select_water = [(output_watermarked[i, args.target_label]).cpu().detach().numpy() for i in range(len(output_watermarked))]
        target_select_stand = [(output_standard[i, args.target_label]).cpu().detach().numpy() for i in range(len(output_standard))]

        target_select_water = np.array(target_select_water)
        target_select_stand = np.array(target_select_stand)

        T_test = ttest_rel(target_select_stand + args.margin, target_select_water)

        Stats[iters], p_value[iters] = T_test[0], T_test[1]

        print("%i/%i" % (iters, args.num_test))

    idx_success_detection = [i for i in range(args.num_test) if
                             (Stats[i] < 0) and (p_value[i] < 0.05 / 2)]  # single-sided hypothesis test
    rsd = float(len(idx_success_detection)) / args.num_test

    path_folder = args.model_path[:-len(args.model_path.split("/")[-1])]  # remove "checkpoint.pth.tar"

    pd.DataFrame(Stats).to_csv(path_folder + "Stats.csv", header=None)
    pd.DataFrame(p_value).to_csv(path_folder + "p_value.csv", header=None)
    pd.DataFrame([rsd]).to_csv(path_folder + "RSD.csv", header=None)

    print("RSD =", rsd)


def test(testloader, model, use_cuda):

    # switch to evaluate mode
    model.eval()


    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        p = torch.nn.functional.softmax(outputs)


    return p





if __name__ == '__main__':
    main()
