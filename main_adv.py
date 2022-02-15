#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import Adam, SGD
import torchvision
import torchvision.transforms as transforms

import math
import os
import argparse
from tqdm import tqdm

from loader.loader import dataset_loader, attack_loader

os.environ["CUDA_VISIBLE_DEVICES"]="0"
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--steps', default=10, type=int, help='#adv. steps')
parser.add_argument('--max_norm', default=0, type=float, help='Linf-norm in PGD')
parser.add_argument('--data', default='cifar10', type=str, help='dataset name')
parser.add_argument('--model', default='wide', type=str, help='model name')
parser.add_argument('--root', default='./datasets', type=str, help='path to dataset')

print('==> Preparing data..')
opt = parser.parse_args()
trainloader, testloader, _= dataset_loader(opt)

model_out = './checkpoint/' + opt.data+'_'+opt.model+'_'+str(opt.max_norm)+'_adv.pth'

# Model
if opt.model == 'vgg':
    from models.vgg import VGG
    net = VGG('VGG16', opt.n_classes, img_width=opt.img_size, mean=opt.mean, std=opt.std).cuda()
elif opt.model == 'resnet':
    from models.resnet import resnet
    net = resnet(num_classes=opt.n_classes, mean=opt.mean, std=opt.std).cuda()
elif opt.model == 'wide':
    from models.wideresnet import WideResNet
    net = WideResNet(opt, depth=16, num_classes=opt.n_classes, widen_factor=8, mean=opt.mean, std=opt.std).cuda()
elif opt.model == 'aaron':
    from models.aaron import Aaron
    net = Aaron(opt.n_classes, mean=opt.mean, std=opt.std).cuda()
else:
    raise NotImplementedError('Invalid model')

opt.attack='pgd'
attack = attack_loader(opt, net, opt.max_norm)


cudnn.benchmark = True

# Loss function
criterion = nn.CrossEntropyLoss().cuda()

# Training
def train(epoch):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        adv_x = attack(inputs, targets) if opt.max_norm != 0 else inputs
        optimizer.zero_grad()
        outputs = net(adv_x)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        pred = torch.max(outputs, dim=1)[1]
        correct += torch.sum(pred.eq(targets)).item()
        total += targets.numel()
        if batch_idx % 50 == 0 and batch_idx != 0:
            print('[TRAIN] Iter: {}, Acc: {:.3f}, Loss: {:.3f}'.format(
                batch_idx, 100.*correct/total,
                loss.item()))


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('[TEST] Acc: {:.3f}'.format(100.*correct/total))
    # Save checkpoint.
    torch.save(net.state_dict(), model_out)

# For early stopping.
# Ref: Eric Wong et al.           (https://openreview.net/pdf?id=BJx040EFvH)
# Ref: Chawin Sitawarin et al.    (https://arxiv.org/abs/2003.09347)
if opt.data == 'cifar10':
    epochs = [30, 20, 10]
elif opt.data == 'cifar100':
    epochs = [30, 20, 10]
elif opt.data == 'stl10':
    epochs = [30, 20, 10]
elif opt.data == 'tiny':
    epochs = [30, 20, 10]

count = 0
for epoch in epochs:
    optimizer = Adam(net.parameters(), lr=opt.lr)
    for _ in range(epoch):
        train(count)
        test(count)
        count += 1
    opt.lr /= 10