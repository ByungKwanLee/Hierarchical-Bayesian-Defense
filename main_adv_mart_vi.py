#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import Adam

import math
import os
import argparse
from tqdm import tqdm

from utils.loss import elbo
from loader.loader import dataset_loader, attack_loader

os.environ["CUDA_VISIBLE_DEVICES"]="1"
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--steps', default=10, type=int, help='#adv. steps')
parser.add_argument('--max_norm', default=0.03, type=float, help='Linf-norm in PGD')
parser.add_argument('--sigma_0', default=0.1, type=float, help='Gaussian prior')
parser.add_argument('--init_s', default=0.1, type=float, help='Initial log(std) of posterior')
parser.add_argument('--data', default='cifar10', type=str, help='dataset name')
parser.add_argument('--model', default='wide', type=str, help='model name')
parser.add_argument('--root', default='./datasets', type=str, help='path to dataset')
opt = parser.parse_args()
opt.init_s = math.log(opt.init_s) # init_s is log(std)

model_out = './checkpoint/' + opt.data+'_'+opt.model+'_'+str(opt.max_norm)+'_adv_mart_vi.pth'


# Data
print('==> Preparing data..')
trainloader, testloader, N = dataset_loader(opt)

# Model
if opt.model == 'vgg':
    from models.vgg_vi import VGG
    net = VGG(opt.sigma_0, 0, opt.init_s, 'VGG16', opt.n_classes, img_width=opt.img_size, mean=opt.mean, std=opt.std).cuda()
elif opt.model == 'resnet':
    from models.resnet_vi import resnet
    net = resnet(opt.sigma_0, opt.init_s, num_classes=opt.n_classes, mean=opt.mean, std=opt.std).cuda()
elif opt.model == 'wide':
    from models.wideresnet_vi import WideResNet
    net = WideResNet(opt, sigma_0=opt.sigma_0, init_s=opt.init_s, depth=16, num_classes=opt.n_classes, widen_factor=8, mean=opt.mean, std=opt.std).cuda()
elif opt.model == 'aaron':
    from models.aaron_vi import Aaron
    net = Aaron(opt.sigma_0, 0, opt.init_s, opt.n_classes, mean=opt.mean, std=opt.std).cuda()
else:
    raise NotImplementedError('Invalid model')


opt.attack='pgd'
attack = attack_loader(opt, net, opt.max_norm)
kl_func = nn.KLDivLoss(reduction='none')

cudnn.benchmark = True
beta = 0.01 / N

print('beta is ', beta*N)
print('sigma_0 ', opt.sigma_0)
print('')

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
        outputs, kl = net(adv_x, flag=True)

        logits = net(inputs)
        adv_probs = F.softmax(outputs, dim=1)
        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == targets, tmp1[:, -2], tmp1[:, -1])
        loss_adv = F.cross_entropy(outputs, targets) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
        nat_probs = F.softmax(logits, dim=1)
        true_probs = torch.gather(nat_probs, 1, (targets.unsqueeze(1)).long()).squeeze()
        loss_robust = (1.0 / inputs.shape[0]) * torch.sum(
            torch.sum(kl_func(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
        loss = loss_adv + 10*loss_robust + beta*kl

        loss.backward()
        optimizer.step()
        pred = torch.max(outputs, dim=1)[1]
        correct += torch.sum(pred.eq(targets)).item()
        total += targets.numel()
        if batch_idx % 50 == 0 and batch_idx != 0:
            print('[TRAIN] Iter: {}, Acc: {:.3f}, Loss: {:.3f}, CE: {:.3f}, KL: {:.3f}'.format(
                batch_idx, 100.*correct/total,
                loss.item(),
                loss.item()-beta*kl.item(),
                beta*kl.item()))

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
        print('beta is ', beta*N)
        print('sigma_0 ', opt.sigma_0)
        print('')
    # Save checkpoint.
    torch.save(net.state_dict(), model_out)

# For early stopping.
# Ref: Eric Wong et al.           (https://openreview.net/pdf?id=BJx040EFvH)
# Ref: Chawin Sitawarin et al.    (https://arxiv.org/abs/2003.09347)
if opt.data == 'cifar10':
    epochs = [20, 20, 20]
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