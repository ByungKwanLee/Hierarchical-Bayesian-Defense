#!/usr/bin/env python
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import math
from loader.loader import dataset_loader

# arguments
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
parser = argparse.ArgumentParser(description='Bayesian Inference')
parser.add_argument('--model', type=str, default='wide')
parser.add_argument('--defense', type=str, default='adv_hvi')
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--root', type=str, default='./datasets')
parser.add_argument('--n_ensemble', type=str, default='50')
parser.add_argument('--steps', type=int, default=10)
parser.add_argument('--max_norm', type=float, default=0.03)
opt = parser.parse_args()


# dataset
print('==> Preparing data..')
opt.root='./datasets'
_, _, _ = dataset_loader(opt)

# load model
if opt.model == 'vgg':
    if opt.defense in ('adv'):
        from models.vgg import VGG
        net = VGG('VGG16', opt.n_classes, img_width=opt.img_size, mean=opt.mean, std=opt.std).cuda()
    elif opt.defense in ('adv_vi'):
        from models.vgg_vi import VGG
        net = VGG(0.1, 0, 0.1, 'VGG16', opt.n_classes, img_width=opt.img_size, mean=opt.mean, std=opt.std).cuda()
    elif opt.defense in ('adv_hvi'):
        from models.vgg_hvi import VGG
        net = VGG(0.1, 0, 0.1, 'VGG16', opt.n_classes, img_width=opt.img_size, mean=opt.mean, std=opt.std).cuda()
elif opt.model == 'aaron':
    if opt.defense in ('adv'):
        from models.aaron import Aaron
        net = Aaron(opt.n_classes, mean=opt.mean, std=opt.std).cuda()
    elif opt.defense in ('adv_vi'):
        from models.aaron_vi import Aaron
        net = Aaron(0.1, 0, 0.1, opt.n_classes, mean=opt.mean, std=opt.std).cuda()
    elif opt.defense in ('adv_hvi'):
        from models.aaron_hvi import Aaron
        net = Aaron(0.1, 0, 0.1, opt.n_classes, mean=opt.mean, std=opt.std).cuda()
elif opt.model == 'resnet':
    if opt.defense in ('adv'):
        from models.resnet import resnet
        net = resnet(num_classes=opt.n_classes, mean=opt.mean, std=opt.std).cuda()
    elif opt.defense in ('adv_vi'):
        from models.resnet_vi import resnet
        net = resnet(sigma_0=0.1, init_s=0.1, num_classes=opt.n_classes, mean=opt.mean, std=opt.std).cuda()
    elif opt.defense in ('adv_hvi'):
        from models.resnet_hvi import resnet
        net = resnet(sigma_0=0.1, init_s=0.1, num_classes=opt.n_classes, mean=opt.mean, std=opt.std).cuda()
elif opt.model == 'wide':
    if not 'vi' in opt.defense:
        from models.wideresnet import WideResNet
        net = WideResNet(opt, depth=16, num_classes=opt.n_classes, widen_factor=8, mean=opt.mean, std=opt.std).cuda()
    elif '_vi' in opt.defense:
        from models.wideresnet_vi import WideResNet
        net = WideResNet(opt, sigma_0=0.1, init_s=0.1, depth=16, num_classes=opt.n_classes, widen_factor=8, mean=opt.mean, std=opt.std).cuda()
    elif '_hvi' in opt.defense:
        from models.wideresnet_hvi import WideResNet
        net = WideResNet(opt, sigma_0=0.1, init_s=0.1, depth=16, num_classes=opt.n_classes, widen_factor=8, mean=opt.mean, std=opt.std).cuda()
else:
    raise ValueError('invalid opt.model')



net.load_state_dict(torch.load('./checkpoint/{}_{}_{}_{}.pth'.format(opt.data, opt.model, opt.max_norm, opt.defense)))
net.cuda()
net.eval() # must set to evaluation mode
loss_f = nn.CrossEntropyLoss()
softmax = nn.Softmax(dim=1)
cudnn.benchmark = True

mu_weight = 0
sigma_weight = 0
len_ = 0


if opt.model == "wide" and '_hvi' in opt.defense:
    kl_weight = []

    for layer in [net.conv1, net.bn1, net.fc]:
        if 'Rand' in type(layer).__name__:
            mu_weight = layer.mu_weight
            sigma_weight = layer.sigma_weight

            sig_weight = torch.exp(sigma_weight)
            kl_weight.append(torch.mean(1/2*torch.log(1+mu_weight**2/sig_weight**2)).item())
            print(type(layer).__name__ + ' : ', torch.mean(1/2*torch.log(1+mu_weight**2/sig_weight**2)).item())


    for blocks in [net.block1, net.block2, net.block3]:
        for block in blocks.layer:
                for layer in [block.bn1, block.conv1, block.bn2, block.conv2, block.convShortcut]:
                    if 'Rand' in type(layer).__name__:
                        mu_weight = layer.mu_weight
                        sigma_weight = layer.sigma_weight

                        sig_weight = torch.exp(sigma_weight)
                        kl_weight.append(torch.mean(1/2*torch.log(1+mu_weight**2/sig_weight**2)).item())
                        print(type(layer).__name__ + ' : ', torch.mean(1/2*torch.log(1+mu_weight**2/sig_weight**2)).item())


    print('[adv-hvi] KLD: ', sum(kl_weight)/len(kl_weight))

    len_ = 0
    numbering = 0
    mu_weight = []
    sigma_weight = []
    for idx, (name, param) in enumerate(net.named_parameters()):
        if 'sigma_weight' in name:
            len_ += 1
            sigma_weight.append(torch.mean(torch.exp(param.data)).item())

        elif 'mu_weight' in name:
            mu_weight.append(torch.sum(param.data).item())
            numbering += param.data.numel()
        else:
            continue

    avg_mu = sum(mu_weight)/numbering
    avg_sigma = sum(sigma_weight)/len_

    print('avg_mu: {:.5f}, avg_sigma: {:.5f}'.format(avg_mu, avg_sigma))

    exit(0)

if opt.model == "wide" and '_vi' in opt.defense:
    kl_weight = []

    for layer in [net.conv1, net.bn1, net.fc]:
        if 'Rand' in type(layer).__name__:
            mu_weight = layer.mu_weight
            sigma_weight = layer.sigma_weight

            sig_weight = torch.exp(sigma_weight)
            kl_weight.append(torch.mean(math.log(0.1) - sigma_weight + (sig_weight**2 + mu_weight**2) / (2*0.1**2 ) - 0.5).item())
            print(type(layer).__name__ + ' : ', torch.mean(math.log(0.1)- sigma_weight + (sig_weight**2 + mu_weight**2) / (2*0.1**2 ) - 0.5).item())


    for blocks in [net.block1, net.block2, net.block3]:
        for block in blocks.layer:
                for layer in [block.bn1, block.conv1, block.bn2, block.conv2, block.convShortcut]:
                    if 'Rand' in type(layer).__name__:
                        mu_weight = layer.mu_weight
                        sigma_weight = layer.sigma_weight

                        sig_weight = torch.exp(sigma_weight)
                        kl_weight.append(torch.mean(math.log(0.1) - sigma_weight + (sig_weight**2 + mu_weight**2) / (2*0.1**2 ) - 0.5).item())
                        print(type(layer).__name__ + ' : ', torch.mean(math.log(0.1)- sigma_weight + (sig_weight**2 + mu_weight**2) / (2*0.1**2 ) - 0.5).item())

    print('[adv-vi] KLD: ', sum(kl_weight)/len(kl_weight))

    len_ = 0
    numbering = 0
    mu_weight = []
    sigma_weight = []
    for idx, (name, param) in enumerate(net.named_parameters()):
        if 'sigma_weight' in name:
            len_ += 1
            sigma_weight.append(torch.mean(torch.exp(param.data)).item())

        elif 'mu_weight' in name:
            mu_weight.append(torch.sum(param.data).item())
            numbering += param.data.numel()
        else:
            continue

    avg_mu = sum(mu_weight)/numbering
    avg_sigma = sum(sigma_weight)/len_

    print('avg_mu: {:.5f}, avg_sigma: {:.5f}'.format(avg_mu, avg_sigma))

    exit(0)




if '_hvi' in opt.defense:
    numbering = 0
    kl_weight = []
    for layer in net.features:
        if 'Rand' in type(layer).__name__:

            mu_weight = layer.mu_weight
            sigma_weight = layer.sigma_weight

            sig_weight = torch.exp(sigma_weight)
            kl_weight.append(torch.mean(1/2*torch.log(1+mu_weight**2/sig_weight**2)).item())
            print(type(layer).__name__ + ' : ', torch.mean(1/2*torch.log(1+mu_weight**2/sig_weight**2)).item())

    layer = net.classifier
    mu_weight = layer.mu_weight
    sigma_weight = layer.sigma_weight

    sig_weight = torch.exp(sigma_weight)
    kl_weight.append(torch.mean(1/2*torch.log(1+mu_weight**2/sig_weight**2)).item())
    print('[adv-hvi] KLD: ', sum(kl_weight)/len(kl_weight))


elif '_vi' in opt.defense:
    kl_weight = []
    import math
    for layer in net.features:
        if 'Rand' in type(layer).__name__:

            len_ += 1
            mu_weight = layer.mu_weight
            sigma_weight = layer.sigma_weight
            sig_weight = torch.exp(sigma_weight)

            kl_weight.append(torch.mean(math.log(0.1) - sigma_weight + (sig_weight**2 + mu_weight**2) / (2*0.1**2 ) - 0.5).item())
            print(type(layer).__name__ + ' : ', torch.mean(math.log(0.1)- sigma_weight + (sig_weight**2 + mu_weight**2) / (2*0.1**2 ) - 0.5).item())

    layer = net.classifier
    len_ += 1
    mu_weight = layer.mu_weight
    sigma_weight = layer.sigma_weight
    sig_weight = torch.exp(sigma_weight)

    kl_weight.append(torch.mean( math.log(0.1)- sigma_weight + (sig_weight**2 + mu_weight**2) / (2*0.1**2) - 0.5).item())
    print('[adv-vi] KLD: ', sum(kl_weight)/len(kl_weight))


len_ = 0
numbering = 0
mu_weight = []
sigma_weight = []
for idx, (name, param) in enumerate(net.named_parameters()):
    if 'sigma_weight' in name:
        len_ += 1
        sigma_weight.append(torch.mean(torch.exp(param.data)).item())

    elif 'mu_weight' in name:
        mu_weight.append(torch.sum(param.data).item())
        numbering += param.data.numel()
    else:
        continue

avg_mu = sum(mu_weight)/numbering
avg_sigma = sum(sigma_weight)/len_

print('avg_mu: {:.5f}, avg_sigma: {:.5f}'.format(avg_mu, avg_sigma))