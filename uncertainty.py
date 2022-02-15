#!/usr/bin/env python
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torchvision
import torchvision.transforms as transforms

import math
import os
from loader.loader import dataset_loader, attack_loader
from tqdm import tqdm

# arguments
os.environ["CUDA_VISIBLE_DEVICES"]="3"
parser = argparse.ArgumentParser(description='Bayesian Inference')
parser.add_argument('--model', type=str, default='wide')
parser.add_argument('--defense', type=str, default='adv_vi')
parser.add_argument('--data', type=str, default='cifar100')
parser.add_argument('--root', type=str, default='./datasets')
parser.add_argument('--n_ensemble', type=str, default='50')
parser.add_argument('--steps', type=int, default=10)
parser.add_argument('--max_norm', type=str, default='0.03')
opt = parser.parse_args()


opt.max_norm = [float(s) for s in opt.max_norm.split(',')]
opt.n_ensemble = [int(n) for n in opt.n_ensemble.split(',')]


# dataset
print('==> Preparing data..')
_, testloader, _ = dataset_loader(opt)

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


if opt.max_norm[0] == 0:
    opt.max_norm[0] = int(opt.max_norm[0])

print('./checkpoint/{}_{}_{}_{}.pth'.format(opt.data, opt.model, opt.max_norm[0], opt.defense))
state_dict = torch.load('./checkpoint/{}_{}_{}_{}.pth'.format(opt.data, opt.model, opt.max_norm[0], opt.defense))
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
net.load_state_dict(new_state_dict)
# net.load_state_dict(state_dict)
net.cuda()
net.eval() # must set to evaluation mode
loss_f = nn.CrossEntropyLoss()
softmax = nn.Softmax(dim=1)
cudnn.benchmark = True



def ensemble_inference(x_in, n_ensemble, flag=False):
    batch = x_in.size(0)
    prev = 0
    prob = torch.FloatTensor(batch, opt.n_classes).zero_().cuda()
    answer = []
    uncertainty = []
    with torch.no_grad():
        for n in n_ensemble:
            for i in range(n - prev):
                p = softmax(net(x_in, flag)[0]) if flag else softmax(net(x_in))
                prob.add_(p)
            answer.append((prob/n).clone())
            prev = n
        for i, a in enumerate(answer):
            uncertainty.append((-a*torch.log2(a+1e-10)).sum(dim=1).mean())
    return uncertainty

def distance(x_adv, x):
    diff = (x_adv - x).view(x.size(0), -1)
    out = torch.mean(torch.max(torch.abs(diff), 1)[0]).item()
    return out

def noperturb_test(n_ensemble, flag=False):
    correct = 0
    total = 0
    uncertainty = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            soft = 0
            for i in range(n_ensemble):
                outputs_ = net(inputs, flag=flag)[0] if flag else net(inputs)
                soft += F.softmax(outputs_, dim=1)

            uncertainty += (-soft/n_ensemble*torch.log2(soft/n_ensemble+1e-10)).sum(dim=1).mean()

        print('[{} No perturbation] Unc: {:.2f}'.format(1, uncertainty/(batch_idx+1)))



# Iterate over test set
if not 'vi' in opt.defense:
    noperturb_test(1)
    # for eps in list(map(float, np.linspace(0, 0.03, 14)[1:])):
    for eps in [0.03, 0.06, 0.08]:


        opt.attack='pgd'
        attack = attack_loader(opt, net, eps)

        u_ = [0] * len(opt.n_ensemble)
        total = 0
        max_iter = 100
        distortion = 0
        batch = 0
        for it, (x, y) in enumerate(tqdm(testloader)):
            x, y = x.cuda(), y.cuda()
            x_adv = attack(x, y) if eps != 0 else x
            uncertainty = ensemble_inference(x_adv, [1] * len(opt.n_ensemble))
            for i, u in enumerate(uncertainty):
                u_[i] += u.item()
            distortion += distance(x_adv, x)
            batch += 1
            if it >= max_iter:
                break
        for i, u in enumerate(u_):
            u_[i] = str(u/(it+1))
        print('(1) [pgd attack]' + ' Unc: {}'.format(u_) + 'max_norm: {:.3f}'.format(distortion / batch))

# Iterate over test set
if 'vi' in opt.defense:
    noperturb_test(opt.n_ensemble[0], flag=True)
    # for eps in list(map(float, np.linspace(0, 0.03, 14)[1:])):
    for eps in [0.03, 0.06, 0.08]:


        opt.attack='pgd'
        attack = attack_loader(opt, net, eps)


        u_ = [0] * len(opt.n_ensemble)
        total = 0
        max_iter = 100
        distortion = 0
        batch = 0
        for it, (x, y) in enumerate(tqdm(testloader)):
            x, y = x.cuda(), y.cuda()
            x_adv = attack(x, y) if eps != 0 else x
            uncertainty = ensemble_inference(x_adv, opt.n_ensemble, flag=True)
            for i, u in enumerate(uncertainty):
                u_[i] += u.item()
            distortion += distance(x_adv, x)
            batch += 1
            if it >= max_iter:
                break
        for i, u in enumerate(u_):
            u_[i] = str(u/(it+1))
        print('(1) [pgd attack]' + ' Unc: {}'.format(u_) + 'max_norm: {:.3f}'.format(distortion / batch))