#!/usr/bin/env python
import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from loader.loader import dataset_loader, attack_loader

# arguments
os.environ["CUDA_VISIBLE_DEVICES"]="0"
parser = argparse.ArgumentParser(description='Bayesian Inference')
parser.add_argument('--defense', type=str, default='adv_mart')
parser.add_argument('--model', type=str, default='wide')
parser.add_argument('--data', type=str, default='cifar10')
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
net.load_state_dict(torch.load('./checkpoint/{}_{}_{}_{}.pth'.format(opt.data, opt.model, opt.max_norm[0], opt.defense)))
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
    with torch.no_grad():
        for n in n_ensemble:
            for _ in range(n - prev):
                p = softmax(net(x_in, flag)[0]) if flag else softmax(net(x_in))
                prob.add_(p)
            answer.append(prob.clone())
            prev = n
        for i, a in enumerate(answer):
            answer[i] = torch.max(a, dim=1)[1]
    return answer

def distance(x_adv, x):
    diff = (x_adv - x).view(x.size(0), -1)
    out = torch.mean(torch.max(torch.abs(diff), 1)[0]).item()
    return out

def noperturb_test(n_ensemble, flag=False):
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            output_sum = 0
            for _ in range(n_ensemble):
                outputs_ = net(inputs, flag=flag)[0] if flag else net(inputs)
                output_sum += outputs_
            _, predicted = output_sum.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('[{} Ensemble with No perturbation] Acc: {:.2f}'.format(n_ensemble, 100.*correct/total))


# opt.attack='pgd'
# # Iterate over test set
# if not 'vi' in opt.defense:
#     noperturb_test(1)
#     for eps in list(map(float, np.linspace(0, 0.03, 14)[1:])):
#     # for eps in [0.015, 0.03, 0.06, 0.08]:
#
#
#         attack = attack_loader(opt, net, eps)
#
#
#         correct = [0] * len(opt.n_ensemble)
#         total = 0
#         max_iter = 100
#         distortion = 0
#         batch = 0
#         for it, (x, y) in enumerate(tqdm(testloader)):
#             x, y = x.cuda(), y.cuda()
#             x_adv = attack(x, y) if eps != 0 else x
#             pred = ensemble_inference(x_adv, [1] * len(opt.n_ensemble))
#             for i, p in enumerate(pred):
#                 correct[i] += torch.sum(p.eq(y)).item()
#             total += y.numel()
#             distortion += distance(x_adv, x)
#             batch += 1
#             if it >= max_iter:
#                 break
#         for i, c in enumerate(correct):
#             correct[i] = str(100 * c / total)
#         print('(1) [pgd attack]' + ' acc: {}, '.format(correct) + 'max_norm: {:.3f}'.format(distortion / batch))
#
# if 'vi' in opt.defense:
#     noperturb_test(opt.n_ensemble[0], flag=True)
#     # for eps in list(map(float, np.linspace(0, 0.03, 14)[1:])):
#     for eps in [0.015, 0.03, 0.06, 0.08]:
#
#         attack = attack_loader(opt, net, eps)
#
#         correct = [0] * len(opt.n_ensemble)
#         total = 0
#         max_iter = 100
#         distortion = 0
#         batch = 0
#         for it, (x, y) in enumerate(tqdm(testloader)):
#             x, y = x.cuda(), y.cuda()
#             x_adv = attack(x, y) if eps != 0 else x
#             pred = ensemble_inference(x_adv, opt.n_ensemble, flag=True)
#             for i, p in enumerate(pred):
#                 correct[i] += torch.sum(p.eq(y)).item()
#             total += y.numel()
#             distortion += distance(x_adv, x)
#             batch += 1
#             if it >= max_iter:
#                 break
#         for i, c in enumerate(correct):
#             correct[i] = str(100 * c / total)
#         print('(1) [pgd attack]' + ' acc: {}, '.format(correct) + 'max_norm: {:.3f}'.format(distortion / batch))
#
# print('')
# if 'vi' in opt.defense:
#     noperturb_test(opt.n_ensemble[0], flag=True)
#     for eps in list(map(float, np.linspace(0, 0.03, 14)[1:])):
#
#         opt.attack='eot'
#         attack = attack_loader(opt, net, eps)
#
#         correct = [0] * len(opt.n_ensemble)
#         total = 0
#         max_iter = 100
#         distortion = 0
#         batch = 0
#         for it, (x, y) in enumerate(tqdm(testloader)):
#             x, y = x.cuda(), y.cuda()
#             x_adv = attack(x, y) if eps != 0 else x
#             pred = ensemble_inference(x_adv, opt.n_ensemble, flag=True)
#             for i, p in enumerate(pred):
#                 correct[i] += torch.sum(p.eq(y)).item()
#             total += y.numel()
#             distortion += distance(x_adv, x)
#             batch += 1
#             if it >= max_iter:
#                 break
#         for i, c in enumerate(correct):
#             correct[i] = str(100 * c / total)
#         print('(2) [eot attack]' + ' acc: {}, '.format(correct) + 'max_norm: {:.3f}'.format(distortion / batch))






# loading various attack
if not 'vi' in opt.defense:
    noperturb_test(1)
else:
    noperturb_test(opt.n_ensemble[0], flag=True)

if opt.max_norm[0] == 0:
    opt.max_norm[0] = 0.03

attack_list = []
# attack_name_list = ['pgd']
attack_name_list = ['fgsm', 'pgd', 'auto_art', 'eot']
for attack_name in attack_name_list:
    opt.attack=attack_name
    attack_list.append(attack_loader(opt, net, opt.max_norm[0]))

print('')
for name, attack in zip(attack_name_list, attack_list):

    # Iterate over test set
    if not 'vi' in opt.defense:
        if name == "eot":
            continue
        correct = [0] * len(opt.n_ensemble)
        total = 0
        max_iter = 100
        distortion = 0
        batch = 0
        for it, (x, y) in enumerate(tqdm(testloader)):
            x, y = x.cuda(), y.cuda()
            x_adv = attack(x, y)
            pred = ensemble_inference(x_adv, [1] * len(opt.n_ensemble))
            for i, p in enumerate(pred):
                correct[i] += torch.sum(p.eq(y)).item()
            total += y.numel()
            distortion += distance(x_adv, x)
            batch += 1
            if it >= max_iter:
                break
        for i, c in enumerate(correct):
            correct[i] = str(100 * c / total)
        print('[{} attack]'.format(name) + ' acc: {}, '.format(correct) + 'max_norm: {:.3f}'.format(distortion / batch))

    if 'vi' in opt.defense:

        correct = [0] * len(opt.n_ensemble)
        total = 0
        max_iter = 100
        distortion = 0
        batch = 0
        for it, (x, y) in enumerate(tqdm(testloader)):
            x, y = x.cuda(), y.cuda()
            x_adv = attack(x, y)
            pred = ensemble_inference(x_adv, opt.n_ensemble, flag=True)
            for i, p in enumerate(pred):
                correct[i] += torch.sum(p.eq(y)).item()
            total += y.numel()
            distortion += distance(x_adv, x)
            batch += 1
            if it >= max_iter:
                break
        for i, c in enumerate(correct):
            correct[i] = str(100 * c / total)
        print('[{} attack]'.format(name) + ' acc: {}, '.format(correct) + 'max_norm: {:.3f}'.format(distortion / batch))