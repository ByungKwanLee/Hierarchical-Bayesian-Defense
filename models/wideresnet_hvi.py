import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.batchnorm2d import RandHierBatchNorm2d
from .layers.conv2d import RandHierConv2d
from .layers.hierarchical_linear import RandHierarchicalLinear

class BasicBlock(nn.Module):
    def __init__(self, sigma_0, init_s, in_planes, out_planes, stride):
        super(BasicBlock, self).__init__()
        self.sigma_0 = sigma_0
        self.init_s = init_s
        self.bn1 = RandHierBatchNorm2d(self.sigma_0,0,self.init_s,in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = RandHierConv2d(self.sigma_0,0,self.init_s,in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = RandHierBatchNorm2d(self.sigma_0,0,self.init_s,out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = RandHierConv2d(self.sigma_0,0,self.init_s,out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and RandHierConv2d(self.sigma_0,0,self.init_s,in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        kl_sum = 0
        if not self.equalInOut:
            out, kl = self.bn1(x)
            kl_sum += kl
            x = self.relu1(out)
            out, kl = self.conv1(x)
            kl_sum += kl
        else:
            out, kl = self.bn1(x)
            kl_sum += kl
            out = self.relu1(out)
            out, kl = self.conv1(out)
            kl_sum += kl

        out, kl = self.bn2(out)
        kl_sum += kl
        out = self.relu2(out)
        out, kl = self.conv2(out)
        kl_sum += kl

        if self.equalInOut:
            return torch.add(x, out), kl_sum
        else:
            iden, kl = self.convShortcut(x)
            kl_sum += kl
            return torch.add(iden, out), kl_sum

class NetworkBlock(nn.Module):
    def __init__(self, sigma_0, init_s, nb_layers, in_planes, out_planes, block, stride):
        super(NetworkBlock, self).__init__()
        self.sigma_0 = sigma_0
        self.init_s = init_s
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(self.sigma_0, self.init_s, i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1))
        return nn.Sequential(*layers)
    def forward(self, x):
        kl_sum = 0
        out = x
        for l in self.layer:
            out, kl = l(out)
            kl_sum += kl
        return out, kl_sum

class WideResNet(nn.Module):
    def __init__(self, args, sigma_0, init_s, depth, num_classes, widen_factor=1, mean=0.5, std=0.25):
        super(WideResNet, self).__init__()

        self.args = args
        self.mean = mean
        self.std = std
        self.sigma_0 = sigma_0
        self.init_s = init_s

        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = RandHierConv2d(self.sigma_0,0,self.init_s, 3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(self.sigma_0, self.init_s, n, nChannels[0], nChannels[1], block, 1)
        # 2nd block
        self.block2 = NetworkBlock(self.sigma_0, self.init_s, n, nChannels[1], nChannels[2], block, 2)
        # 3rd block
        self.block3 = NetworkBlock(self.sigma_0, self.init_s, n, nChannels[2], nChannels[3], block, 2)
        # global average pooling and classifier
        self.bn1 = RandHierBatchNorm2d(self.sigma_0,0,self.init_s,nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = RandHierarchicalLinear(self.sigma_0,0,self.init_s,nChannels[3], num_classes)
        self.nChannels = nChannels[3]

    def forward(self, x, flag=False):

        kl_sum = 0

        out = (x - self.mean) / self.std
        out, kl = self.conv1(out)
        kl_sum += kl

        out, kl = self.block1(out)
        kl_sum += kl

        out, kl = self.block2(out)
        kl_sum += kl

        out, kl = self.block3(out)
        kl_sum += kl

        out, kl = self.bn1(out)
        kl_sum += kl

        out = self.relu(out)
        out = F.avg_pool2d(out, 16) if self.args.data=="tiny" else F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)

        out, kl = self.fc(out)
        kl_sum += kl

        if flag:
            return out, kl_sum
        else:
            return out