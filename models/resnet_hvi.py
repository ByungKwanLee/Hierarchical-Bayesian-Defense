'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.batchnorm2d import RandHierBatchNorm2d
from .layers.conv2d import RandHierConv2d
from .layers.hierarchical_linear import RandHierarchicalLinear

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, sigma_0, init_s, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.sigma_0 = sigma_0
        self.init_s = init_s
        self.conv1 = RandHierConv2d(self.sigma_0, 0 , self.init_s,
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = RandHierBatchNorm2d(self.sigma_0, 0 , self.init_s, planes)
        self.conv2 = RandHierConv2d(self.sigma_0, 0 , self.init_s, planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = RandHierBatchNorm2d(self.sigma_0, 0 , self.init_s, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                RandHierConv2d(self.sigma_0, 0 , self.init_s, in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                RandHierBatchNorm2d(self.sigma_0, 0 , self.init_s, self.expansion*planes)
            )

    def forward(self, x):
        kl_sum = 0
        out, kl = self.conv1(x)
        kl_sum += kl
        out, kl = self.bn1(out)
        kl_sum += kl
        out = F.relu(out)
        out, kl = self.conv2(out)
        kl_sum += kl
        out, kl = self.bn2(out)
        kl_sum += kl

        identity = x
        for l in self.shortcut:
            identity, kl = l(identity)
            kl_sum += kl
        out += identity
        out = F.relu(out)
        return out, kl_sum


class ResNet(nn.Module):
    def __init__(self, sigma_0, init_s, block, num_blocks, num_classes=10, mean=0.5, std=0.25):
        super(ResNet, self).__init__()

        self.mean = mean
        self.std = std
        self.sigma_0 = sigma_0
        self.init_s = init_s

        self.in_planes = 64

        self.conv1 = RandHierConv2d(self.sigma_0, 0, self.init_s, 3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = RandHierBatchNorm2d(self.sigma_0, 0, self.init_s, 64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = RandHierarchicalLinear(self.sigma_0, 0, self.init_s, 512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.sigma_0, self.init_s, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, flag=False):
        kl_sum = 0
        out = (x - self.mean) / self.std
        out, kl = self.conv1(out)
        kl_sum += kl
        out, kl = self.bn1(out)
        kl_sum += kl
        out = F.relu(out)
        for l in self.layer1:
            out, kl = l(out)
            kl_sum += kl
        for l in self.layer2:
            out, kl = l(out)
            kl_sum += kl
        for l in self.layer3:
            out, kl = l(out)
            kl_sum += kl
        for l in self.layer4:
            out, kl = l(out)
            kl_sum += kl
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out, kl = self.linear(out)
        kl_sum += kl

        if flag:
            return out, kl_sum
        else:
            return out


def resnet(sigma_0, init_s, num_classes, mean, std):
    return ResNet(sigma_0, init_s, BasicBlock, [2, 2, 2, 2], num_classes, mean, std)
