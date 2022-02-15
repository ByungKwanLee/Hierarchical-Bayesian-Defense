import torch
import torch.nn as nn
from .layers.batchnorm2d import RandBatchNorm2d
from .layers.conv2d import RandConv2d
from .layers.linear import RandLinear


class Aaron(nn.Module):
    def __init__(self, sigma_0, N, init_s, nclass, mean, std):
        super(Aaron, self).__init__()
        nchannel = 32
        self.mean = mean
        self.std = std
        self.features = nn.Sequential(
                # 3x96x96
                RandConv2d(sigma_0, N, init_s, 3, nchannel, kernel_size=3, padding=1),
                RandBatchNorm2d(sigma_0, N, init_s, nchannel),
                # nn.BatchNorm2d(nchannel),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # chx48x48
                RandConv2d(sigma_0, N, init_s, nchannel, 2*nchannel, kernel_size=3, padding=1),
                RandBatchNorm2d(sigma_0, N, init_s, 2*nchannel),
                # nn.BatchNorm2d(2*nchannel),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 2chx24x24
                RandConv2d(sigma_0, N, init_s, 2*nchannel, 4*nchannel, kernel_size=3, padding=1),
                RandBatchNorm2d(sigma_0, N, init_s, 4*nchannel),
                # nn.BatchNorm2d(4*nchannel),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 4chx12x12
                RandConv2d(sigma_0, N, init_s, 4*nchannel, 8*nchannel, kernel_size=3, padding=1),
                RandBatchNorm2d(sigma_0, N, init_s, 8*nchannel),
                # nn.BatchNorm2d(8*nchannel),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 8chx6x6
                RandConv2d(sigma_0, N, init_s, 8*nchannel, 8*nchannel, kernel_size=3, padding=0),
                RandBatchNorm2d(sigma_0, N, init_s, 8*nchannel),
                # nn.BatchNorm2d(8*nchannel),
                nn.ReLU(),
                # 8chx4x4
                RandConv2d(sigma_0, N, init_s, 8*nchannel, 16*nchannel, kernel_size=3, padding=0),
                RandBatchNorm2d(sigma_0, N, init_s, 16*nchannel),
                # nn.BatchNorm2d(16*nchannel),
                nn.ReLU(),
                # 8chx2x2
                nn.AvgPool2d(kernel_size=2, stride=2)
                )
        self.classifier = RandLinear(sigma_0, N, init_s, 16*nchannel, nclass)

    def forward(self, input, flag=False):
        kl_sum = 0
        out = (input-self.mean)/self.std
        for l in self.features:
            if type(l).__name__.startswith("Rand"):
                out, kl = l.forward(out)
                if kl is not None:
                    kl_sum += kl
            else:
                out = l.forward(out)
        out = out.view(out.size(0), -1)
        out, kl = self.classifier(out)
        kl_sum += kl

        if flag:
            return out, kl_sum
        else:
            return out


