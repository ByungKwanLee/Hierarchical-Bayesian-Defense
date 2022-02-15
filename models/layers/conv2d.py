import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class RandConv2d(nn.Module):
    def __init__(self, sigma_0, N, init_s, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(RandConv2d, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.sigma_0 = sigma_0
        self.N = N
        self.init_s = init_s
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.eps_weight = torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size).cuda()
        self.eps_bias   = torch.Tensor(out_channels).cuda()

        self.mu_weight = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.sigma_weight = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_channels))
            self.sigma_bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('sigma_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        n = n * self.kernel_size ** 2
        stdv = 1.0 / math.sqrt(n)
        self.mu_weight.data.uniform_(-stdv, stdv)
        self.sigma_weight.data.fill_(self.init_s)
        if self.mu_bias is not None:
            self.mu_bias.data.uniform_(-stdv, stdv)
            self.sigma_bias.data.fill_(self.init_s)

    def reset_epsilon(self):
        self.eps_weight.normal_()
        if self.mu_bias is not None:
            self.eps_bias.normal_()


    def forward(self, input, KL_flag=True, reset=True):

        if reset:
            self.reset_epsilon()

        sig_weight = torch.exp(self.sigma_weight)
        weight = self.mu_weight + sig_weight * self.eps_weight.clone()
        if KL_flag:
            kl_weight = math.log(self.sigma_0) - self.sigma_weight + (sig_weight**2 + self.mu_weight**2) / (2 * self.sigma_0 ** 2) - 0.5
        bias = None
        if self.mu_bias is not None:
            sig_bias = torch.exp(self.sigma_bias)
            bias = self.mu_bias + sig_bias * self.eps_bias.clone()
            if KL_flag:
                kl_bias = math.log(self.sigma_0) - self.sigma_bias + (sig_bias**2 + self.mu_bias**2) / (2 * self.sigma_0 ** 2) - 0.5
        out = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        if KL_flag:
            kl = kl_weight.sum() + kl_bias.sum() if self.mu_bias is not None else kl_weight.sum()
            return out, kl
        else:
            return out




class RandHierConv2d(nn.Module):
    def __init__(self, sigma_0, N, init_s, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(RandHierConv2d, self).__init__()


        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.sigma_0 = sigma_0
        self.N = N
        self.init_s = init_s
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.eps_mu      = torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size).cuda()
        self.eps_mu_bias = torch.Tensor(out_channels).cuda()


        self.mu_weight = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.sigma_weight = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))

        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_channels))
            self.sigma_bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('sigma_bias', None)
            self.register_parameter('eps_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        n = n *  self.kernel_size ** 2
        stdv = 1.0 / math.sqrt(n)
        self.mu_weight.data.uniform_(-stdv, stdv)
        self.sigma_weight.data.fill_(self.init_s)
        if self.mu_bias is not None:
            self.mu_bias.data.uniform_(-stdv, stdv)
            self.sigma_bias.data.fill_(self.init_s)

    def reset_epsilon(self):
        self.eps_mu.normal_()
        if self.mu_bias is not None:
            self.eps_mu_bias.normal_()


    def forward(self, input, KL_flag=True, reset=True):

        if reset:
            self.reset_epsilon()

        sig_weight = torch.exp(self.sigma_weight)
        weight = self.mu_weight + sig_weight * self.eps_mu.clone()

        if KL_flag:
            kl_weight = 1/2*torch.log(1+self.mu_weight**2/sig_weight**2)

        bias = None
        if self.mu_bias is not None:
            sig_bias = torch.exp(self.sigma_bias)
            bias = self.mu_bias + sig_bias * self.eps_mu_bias.clone()

            if KL_flag:
                kl_bias = 1/2*torch.log(1+self.mu_bias**2/sig_bias**2)
        out = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

        if KL_flag:
            kl = kl_weight.sum() + kl_bias.sum() if self.mu_bias is not None else kl_weight.sum()
            return out, kl
        else:
            return out

