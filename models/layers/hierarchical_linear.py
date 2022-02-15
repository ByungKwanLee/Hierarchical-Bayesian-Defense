import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

class RandHierarchicalLinear(nn.Module):
    def __init__(self, sigma_0, N, init_s, in_features, out_features, bias=True):
        super(RandHierarchicalLinear, self).__init__()
        self.a = torch.Tensor(out_features, in_features)
        self.b = torch.Tensor(out_features)

        self.sigma_0 = sigma_0
        self.N = N
        self.in_features = in_features
        self.out_features = out_features
        self.init_s = init_s
        self.mu_weight = Parameter(torch.Tensor(out_features, in_features))    #hierarchical mu
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features)) #hierarchical sigma
        self.register_buffer('eps_mu', torch.Tensor(out_features, in_features))
        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_features))
            self.sigma_bias = Parameter(torch.Tensor(out_features))
            self.register_buffer('eps_mu_bias', torch.Tensor(out_features))
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('sigma_bias', None)
            self.register_buffer('eps_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.mu_weight.size(1))
        self.mu_weight.data.uniform_(-stdv, stdv)
        self.sigma_weight.data.fill_(self.init_s)
        self.eps_mu.data.zero_()
        if self.mu_bias is not None:
            self.mu_bias.data.uniform_(-stdv, stdv)
            self.sigma_bias.data.fill_(self.init_s)
            self.eps_mu_bias.data.zero_()

    def reset_epsilon(self):
        self.a = self.eps_mu.normal_().clone()
        self.b = self.eps_mu_bias.normal_().clone()


    def forward(self, input, KL_flag=True, reset=True):

        if reset:
            self.reset_epsilon()

        sig_weight = torch.exp(self.sigma_weight)
        weight = self.mu_weight + sig_weight * self.a

        if KL_flag:
            kl_weight = 1/2*torch.log(1+self.mu_weight**2/sig_weight**2)

        if self.mu_bias is not None:
            sig_bias = torch.exp(self.sigma_bias)
            bias = self.mu_bias + sig_bias * self.b
            if KL_flag:
                kl_bias = 1/2*torch.log(1+self.mu_bias**2/sig_bias**2)

        out = F.linear(input, weight, bias) if self.mu_bias is not None else F.linear(input, weight)

        if KL_flag:
            kl = kl_weight.sum() + kl_bias.sum() if self.mu_bias is not None else kl_weight.sum()
            return out, kl
        else:
            return out
