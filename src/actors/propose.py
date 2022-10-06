import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import RelaxedOneHotCategorical

class Attention(nn.Module):
    def __init__(self,  hparam, in_dim):
        super(Attention, self).__init__()
        self.ite = 0
        self.maxite = hparam['max_iteration']
        self.temperature = -1
        self.t_start = hparam['t_start']
        self.t_end = hparam['t_end']
        self.n_pixel = hparam['image_scale']*hparam['image_scale']
        self.in_dim = in_dim
        self.log_alpha = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, self.n_pixel, self.in_dim)))

    def forward(self,):
        if self.training:
            self.temperature = self.t_start * (self.t_end / self.t_start)**(self.ite/self.maxite)
            weights = RelaxedOneHotCategorical(temperature=self.temperature, logits=self.log_alpha, validate_args=False).rsample()
        else:
            weights = F.one_hot(torch.argmax(self.log_alpha, dim=2), num_classes=self.in_dim).float()
                
        return weights

class HACNet(nn.Module):
    def __init__(self, hparam, in_dim):
        super(HACNet, self).__init__()
        self.image_scale = hparam['image_scale']
        self.reg_coef = hparam['reg_coef']
        self.attention = Attention(hparam=hparam, in_dim=in_dim)

    def forward(self, x):
        weights = self.attention()
        return torch.einsum("bpf,bf->bp", (weights, x)), weights
