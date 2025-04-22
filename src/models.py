import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import math

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dt, tau, activation, bias=True, sigma_in = 0.01, sigma_re = 0.01):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.tau = tau
        self.bias = bias
        self.sigma_in = sigma_in
        self.sigma_re = sigma_re

        if activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = F.tanh

        self.ih = nn.Linear(input_size, hidden_size, bias=False) #not sure which should have bias, also can initialize these according to seed?
        self.hh = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.ho = nn.Linear(hidden_size, output_size, bias=bias)

    def forward(self, inputs, hidden, noise=True):
        if noise:
            input_noise = torch.tensor(self.sigma_in) * torch.randn_like(inputs) + 1
            inputs = inputs * input_noise #Should be noise only when input is nonzero?
            re_noise = torch.tensor(self.sigma_re) * torch.randn_like(hidden)
        else:
            re_noise = torch.zeros_like(hidden)

        r = self.activation(hidden)
        dh = -hidden + self.hh(r) + self.ih(inputs) + re_noise
        hidden = hidden + (self.dt / self.tau) * dh

        output = self.ho(self.activation(hidden))

        return hidden, output
    
    def init_hidden(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size))
    

#ei model
class EI_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dt, tau, activation, bias=True, sigma_in = 0.01, sigma_re = 0.01, eprop=0.8):
        super(EI_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.tau = tau
        self.bias = bias
        self.sigma_in = sigma_in
        self.sigma_re = sigma_re
        self.eprop = eprop
        self.e_size = int(hidden_size * eprop)
        self.ih = nn.Linear(input_size, hidden_size, bias=False) #not sure which should have bias, also can initialize these according to seed?
        self.hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.ho = nn.Linear(hidden_size, output_size, bias=bias)
        self.i_size = hidden_size - self.e_size
        mask = np.tile([1]*self.e_size+[-1]*self.i_size, (hidden_size, 1))
        np.fill_diagonal(mask, 0)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        if activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = F.tanh
        
    def reset_parameters(self):
        init.kaiming_uniform_(self.hh, a=math.sqrt(5))
        # Scale E weight by E-I ratio
        self.hh.data[:, :self.e_size] /= (self.e_size/self.i_size)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.hh)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def init_hidden(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size), dtype=torch.float32)
    
    def masked_hh(self):
        return torch.abs(self.hh) * self.mask
        
    def forward(self, inputs, hidden, noise=True):
        if noise:
            input_noise = torch.tensor(self.sigma_in) * torch.randn_like(inputs) + 1
            inputs = inputs * input_noise #Should be noise only when input is nonzero?
            re_noise = torch.tensor(self.sigma_re) * torch.randn_like(hidden)
        else:
            re_noise = torch.zeros_like(hidden)

        r = self.activation(hidden)
        dh = -hidden + F.linear(r, self.masked_hh(), self.bias) + self.ih(inputs) + re_noise
        hidden = hidden + (self.dt / self.tau) * dh

        output = self.ho(self.activation(hidden))

        return hidden, output
    



