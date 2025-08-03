import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import math
    
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dt, tau, activation, bias=True, sigma_in=0.01, sigma_re=0.01):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.tau = tau
        self.sigma_in = sigma_in
        self.sigma_re = sigma_re

        activations = {'relu': F.relu, 'tanh': F.tanh}
        self.activation = activations.get(activation.lower(), F.tanh)

        self.ih = nn.Linear(input_size, hidden_size, bias=False)
        self.hh = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.ho = nn.Linear(hidden_size, output_size, bias=bias)

    def forward(self, inputs, hidden, noise=True):
        # inputs: (B, T, input_size)
        outputs = []
        
        B, T, _ = inputs.shape

        for t in range(T):
            input_t = inputs[:, t, :]  # (B, input_size)

            if noise:
                input_noise = self.sigma_in * torch.randn_like(input_t)
                re_noise = self.sigma_re * torch.randn_like(hidden)
                input_t = input_t * (1 + input_noise)
            else:
                re_noise = torch.zeros_like(hidden)

            r = self.activation(hidden)
            dh = -hidden + self.hh(r) + self.ih(input_t) + re_noise
            hidden = hidden + (self.dt / self.tau) * dh
            output = self.ho(self.activation(hidden))
            outputs.append(output.unsqueeze(1))  # Keep time dim

        return hidden, torch.cat(outputs, dim=1)  # (B, T, output_size)

    def init_hidden(self, batch_size, device=None):
        return torch.zeros(batch_size, self.hidden_size, device=device)

    

#ei model
class EI_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dt, tau, activation, bias=True, sigma_in=0.01, sigma_re=0.01, eprop=0.8):
        super(EI_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.tau = tau
        self.sigma_in = sigma_in
        self.sigma_re = sigma_re
        self.eprop = eprop
        self.e_size = int(hidden_size * eprop)
        self.i_size = hidden_size - self.e_size

        self.ih = nn.Linear(input_size, hidden_size, bias=False)
        self.hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.ho = nn.Linear(hidden_size, output_size, bias=bias)

        # Mask for E/I structure
        labels = np.array([1]*self.e_size + [-1]*self.i_size)
        mask = np.tile(labels[:hidden_size], (hidden_size, 1))
        np.fill_diagonal(mask, 0)
        self.mask = torch.tensor(mask, dtype=torch.float32)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        activations = {'relu': F.relu, 'tanh': F.tanh}
        self.activation = activations.get(activation.lower(), F.tanh)

    def reset_parameters(self):
        init.kaiming_uniform_(self.hh, a=math.sqrt(5))
        self.hh.data[:, :self.e_size] /= (self.e_size / self.i_size)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.hh)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def masked_hh(self):
        return torch.abs(self.hh) * self.mask

    def forward(self, inputs, hidden, noise=True):
        # inputs: (B, T, input_size)
        outputs = []
        B, T, _ = inputs.shape

        for t in range(T):
            input_t = inputs[:, t, :]

            if noise:
                input_noise = self.sigma_in * torch.randn_like(input_t)
                re_noise = self.sigma_re * torch.randn_like(hidden)
                input_t = input_t * (1 + input_noise)
            else:
                re_noise = torch.zeros_like(hidden)

            r = self.activation(hidden)
            dh = -hidden + F.linear(r, self.masked_hh(), self.bias) + self.ih(input_t) + re_noise
            hidden = hidden + (self.dt / self.tau) * dh
            output = self.ho(self.activation(hidden))
            outputs.append(output.unsqueeze(1))

        return hidden, torch.cat(outputs, dim=1)  # (B, T, output_size)

    def init_hidden(self, batch_size, device=None):
        return torch.zeros(batch_size, self.hidden_size, dtype=torch.float32, device=device)

    
# multi tau model
class MultiTauRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dt, tau_array, activation, bias=True, sigma_in=0.01, sigma_re=0.01):
        super(MultiTauRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.taus = tau_array
        self.sigma_in = sigma_in
        self.sigma_re = sigma_re

        activations = {'relu': F.relu, 'tanh': F.tanh}
        self.activation = activations.get(activation.lower(), F.tanh)

        self.ih = nn.Linear(input_size, hidden_size, bias=False)
        self.hh = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.ho = nn.Linear(hidden_size, output_size, bias=bias)

    def forward(self, inputs, hidden, noise=True):
        # inputs: (B, T, input_size)
        outputs = []
        B, T, _ = inputs.shape

        for t in range(T):
            input_t = inputs[:, t, :]  # (B, input_size)

            if noise:
                input_noise = self.sigma_in * torch.randn_like(input_t)
                re_noise = self.sigma_re * torch.randn_like(hidden)
                input_t = input_t * (1 + input_noise)
            else:
                re_noise = torch.zeros_like(hidden)

            r = self.activation(hidden)
            #print(f"Hidden shape: {hidden.shape}, Tau shape: {self.taus.shape}, Input shape: {input_t.shape}")
            dh = -hidden + self.hh(r) + self.ih(input_t) + re_noise
            hidden = hidden + (self.dt / self.taus) * dh
            output = self.ho(self.activation(hidden))
            outputs.append(output.unsqueeze(1))  # Keep time dim
            #print (f"taus: {self.taus}, dt: {self.dt}, dh: {dh}")

        return hidden, torch.cat(outputs, dim=1)  # (B, T, output_size)

    def init_hidden(self, batch_size, device=None):
        return torch.zeros(batch_size, self.hidden_size, device=device)

