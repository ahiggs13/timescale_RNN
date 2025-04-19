import torch
import torch.nn as nn
import torch.nn.functional as F

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
            input_noise = torch.tensor(self.sigma_in) * torch.randn_like(inputs)
            inputs = inputs + input_noise
            re_noise = torch.tensor(self.sigma_re) * torch.randn_like(hidden)
        else:
            re_noise = torch.zeros_like(hidden)

        r = self.activation(hidden)
        dh = -hidden + self.hh(r) + self.ih(inputs) + re_noise
        hidden = hidden + (self.dt / self.tau) * dh

        output = self.ho(self.activation(hidden))

        return hidden, output


