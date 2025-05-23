import numpy as np
import torch
from torch.utils.data import Dataset

def generate_instant_DM(stim, start, s_length, d_length, dt, duration):
    t = np.arange(0, duration + dt, dt)
    stim_bins = np.where((t>=start)&(t<(start+s_length)))
    dec_bins = np.where((t>=(start+s_length))&(t<=start+s_length+d_length))

    I_stim = np.zeros_like(t)
    I_stim[stim_bins] = stim

    output = np.zeros_like(t)
    output[dec_bins] = np.sign(stim)

    return t, I_stim, output

def generate_hold_DM(stim, start, s_length, dt, duration):
    t = np.arange(0, duration + dt, dt)
    stim_bins = np.where((t>=start)&(t<(start+s_length)))
    dec_bins = np.where((t>=(start+s_length)))

    I_stim = np.zeros_like(t)
    I_stim[stim_bins] = stim

    output = np.zeros_like(t)
    output[dec_bins] = np.sign(stim)

    return t, I_stim, output

def generate_perceptual_classification(cohs, starts, dt, duration):
    t = np.arange(0, duration + dt, dt)
    I_stim = np.zeros((2, len(t)))
    output = np.zeros_like(t)

    for i in range(len(cohs)):
        L_stim = cohs[i]
        R_stim = 1-cohs[i]
        stim = np.array([L_stim, R_stim])
        if i < len(cohs)-1:
            s_bins = np.where((t>=starts[i])&(t<(starts[i+1])))
        else:
            s_bins = np.where((t>=starts[i]))
        I_stim[0, s_bins] = stim[0]
        I_stim[1, s_bins] = stim[1]

        if L_stim > R_stim:
            output[s_bins] = 1
        elif L_stim < R_stim:
            output[s_bins] = -1
        else:
            output[s_bins] = 0

    return t, I_stim, output

class decisionMakingInstant(Dataset): #this looks for 0 when not output time, should it? Not if we want fixed point...
    def __init__(self, seed, stim_start_min, stim_start_max, stim_length, decision_length, sigma_length=0.01, duration=20.0, dt=0.01, size=1000):
        self.stim_start_min = stim_start_min
        self.stim_start_max = stim_start_max
        self.stim_length = stim_length
        self.decision_length = decision_length
        self.sigma_length = sigma_length
        self.duration = duration
        self.dt = dt
        self.size = size
        self.rand = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # first, randomly choose start times and both stim and decision lengths
        start = self.rand.uniform(self.stim_start_min, self.stim_start_max)
        s_length = self.rand.normal(self.stim_length, self.sigma_length)
        d_length = self.rand.normal(self.decision_length, self.sigma_length)
        while (start + s_length + d_length) > self.duration:
            start = self.rand.uniform(self.stim_start_min, self.stim_start_max)
            s_length = self.rand.normal(self.stim_length, self.sigma_length)
            d_length = self.rand.normal(self.decision_length, self.sigma_length)


        # Now, randomly choose a stimuli strength (-1, 1)
        stim = self.rand.uniform(-1, 1)

        _, I_stim, output = generate_instant_DM(stim, start, s_length, d_length, self.dt, self.duration)

        input = torch.tensor(I_stim, dtype=torch.float32)
        output = torch.tensor(output, dtype=torch.float32)

        return input, output

class decisionMakingHold(Dataset): #this looks for 0 when not output time, should it? Not if we want fixed point...
    def __init__(self, seed, stim_start_min, stim_start_max, stim_length, sigma_length=0.01, duration=20.0, dt=0.01, size=1000):
        self.stim_start_min = stim_start_min
        self.stim_start_max = stim_start_max
        self.stim_length = stim_length
        self.sigma_length = sigma_length
        self.duration = duration
        self.dt = dt
        self.size = size
        self.rand = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # first, randomly choose start times and both stim and decision lengths
        start = self.rand.uniform(self.stim_start_min, self.stim_start_max)
        s_length = self.rand.normal(self.stim_length, self.sigma_length)
        
        while (start + s_length) > self.duration:
            start = self.rand.uniform(self.stim_start_min, self.stim_start_max)
            s_length = self.rand.normal(self.stim_length, self.sigma_length)

        # Now, randomly choose a stimuli strength (-1, 1)
        stim = self.rand.uniform(-1, 1)

        _, I_stim, output = generate_hold_DM(stim, start, s_length, self.dt, self.duration)

        input = torch.tensor(I_stim, dtype=torch.float32)
        output = torch.tensor(output, dtype=torch.float32)

        return input, output
    

class perceptualClassification(Dataset):
    def __init__(self, seed, num_cohs, stim_start_min, stim_start_max, coh_radius=0.1, duration=20.0, dt=0.01, size=1000):
        self.stim_start_min = stim_start_min
        self.stim_start_max = stim_start_max
        self.duration = duration
        self.num_cohs = num_cohs
        self.coh_radius = coh_radius
        self.dt = dt
        self.size = size
        self.rand = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # first, randomly choose start times and both stim and decision lengths
        start_dists = self.rand.uniform(self.stim_start_min, self.stim_start_max, self.num_cohs)
        while np.sum(start_dists) > self.duration:
            start_dists = self.rand.uniform(self.stim_start_min, self.stim_start_max, self.num_cohs)
        starts = np.zeros(self.num_cohs)
        for i in range(self.num_cohs):
            if i == 0:
                starts[i] = start_dists[i]
            elif i > 0:
                starts[i] = starts[i-1] + start_dists[i]
        print(starts)
        cohs = np.concatenate([
            self.rand.uniform(0, 0.5 - self.coh_radius, self.num_cohs // 2),
            self.rand.uniform(0.5 + self.coh_radius, 1, self.num_cohs - self.num_cohs // 2)
        ])
        self.rand.shuffle(cohs)
        print(cohs)
        _, I_stim, output = generate_perceptual_classification(cohs, starts, self.dt, self.duration)
        input = torch.tensor(I_stim, dtype=torch.float32)
        output = torch.tensor(output, dtype=torch.float32)

        return input, output





