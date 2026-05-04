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

def generate_delay_DM(stim, start, s_length, delay, dt, duration):
    t = np.arange(0, duration + dt, dt)
    stim_bins = np.where((t>=start)&(t<(start+s_length)))
    dec_bins = np.where((t>=(start+s_length+delay)))

    I_stim = np.zeros_like(t)
    I_stim[stim_bins] = stim

    output = np.zeros_like(t)
    output[dec_bins] = np.sign(stim)

    return t, I_stim, output

def generate_perceptual_classification(cohs, starts, symmetric, dt, duration):
    t = np.arange(0, duration + dt, dt)
    I_stim = np.zeros((len(t), 2))
    output = np.zeros_like(t)

    for i in range(len(cohs)):
        if symmetric:
            L_stim = cohs[i]
            R_stim = cohs[i]-1
        else:
            L_stim = cohs[i]
            R_stim = 1-cohs[i]

        stim = np.array([L_stim, R_stim])
        if i < len(cohs)-1:
            s_bins = np.where((t>=starts[i])&(t<(starts[i+1])))
        else:
            s_bins = np.where((t>=starts[i]))
        I_stim[s_bins, 0] = stim[0]
        I_stim[s_bins, 1] = stim[1]

        if L_stim > np.abs(R_stim):
            output[s_bins] = 1
        elif L_stim < np.abs(R_stim):
            output[s_bins] = -1
        else:
            output[s_bins] = 0

    return t, I_stim, output

def generate_perceptual_classification_50_test(start, symmetric, dt, duration):
    t = np.arange(0, duration + dt, dt)
    I_stim = np.zeros((len(t), 2))
    output = np.zeros_like(t)

    L_stim = 0.5
    if symmetric:
        R_stim = -0.5
    else:
        R_stim = 0.5
    
    stim = np.array([L_stim, R_stim])
    s_bins = np.where((t>=start))
    I_stim[s_bins, 0] = stim[0]
    I_stim[s_bins, 1] = stim[1]

    output[s_bins] = 0

    return t, I_stim, output

def generate_review_task(values, start, delay, cue, s_length, dt, duration):
    t = np.arange(0, duration + dt, dt)
    I_stim = np.zeros((len(t), len(values) + 1))
    output = np.zeros_like(t)
    curtime = start
    target_sum = 0
    for i in range(len(values)):
        s_bins = np.where((t>=curtime)&(t<(curtime + s_length)))
        I_stim[s_bins, 0] = values[i]
        curtime += s_length
        if cue[i] == 1:
            target_sum += values[i]

    cue_bins = np.where((t>=curtime + delay))
    I_stim[cue_bins, 1:] = cue.numpy()
    output[cue_bins] = target_sum

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
    
class decisionMakingDelay(Dataset): #this looks for 0 when not output time, should it? Not if we want fixed point...
    def __init__(self, seed, stim_start_min, stim_start_max, stim_length, delay_length, sigma_length=0.01, sigma_delay=0, duration=20.0, dt=0.01, size=1000):
        self.stim_start_min = stim_start_min
        self.stim_start_max = stim_start_max
        self.stim_length = stim_length
        self.sigma_length = sigma_length
        self.delay_length = delay_length
        self.sigma_delay = sigma_delay
        self.duration = duration
        self.dt = dt
        self.size = size
        self.rand = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # first, randomly choose start times and both stim and decision lengths
        if self.sigma_delay == 0:
            d_length = self.delay_length
            start = self.rand.uniform(self.stim_start_min, self.stim_start_max)
            s_length = self.rand.normal(self.stim_length, self.sigma_length)
            while (start + s_length + d_length) > self.duration:
                start = self.rand.uniform(self.stim_start_min, self.stim_start_max)
                s_length = self.rand.normal(self.stim_length, self.sigma_length)
        else:
            start = self.rand.uniform(self.stim_start_min, self.stim_start_max)
            s_length = self.rand.normal(self.stim_length, self.sigma_length)
            d_length = self.rand.normal(self.delay_length, self.sigma_delay)
            while (start + s_length + d_length) > self.duration:
                start = self.rand.uniform(self.stim_start_min, self.stim_start_max)
                s_length = self.rand.normal(self.stim_length, self.sigma_length)
                d_length = self.rand.normal(self.delay_length, self.sigma_delay)

        # Now, randomly choose a stimuli strength (-1, 1)
        stim = self.rand.uniform(-1, 1)

        _, I_stim, output = generate_delay_DM(stim, start, s_length, d_length, self.dt, self.duration)

        input = torch.tensor(I_stim, dtype=torch.float32)
        output = torch.tensor(output, dtype=torch.float32)

        return input, output
    
class perceptualClassification(Dataset):
    def __init__(self, seed, num_cohs, stim_start_min, stim_start_max, coh_radius=0.1, symmetric=True, duration=20.0, dt=0.01, size=1000):
        self.stim_start_min = stim_start_min
        self.stim_start_max = stim_start_max
        self.duration = duration
        self.num_cohs = num_cohs
        self.coh_radius = coh_radius
        self.symmetric = symmetric
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
        #print(starts)
        cohs = np.concatenate([
            self.rand.uniform(0, 0.5 - self.coh_radius, self.num_cohs // 2),
            self.rand.uniform(0.5 + self.coh_radius, 1, self.num_cohs - self.num_cohs // 2)
        ])
        self.rand.shuffle(cohs)
        #print(cohs)
        _, I_stim, output = generate_perceptual_classification(cohs, starts, self.symmetric, self.dt, self.duration)
        input = torch.tensor(I_stim, dtype=torch.float32)
        output = torch.tensor(output, dtype=torch.float32)

        return input, output

class perceptualClassification_50_test(Dataset):
    def __init__(self, seed, stim_start_min, stim_start_max, symmetric=True, duration=20.0, dt=0.01, size=1000):
        self.stim_start_min = stim_start_min
        self.stim_start_max = stim_start_max
        self.duration = duration
        self.symmetric = symmetric
        self.dt = dt
        self.size = size
        self.rand = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # first, randomly choose start times and both stim and decision lengths
        start = self.rand.uniform(self.stim_start_min, self.stim_start_max)

        # Now, randomly choose a stimuli strength (-1, 1)
        #stim = self.rand.uniform(-1, 1)

        _, I_stim, output = generate_perceptual_classification_50_test(start, self.symmetric, self.dt, self.duration)

        input = torch.tensor(I_stim, dtype=torch.float32)
        output = torch.tensor(output, dtype=torch.float32)

        return input, output

class reviewTask(Dataset):
    def __init__(self, seed, num_values, stim_start_min, stim_start_max, stim_length, value_min, value_max, delay, delay_sigma, stim_sigma = 0, duration=20.0, dt=0.01, size=1000):
        self.stim_start_min = stim_start_min
        self.stim_start_max = stim_start_max
        self.num_values = num_values
        self.stim_length = stim_length
        self.value_min = value_min
        self.value_max = value_max
        self.delay = delay
        self.delay_sigma = delay_sigma
        self.stim_sigma = stim_sigma
        self.duration = duration
        self.dt = dt
        self.size = size
        self.rand = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # first, randomly choose start times and both stim and decision lengths
        start = self.rand.uniform(self.stim_start_min, self.stim_start_max, 1)
        delay = self.rand.normal(self.delay, self.delay_sigma)
        if self.stim_sigma == 0:
            s_length = self.stim_length
            while (start + (self.num_values * s_length) + delay) >= self.duration:
                start = self.rand.uniform(self.stim_start_min, self.stim_start_max, 1)
                delay = self.rand.normal(self.delay, self.delay_sigma)
        else:
            s_length = self.rand.normal(self.stim_length, self.stim_sigma)
            while (start + (self.num_values * s_length) + delay) >= self.duration:
                start = self.rand.uniform(self.stim_start_min, self.stim_start_max, 1)
                s_length = self.rand.normal(self.stim_length, self.stim_sigma)
                delay = self.rand.normal(self.delay, self.delay_sigma)
        
        #print(starts)
        values = self.rand.uniform(self.value_min, self.value_max, self.num_values)
        self.rand.shuffle(values)

        cue = torch.full((self.num_values,), 0.5, dtype=torch.float32)
        cue = torch.bernoulli(cue).float()

        #print(cohs)
        _, I_stim, output = generate_review_task(values, start, delay, cue, s_length, self.dt, self.duration)
        input = torch.tensor(I_stim, dtype=torch.float32)
        output = torch.tensor(output, dtype=torch.float32)

        return input, output

class NonstationaryRewardDelayDataset(Dataset):
    def __init__(self, seed, kernel_tau, read_delay=3, cue_onset=3, integration_window=10,
                 noise_mean=0.0, noise_std=1.0, autocorr=0.0, duration=20, dt=0.01, size=1000):
        # General parameters
        self.seed = seed
        self.duration = duration
        self.dt = dt
        self.size = size
        self.sequence_length = int((duration + dt) / dt)

        # Task parameters
        self.cue_onset = int(cue_onset / dt)
        self.kernel_tau = int(kernel_tau / dt)
        self.integration_window = int(integration_window / self.dt)
        self.read_delay = int(read_delay / dt)

        # Noise parameters
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.autocorr = autocorr  # 0 = white noise, 0.99 = very slow

    def __len__(self):
        return self.size

    def _exp_weighted_avg(self, rewards, tau, n=None):
        """Compute exponential-weighted average of past rewards with timescale tau, using only n steps back if specified."""
        T = len(rewards)
        if n is not None:
            rewards = rewards[-n:]
            T = len(rewards)
        weights = np.exp(-np.arange(1, T+1) / tau) # 1 step ago gets largest weight
        return np.sum(rewards[::-1] * weights) / np.sum(weights)

    def __getitem__(self, index):
        rng = np.random.default_rng([self.seed, index])
        inputs = torch.zeros((self.sequence_length, 1))
        outputs = torch.zeros(self.sequence_length)

        x = 0.0
        for t in range(self.cue_onset, self.sequence_length):
            noise = rng.normal(self.noise_mean, self.noise_std)
            x = self.autocorr * x + np.sqrt(1 - self.autocorr**2) * noise
            inputs[t] = x
            current_output_bin = max(t + 1 - self.read_delay, 1)
            outputs[t] = self._exp_weighted_avg(inputs[:current_output_bin, 0].numpy(), self.kernel_tau, n=self.integration_window)
            outputs = outputs*10 # scale up to make it more learnable

        return inputs, outputs

class DelayDiscrimination(Dataset):
    def __init__(self, seed, interval_length, stim_length, stim_noise, stim1_strength, stim2_strength,
        stim_cue_on=False, stim_cue_strength=1.0, temporal_noise=100, duration=20, dt=0.01, size=1000):
        
        # General parameters
        self.seed = seed
        self.duration = duration
        self.dt = dt
        self.size = size
        self.sequence_length = int((duration + dt) / dt)

        # Task parameters
        self.temporal_noise = temporal_noise # 100 is a good value for strong temporal noise, 0 for no temporal noise
        self.interval_length = interval_length
        self.stim_length = stim_length
        self.stim_noise = stim_noise
        self.stim1_strength = stim1_strength
        self.stim2_strength = stim2_strength

        # Cue parameters
        self.stim_cue_on = stim_cue_on
        self.stim_cue_strength = stim_cue_strength

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Use a different random seed for each sample to ensure variability
        rng = np.random.default_rng(self.seed + index*10)

        # Add temporal noise to the interval length and stimulus onsets
        interval_length = self.interval_length + rng.normal(0, self.temporal_noise)
        stim1_onset = int((self.sequence_length - interval_length - self.stim_length * 2) / 2 + rng.normal(0, self.temporal_noise)) 
        stim2_onset = int(stim1_onset + self.stim_length + interval_length + rng.normal(0, self.temporal_noise))

        # Generate input and output sequences
        inputs = torch.zeros(self.sequence_length)
        outputs = torch.zeros(self.sequence_length)

        # Add the two stimuli with noise
        inputs[stim1_onset:stim1_onset + self.stim_length] = torch.tensor(self.stim1_strength + rng.normal(0, self.stim_noise, self.stim_length), dtype=torch.float32)
        inputs[stim2_onset:stim2_onset + self.stim_length] = torch.tensor(self.stim2_strength + rng.normal(0, self.stim_noise, self.stim_length), dtype=torch.float32)

        # Add a cue signal if enabled
        if self.stim_cue_on:
            cue = torch.zeros(self.sequence_length)
            cue[stim1_onset:stim1_onset + self.stim_length] = self.stim_cue_strength
            cue[stim2_onset:stim2_onset + self.stim_length] = self.stim_cue_strength
            inputs = torch.stack([inputs, cue], dim=1)

        # Set the output to 1 if stimulus 1 is stronger, otherwise 0, starting after the second stimulus ends
        outputs[stim2_onset + self.stim_length:] = 1 if self.stim1_strength > self.stim2_strength else -1

        return inputs, outputs