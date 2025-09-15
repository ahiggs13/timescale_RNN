import os
from itertools import accumulate
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
import yaml
import fire 
import numpy as np
from scipy.stats import truncnorm

from src import analysis, datasets, models, trainer

def get_tau_array(distribution, hidden_size,  device, tau_groups=None, tau_proportions=None, tau_min=None, tau_max=None, tau_mean1=None, tau_mean2=None, tau_std1=None, tau_std2=None, tau_change=None):
    if distribution == 'groups':
        tau_array = torch.zeros(hidden_size, dtype=torch.float32, device=device)
        tausizes = [int(p * hidden_size) for p in tau_proportions]
        tauindices = [0] + list(accumulate(tausizes))
        for i, tau in enumerate(tau_groups):
            tau_array[tauindices[i]:tauindices[i+1]] = tau
        return tau_array
    elif distribution == 'uniform':
        if tau_min is None or tau_max is None:
            raise ValueError("tau_min and tau_max must be provided for uniform distribution.")
        return torch.FloatTensor(hidden_size).uniform_(tau_min, tau_max).to(device)
    elif distribution == 'bimodal_normal':
        tau_array = torch.zeros(hidden_size, dtype=torch.float32, device=device)
        tausizes = [int(p * hidden_size) for p in tau_proportions]
        tauindices = [0] + list(accumulate(tausizes))

        if tau_mean1 < tau_mean2:
            lesser_mean = tau_mean1
            greater_mean = tau_mean2
            lesser_std = tau_std1
            greater_std = tau_std2
        else:
            lesser_mean = tau_mean2
            greater_mean = tau_mean1
            lesser_std = tau_std2
            greater_std = tau_std1
        #normal1
        a1 = (tau_min - lesser_mean) / lesser_std
        b1 = (tau_change - lesser_mean) / lesser_std

        tau_array[tauindices[0]:tauindices[1]] = torch.tensor(truncnorm.rvs(a1, b1, loc=lesser_mean, scale=lesser_std, size=tausizes[0]))

        #normal2
        a2 = (tau_change - greater_mean) / greater_std
        b2 = (tau_max - greater_mean) / greater_std
        tau_array[tauindices[1]:tauindices[2]] = torch.tensor(truncnorm.rvs(a2, b2, loc=greater_mean, scale=greater_std, size=tausizes[1]))

        return tau_array

    else:
        raise ValueError("Unsupported tau distribution type.")

def main(config, seed, name):
    with open(config, 'r') as file:
        conf = yaml.safe_load(file)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = torch.Generator().manual_seed(seed)
    savepath = 'rnn_results/' + name + '_' + str(seed) + '/'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    
    # Save a copy of the configuration file to the savepath directory for reproducibility
    with open(os.path.join(savepath, 'config.yaml'), 'w') as outfile:
        yaml.dump(conf, outfile, default_flow_style=False)

    print(f'Training with device: {device}')

    if conf['expt']['type'] == 'instant_DM':
        alldata = datasets.decisionMakingInstant(seed, conf['expt']['stim_start_min'], conf['expt']['stim_start_max'], conf['expt']['stim_length'], conf['expt']['decision_length'], conf['expt']['sigma_length'], conf['expt']['duration'], conf['model']['dt'], conf['training']['size'])
    elif conf['expt']['type'] == 'hold_DM':
        alldata = datasets.decisionMakingHold(seed, conf['expt']['stim_start_min'], conf['expt']['stim_start_max'], conf['expt']['stim_length'], conf['expt']['sigma_length'], conf['expt']['duration'], conf['model']['dt'], conf['training']['size'])
    elif conf['expt']['type'] == 'delay_DM':
        alldata = datasets.decisionMakingDelay(seed, conf['expt']['stim_start_min'], conf['expt']['stim_start_max'], conf['expt']['stim_length'], conf['expt']['delay_length'], conf['expt']['sigma_length'], conf['expt']['sigma_delay'], conf['expt']['duration'], conf['model']['dt'], conf['training']['size'])    
    elif conf['expt']['type'] == 'PC':
        alldata = datasets.perceptualClassification(seed, conf['expt']['num_cohs'], conf['expt']['stim_start_min'], conf['expt']['stim_start_max'], conf['expt']['coh_radius'], conf['expt']['symmetric'], conf['expt']['duration'], conf['model']['dt'], conf['training']['size'])
    elif conf['expt']['type'] == 'review_task':
        alldata = datasets.reviewTask(seed, conf['expt']['num_values'], conf['expt']['stim_start_min'], conf['expt']['stim_start_max'], conf['expt']['stim_length'], conf['expt']['value_min'], conf['expt']['value_max'], conf['expt']['delay'], conf['expt']['delay_sigma'], conf['expt']['stim_sigma'], conf['expt']['duration'], conf['model']['dt'], conf['training']['size'])
    elif conf['expt']['type'] == 'raiyyan_task':
        alldata = datasets.NonstationaryRewardDataset(tau_map=conf['expt']['tau_map'],
                                                      n_cue_events=conf['expt']['n_cue_events'],
                                                      cue_length=conf['expt']['cue_length'],
                                                      seed=seed)
    #put other datasets here 

    traindata, validdata = random_split(alldata, conf['training']['valid_split'], generator=generator)
    trainloader = DataLoader(traindata, batch_size=conf['training']['batch_size'], shuffle=True, num_workers=0)
    valloader = DataLoader(validdata, batch_size=conf['training']['batch_size'], shuffle=True, num_workers=0)

    #Save dataset?

    if conf['model']['type'] == 'RNN':
        model = models.RNN(conf['model']['input_size'], conf['model']['hidden_size'], conf['model']['output_size'], conf['model']['dt'], conf['model']['tau'], conf['model']['activation'], conf['model']['bias'], conf['model']['sigma_in'], conf['model']['sigma_re'])
        optimizer = optim.Adam(model.parameters(), lr=conf['training']['learning_rate'])
        loss_fxn = nn.MSELoss()
        model, val_losses, losses, wcn = trainer.train_RNN(model, trainloader, valloader, optimizer, loss_fxn, conf, device, generator, savepath)
    if conf['model']['type'] == 'EI_RNN':
        model = models.EI_RNN(conf['model']['input_size'], conf['model']['hidden_size'], conf['model']['output_size'], conf['model']['dt'], conf['model']['tau'], conf['model']['activation'], conf['model']['bias'], conf['model']['sigma_in'], conf['model']['sigma_re'], conf['model']['eprop'])
        optimizer = optim.Adam(model.parameters(), lr=conf['training']['learning_rate'])
        loss_fxn = nn.MSELoss()
        model, val_losses, losses, wcn = trainer.train_RNN(model, trainloader, valloader, optimizer, loss_fxn, conf, device, generator, savepath)
    if conf['model']['type'] == 'MultiTauRNN':
        if conf['model']['tau_distribution'] == 'groups':
            tau_array = get_tau_array(conf['model']['tau_distribution'], conf['model']['hidden_size'], device, tau_groups=conf['model']['tau_groups'], tau_proportions=conf['model']['tau_proportions'])
        elif conf['model']['tau_distribution'] == 'uniform':
            tau_array = get_tau_array(conf['model']['tau_distribution'], conf['model']['hidden_size'], device, tau_min=conf['model']['tau_min'], tau_max=conf['model']['tau_max'])
        elif conf['model']['tau_distribution'] == 'bimodal_normal':
            tau_array = get_tau_array(conf['model']['tau_distribution'], conf['model']['hidden_size'], device, tau_proportions=conf['model']['tau_proportions'], tau_mean1=conf['model']['tau_mean1'], tau_mean2=conf['model']['tau_mean2'], tau_std1=conf['model']['tau_std1'], tau_std2=conf['model']['tau_std2'], tau_min=conf['model']['tau_min'], tau_max=conf['model']['tau_max'], tau_change=conf['model']['tau_change'])
        else:
            raise ValueError("Unsupported tau distribution type.")
        #save tau_array to savepath
        #np.save(os.path.join(savepath, 'tau_array.npy'), tau_array.cpu().numpy())
        model = models.MultiTauRNN(conf['model']['input_size'], conf['model']['hidden_size'], conf['model']['output_size'], conf['model']['dt'], tau_array, conf['model']['activation'], conf['model']['bias'], conf['model']['sigma_in'], conf['model']['sigma_re'])
        optimizer = optim.Adam(model.parameters(), lr=conf['training']['learning_rate'])
        loss_fxn = nn.MSELoss()
        model, val_losses, losses, wcn = trainer.train_RNN(model, trainloader, valloader, optimizer, loss_fxn, conf, device, generator, savepath)
    if conf['model']['type'] == 'expirimental_RNN':
        if conf['model']['tau_distribution'] == 'groups':
            tau_array = get_tau_array(conf['model']['tau_distribution'], conf['model']['hidden_size'], device, tau_groups=conf['model']['tau_groups'], tau_proportions=conf['model']['tau_proportions'])
        elif conf['model']['tau_distribution'] == 'uniform':
            tau_array = get_tau_array(conf['model']['tau_distribution'], conf['model']['hidden_size'], device, tau_min=conf['model']['tau_min'], tau_max=conf['model']['tau_max'])
        elif conf['model']['tau_distribution'] == 'bimodal_normal':
            tau_array = get_tau_array(conf['model']['tau_distribution'], conf['model']['hidden_size'], device, tau_proportions=conf['model']['tau_proportions'], tau_mean1=conf['model']['tau_mean1'], tau_mean2=conf['model']['tau_mean2'], tau_std1=conf['model']['tau_std1'], tau_std2=conf['model']['tau_std2'], tau_min=conf['model']['tau_min'], tau_max=conf['model']['tau_max'], tau_change=conf['model']['tau_change'])
        else:
            raise ValueError("Unsupported tau distribution type.")
        #save tau_array to savepath
        #np.save(os.path.join(savepath, 'tau_array.npy'), tau_array.cpu().numpy())
        model = models.expirimental_RNN(conf['model']['input_size'], conf['model']['hidden_size'], conf['model']['output_size'], conf['model']['dt'], tau_array, conf['model']['activation'], conf['model']['tau_effect'], conf['model']['bias'], conf['model']['sigma_in'], conf['model']['sigma_re'])
        optimizer = optim.Adam(model.parameters(), lr=conf['training']['learning_rate'])
        loss_fxn = nn.MSELoss()
        model, val_losses, losses, wcn = trainer.train_RNN(model, trainloader, valloader, optimizer, loss_fxn, conf, device, generator, savepath)
    if conf['model']['type'] == 'lowrank_expirimental_RNN':
        if conf['model']['tau_distribution'] == 'groups':
            tau_array = get_tau_array(conf['model']['tau_distribution'], conf['model']['hidden_size'], device, tau_groups=conf['model']['tau_groups'], tau_proportions=conf['model']['tau_proportions'])
        elif conf['model']['tau_distribution'] == 'uniform':
            tau_array = get_tau_array(conf['model']['tau_distribution'], conf['model']['hidden_size'], device, tau_min=conf['model']['tau_min'], tau_max=conf['model']['tau_max'])
        elif conf['model']['tau_distribution'] == 'bimodal_normal':
            tau_array = get_tau_array(conf['model']['tau_distribution'], conf['model']['hidden_size'], device, tau_proportions=conf['model']['tau_proportions'], tau_mean1=conf['model']['tau_mean1'], tau_mean2=conf['model']['tau_mean2'], tau_std1=conf['model']['tau_std1'], tau_std2=conf['model']['tau_std2'], tau_min=conf['model']['tau_min'], tau_max=conf['model']['tau_max'], tau_change=conf['model']['tau_change'])
        else:
            raise ValueError("Unsupported tau distribution type.")
        #save tau_array to savepath
        #np.save(os.path.join(savepath, 'tau_array.npy'), tau_array.cpu().numpy())
        model = models.lowrank_expirimental_RNN(conf['model']['input_size'], conf['model']['hidden_size'], conf['model']['output_size'], conf['model']['rank'], conf['model']['dt'], tau_array, conf['model']['activation'], conf['model']['tau_effect'], conf['model']['bias'], conf['model']['sigma_in'], conf['model']['sigma_re'])
        optimizer = optim.Adam(model.parameters(), lr=conf['training']['learning_rate'])
        loss_fxn = nn.MSELoss()
        model, val_losses, losses, wcn = trainer.train_RNN(model, trainloader, valloader, optimizer, loss_fxn, conf, device, generator, savepath)
    #add other models here

    #some plotting here
    plotting_data = next(iter(DataLoader(validdata, batch_size=10, shuffle=True, num_workers=0)))
    analysis.plot_example(model, plotting_data, device, generator, savepath, noise=False)

    # Save training and validation losses
    np.save(os.path.join(savepath, 'train_losses.npy'), np.array(losses))
    np.save(os.path.join(savepath, 'val_losses.npy'), np.array(val_losses))
    np.save(os.path.join(savepath, 'weight_change_norm.npy'), np.array(wcn))

if __name__ == '__main__':
    fire.Fire(main)

    
##########################
### python run_expt.py --config='configs/test_conf.yaml' --seed=0 --name='test'
