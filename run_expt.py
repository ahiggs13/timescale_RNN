import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
matplotlib.pyplot as plt
import yaml
import fire 

from src import analysis, datasets, models, trainer

def main(config, seed):
    with open(config, 'r') as file:
        conf = yaml.safe_load(file)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = torch.Generator().manual_seed(seed)

    print(f'Training with device: {device}')

    if conf['expt']['data'] == 'instant_DM':
        alldata = datasets.decisionMakingInstant(seed, conf['expt']['stim_start_min'], conf['expt']['stim_start_max'], conf['expt']['stim_length'], conf['expt']['decision_length'], conf['expt']['sigma_length'], conf['expt']['duration'], conf['model']['dt'], conf['training']['size'])
    #put other datasets here 

    traindata, validdata = random_split(alldata, conf['training']['valid_split'], generator=generator)
    trainloader = DataLoader(traindata, batch_size=conf['training']['batch_size'], shuffle=True, num_workers=0)
    valloader = DataLoader(validdata, batch_size=conf['training']['batch_size'], shuffle=True, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=conf['training']['learning_rate'])
    loss_fxn = nn.MSELoss()

    if conf['model']['type'] == 'RNN':
        model = models.RNN(conf['model']['input_size'], conf['model']['hidden_size'], conf['model']['output_size'], conf['model']['dt'], conf['model']['tau'], conf['model']['activation'], conf['model']['bias'], conf['model']['sigma_in'], conf['model']['sigma_re'])
        model, val_losses = trainer.train_RNN(model, trainloader, valloader, optimizer, loss_fxn, conf, device, seed)
    #add other models here

    #some plotting here


if __name__ == '__main__':
    fire.Fire(main)

    

    
