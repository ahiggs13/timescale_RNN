import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
import yaml
import fire 

from src import analysis, datasets, models, trainer

def main(config, seed, name):
    with open(config, 'r') as file:
        conf = yaml.safe_load(file)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = torch.Generator().manual_seed(seed)
    savepath = name + '_' + str(seed) + '/'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    
    # Save a copy of the configuration file to the savepath directory for reproducibility
    with open(os.path.join(savepath, 'config.yaml'), 'w') as outfile:
        yaml.dump(conf, outfile, default_flow_style=False)

    print(f'Training with device: {device}')

    if conf['expt']['type'] == 'instant_DM':
        alldata = datasets.decisionMakingInstant(seed, conf['expt']['stim_start_min'], conf['expt']['stim_start_max'], conf['expt']['stim_length'], conf['expt']['decision_length'], conf['expt']['sigma_length'], conf['expt']['duration'], conf['model']['dt'], conf['training']['size'])
    #put other datasets here 

    traindata, validdata = random_split(alldata, conf['training']['valid_split'], generator=generator)
    trainloader = DataLoader(traindata, batch_size=conf['training']['batch_size'], shuffle=True, num_workers=0)
    valloader = DataLoader(validdata, batch_size=conf['training']['batch_size'], shuffle=True, num_workers=0)

    #Save dataset?

    if conf['model']['type'] == 'RNN':
        model = models.RNN(conf['model']['input_size'], conf['model']['hidden_size'], conf['model']['output_size'], conf['model']['dt'], conf['model']['tau'], conf['model']['activation'], conf['model']['bias'], conf['model']['sigma_in'], conf['model']['sigma_re'])
        optimizer = optim.Adam(model.parameters(), lr=conf['training']['learning_rate'])
        loss_fxn = nn.MSELoss()
        model, val_losses = trainer.train_RNN(model, trainloader, valloader, optimizer, loss_fxn, conf, device, generator, savepath)
    #add other models here

    #some plotting here
    plotting_data = next(iter(DataLoader(validdata, batch_size=10, shuffle=True, num_workers=0)))
    analysis.plot_example(model, plotting_data, device, generator, savepath)

if __name__ == '__main__':
    fire.Fire(main)

    
##########################
### python run_expt.py --config='configs/test_conf.yaml' --seed=0 --name='test'
