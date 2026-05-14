import os
from itertools import accumulate

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import fire
from torch.utils.data import DataLoader, random_split
from scipy.stats import gamma, truncnorm

from src import analysis, datasets, models, trainer


# ---------------------------------------------------------------------------
# Tau array construction
# ---------------------------------------------------------------------------

def get_tau_array(
    distribution,
    hidden_size,
    device,
    tau_groups=None,
    tau_proportions=None,
    tau_min=None,
    tau_max=None,
    tau_mean1=None,
    tau_mean2=None,
    tau_std1=None,
    tau_std2=None,
    tau_change=None,
    eta=None
):
    """Build a tau array according to the specified distribution."""
    if distribution == "groups":
        tau_array = torch.zeros(hidden_size, dtype=torch.float32, device=device)
        tau_sizes = [int(p * hidden_size) for p in tau_proportions]
        tau_indices = [0] + list(accumulate(tau_sizes))
        for i, tau in enumerate(tau_groups):
            tau_array[tau_indices[i]:tau_indices[i + 1]] = tau
        return tau_array

    if distribution == "uniform":
        if tau_min is None or tau_max is None:
            raise ValueError(
                "tau_min and tau_max must be provided for uniform distribution."
            )
        return torch.FloatTensor(hidden_size).uniform_(tau_min, tau_max).to(device)

    if distribution == "bimodal_normal":
        tau_array = torch.zeros(hidden_size, dtype=torch.float32, device=device)
        tau_sizes = [int(p * hidden_size) for p in tau_proportions]
        tau_indices = [0] + list(accumulate(tau_sizes))

        # Order the two modes by mean value
        if tau_mean1 <= tau_mean2:
            lesser_mean, greater_mean = tau_mean1, tau_mean2
            lesser_std, greater_std = tau_std1, tau_std2
        else:
            lesser_mean, greater_mean = tau_mean2, tau_mean1
            lesser_std, greater_std = tau_std2, tau_std1

        # Lower mode (truncated below tau_change)
        a1 = (tau_min - lesser_mean) / lesser_std
        b1 = (tau_change - lesser_mean) / lesser_std
        tau_array[tau_indices[0]:tau_indices[1]] = torch.tensor(
            truncnorm.rvs(a1, b1, loc=lesser_mean, scale=lesser_std, size=tau_sizes[0])
        )

        # Upper mode (truncated above tau_change)
        a2 = (tau_change - greater_mean) / greater_std
        b2 = (tau_max - greater_mean) / greater_std
        tau_array[tau_indices[1]:tau_indices[2]] = torch.tensor(
            truncnorm.rvs(a2, b2, loc=greater_mean, scale=greater_std, size=tau_sizes[1])
        )

        return tau_array
    
    if distribution == 'normal':
        return torch.exp(torch.normal(mean=tau_mean1, std=tau_std1, size=(hidden_size,))*eta).to(device)

    raise ValueError(f"Unsupported tau distribution type: '{distribution}'.")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(conf, seed):
    """Instantiate the dataset specified in the experiment config."""
    expt = conf["expt"]
    expt_type = expt["type"]

    if expt_type == "instant_DM":
        return datasets.decisionMakingInstant(
            seed,
            expt["stim_start_min"], expt["stim_start_max"],
            expt["stim_length"], expt["decision_length"],
            expt["sigma_length"], expt["duration"],
            conf["model"]["dt"], conf["training"]["size"],
        )
    
    if expt_type == "hold_DM":
        return datasets.decisionMakingHold(
            seed,
            expt["stim_start_min"], expt["stim_start_max"],
            expt["stim_length"], expt["sigma_length"],
            expt["duration"], conf["model"]["dt"], conf["training"]["size"],
        )
    
    if expt_type == "delay_DM":
        return datasets.decisionMakingDelay(
            seed,
            expt["stim_start_min"], expt["stim_start_max"],
            expt["stim_length"], expt["delay_length"],
            expt["sigma_length"], expt["sigma_delay"],
            expt["duration"], conf["model"]["dt"], conf["training"]["size"],
        )
    
    if expt_type == "PC":
        return datasets.perceptualClassification(
            seed,
            expt["num_cohs"],
            expt["stim_start_min"], expt["stim_start_max"],
            expt["coh_radius"], expt["symmetric"],
            expt["duration"], conf["model"]["dt"], conf["training"]["size"],
        )
    
    if expt_type == "review_task":
        return datasets.reviewTask(
            seed,
            expt["num_values"],
            expt["stim_start_min"], expt["stim_start_max"],
            expt["stim_length"], expt["value_min"], expt["value_max"],
            expt["delay"], expt["delay_sigma"], expt["stim_sigma"],
            expt["duration"], conf["model"]["dt"], conf["training"]["size"],
        )
    
    if expt_type == "integration_task":
        return datasets.NonstationaryRewardDelayDataset(
            seed,
            kernel_tau=expt["kernel_tau"], read_delay=expt["read_delay"], autocorr=expt["autocorr"],
            noise_std=expt['noise_std'],  zero_integration=expt['zero_integration'],
        )
    
    if expt_type == 'delay_discrimination':
        return datasets.DelayDiscrimination(
            seed,
            interval_length=expt["interval_length"], stim_length=expt["stim_length"],
            stim_strengths=expt["stim_strengths"], stim_std=expt["stim_std"],
            stim_cue_type=expt["stim_cue_type"], stim_cue_strength=expt["stim_cue_strength"],
            output_type=expt["output_type"],
            duration=expt["duration"], dt=conf["model"]["dt"], size=conf["training"]["size"],
        )

    raise ValueError(f"Unsupported experiment type: '{expt_type}'.")


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _build_tau_array(conf, device):
    """Helper: build the tau array from model config."""
    distribution = conf["model"]["tau_distribution"]
    hidden_size = conf["model"]["hidden_size"]
    m = conf["model"]

    if distribution == "groups":
        return get_tau_array(
            distribution, hidden_size, device,
            tau_groups=m["tau_groups"],
            tau_proportions=m["tau_proportions"],
        )
    if distribution == "uniform":
        return get_tau_array(
            distribution, hidden_size, device,
            tau_min=m["tau_min"],
            tau_max=m["tau_max"],
        )
    if distribution == "bimodal_normal":
        return get_tau_array(
            distribution, hidden_size, device,
            tau_proportions=m["tau_proportions"],
            tau_mean1=m["tau_mean1"], tau_mean2=m["tau_mean2"],
            tau_std1=m["tau_std1"], tau_std2=m["tau_std2"],
            tau_min=m["tau_min"], tau_max=m["tau_max"],
            tau_change=m["tau_change"],
        )
    if distribution == "normal":
        return get_tau_array(
            distribution, hidden_size, device,
            tau_mean1=m["tau_mean1"], tau_std1=m["tau_std1"],
            eta=m["eta"],
        )
    
    raise ValueError(f"Unsupported tau distribution type: '{distribution}'.")


def build_model(conf, device):
    m = conf["model"]
    model_type = m["type"]

    if model_type == "RNN":
        model = models.RNN(
            m["input_size"], m["hidden_size"], m["output_size"],
            m["dt"], m["tau"], m["activation"], m["bias"],
            m["sigma_in"], m["sigma_re"],
        )

    elif model_type == "EI_RNN":
        model = models.EI_RNN(
            m["input_size"], m["hidden_size"], m["output_size"],
            m["dt"], m["tau"], m["activation"], m["bias"],
            m["sigma_in"], m["sigma_re"], m["eprop"],
        )

    else:
        tau_array = _build_tau_array(conf, device)

        if model_type == "MultiTauRNN":
            model = models.MultiTauRNN(
                m["input_size"], m["hidden_size"], m["output_size"],
                m["dt"], tau_array, m["activation"], m["bias"],
                m["sigma_in"], m["sigma_re"],
            )
        elif model_type == "expirimental_RNN":
            model = models.expirimental_RNN(
                m["input_size"], m["hidden_size"], m["output_size"],
                m["dt"], tau_array, m["activation"], m["tau_effect"],
                m["bias"], m["sigma_in"], m["sigma_re"],
            )
        elif model_type == "lowrank_expirimental_RNN":
            model = models.lowrank_expirimental_RNN(
                m["input_size"], m["hidden_size"], m["output_size"],
                m["rank"], m["dt"], tau_array, m["activation"],
                m["tau_effect"], m["bias"], m["sigma_in"], m["sigma_re"],
            )
        elif model_type == "TwoCompartmentRNN":
            model = models.TwoCompartmentRNN(
                m["input_size"], m["hidden_size"], m["output_size"],
                m["dt"], tau_array, tau_array, 1, m["activation"],
                m["bias"], m["sigma_in"], m["sigma_re"],
            )
        elif model_type =="expirimental_RNN_sigmainit":
            model = models.expirimental_RNN_sigmainit(
                m["input_size"], m["hidden_size"], m["output_size"],
                m["dt"], tau_array, m["activation"], m["tau_effect"],
                m["bias"], m["sigma_in"], m["sigma_re"], m["sigma"]
            )
        else:
            raise ValueError(f"Unsupported model type: '{model_type}'.")

    return model.to(device)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config, seed, name, output_dir, quick_test=False):
    with open(config, "r") as fh:
        conf = yaml.safe_load(fh)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(seed)
 
    savepath = os.path.join(f"rnn_results",f"{output_dir}", f"{name}", "") 
    os.makedirs(savepath, exist_ok=True)
 
    # Persist config alongside results for reproducibility
    with open(os.path.join(savepath, "config.yaml"), "w") as fh:
        yaml.dump(conf, fh, default_flow_style=False)
 
    print(f"Training with device: {device}")
 
    # Dataset
    all_data = load_dataset(conf, seed)
    train_data, valid_data = random_split(
        all_data, conf["training"]["valid_split"], generator=generator
    )
    train_loader = DataLoader(
        train_data, batch_size=conf["training"]["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        valid_data, batch_size=conf["training"]["batch_size"], shuffle=True, num_workers=0
    )
 
    # Model, optimiser, loss
    model = build_model(conf, device)
    optimizer = optim.Adam(model.parameters(), lr=conf["training"]["learning_rate"])
    loss_fn = nn.MSELoss()
 
    model, val_losses, train_losses, wcn = trainer.train_RNN(
        model, train_loader, val_loader, optimizer, loss_fn,
        conf, device, generator, savepath, quick_test=quick_test
    )
 
    # Plot a small validation batch
    plot_batch = next(iter(DataLoader(valid_data, batch_size=10, shuffle=True, num_workers=0)))
    analysis.plot_example(model, plot_batch, device, generator, savepath, noise=False)
 
 
if __name__ == "__main__":
    fire.Fire(main)

# Usage:
# python -u run_expt.py --config='configs/uniform.yaml' --seed=0 --name='uniform' --output_dir='test'