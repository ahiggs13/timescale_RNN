import yaml
import numpy as np
import itertools
from pathlib import Path

# Load base config
with open("configs/base.yaml") as f:
    base = yaml.safe_load(f)

# Parameters
ranks = list(range(1, 11))
tau_distributions = ["bimodal_normal", "uniform", "groups"]
tau_effects = ["all", "recur", "decay", "input"]

# Save directory
outdir = Path("configs2")
outdir.mkdir(exist_ok=True)

def make_config(base, model, rank=None, tau_dist=None, tau_eff=None):
    cfg = yaml.safe_load(yaml.dump(base))  # deep copy
    cfg["model"]["name"] = model
    if rank is not None:
        cfg["model"]["rank"] = rank
    if tau_dist is not None:
        cfg["model"]["tau_distribution"] = tau_dist
    if tau_eff is not None:
        cfg["model"]["tau_effect"] = tau_eff
    return cfg

def make_vanilla(base, model, rank, tau_eff):
    cfg = yaml.safe_load(yaml.dump(base))
    cfg["model"]["name"] = model
    cfg["model"]["rank"] = rank
    cfg["model"]["tau_distribution"] = "groups"
    cfg["model"]["tau_groups"] = [1]
    cfg["model"]["tau_proportions"] = [1]
    cfg["model"]["tau_effect"] = tau_eff
    return cfg

# Low-rank configs (rank × tau_dist × tau_eff)
for rank, tau_dist, tau_eff in itertools.product(ranks, tau_distributions, tau_effects):
    cfg = make_config(base, "lowrank_expirimental_RNN", rank, tau_dist, tau_eff)
    fname = outdir / f"lowrank_rank{rank}_{tau_dist}_{tau_eff}.yaml"
    with open(fname, "w") as f:
        yaml.dump(cfg, f)

# Vanilla low-rank configs (rank × tau_eff)
for rank, tau_eff in itertools.product(ranks, tau_effects):
    cfg = make_vanilla(base, "lowrank_expirimental_RNN", rank, tau_eff)
    fname = outdir / f"lowrank_rank{rank}_vanilla_{tau_eff}.yaml"
    with open(fname, "w") as f:
        yaml.dump(cfg, f)

# Unconstrained experimental RNN configs (tau_dist × tau_eff, no rank)
for tau_dist, tau_eff in itertools.product(tau_distributions, tau_effects):
    cfg = make_config(base, "expirimental_RNN", None, tau_dist, tau_eff)
    fname = outdir / f"unconstrained_{tau_dist}_{tau_eff}.yaml"
    with open(fname, "w") as f:
        yaml.dump(cfg, f)

print(f"Configs written to {outdir}")
