#!/bin/bash
#SBATCH --job-name=rnn_train
#SBATCH --mem=10g
#SBATCH --time=72:00:00
#SBATCH --array=0-999
#SBATCH -o logs/tmp_%A_%a.out
#SBATCH -e logs/tmp_%A_%a.err

source activate py312p9

mapfile -t CONFIGS < <(printf '%s\n' configs/cshl2026/ktau*_delay*_uniform*.yaml \
                                     configs/cshl2026/ktau*_delay*_vanilla*.yaml \
                                     configs/cshl2026/ktau*_delay*_groups*.yaml | sort)
N_SEEDS=30
OFFSET=${1:-0}

CONFIG_IDX=$(( (SLURM_ARRAY_TASK_ID + OFFSET) / N_SEEDS ))
SEED=$(( (SLURM_ARRAY_TASK_ID + OFFSET) % N_SEEDS ))
CONFIG=${CONFIGS[$CONFIG_IDX]}

BASENAME=$(basename $CONFIG .yaml)
KERNEL_TAU=$(echo $BASENAME | grep -oP 'ktau\K[0-9]*\.?[0-9]+')
DELAY=$(echo $BASENAME      | grep -oP 'delay\K[0-9]*\.?[0-9]+')
TAU_EFFECT=$(echo $BASENAME | grep -oP '(all|decay)$')

if echo $BASENAME | grep -q 'uniform'; then
    UNIFORM_TAU=$(echo $BASENAME | grep -oP 'uniform\K[0-9]*\.?[0-9]+')
    NAME="ktau${KERNEL_TAU}_delay${DELAY}_uniform${UNIFORM_TAU}_${TAU_EFFECT}_seed${SEED}"
elif echo $BASENAME | grep -q 'groups'; then
    GROUPS_TAU=$(echo $BASENAME | grep -oP 'groups\K[0-9]*\.?[0-9]+')
    PROP=$(echo $BASENAME       | grep -oP 'prop\K[0-9]*\.?[0-9]+')
    NAME="ktau${KERNEL_TAU}_delay${DELAY}_groups${GROUPS_TAU}_prop${PROP}_${TAU_EFFECT}_seed${SEED}"
else
    VANILLA_TAU=$(echo $BASENAME | grep -oP 'vanilla\K[0-9]*\.?[0-9]+')
    NAME="ktau${KERNEL_TAU}_delay${DELAY}_vanilla${VANILLA_TAU}_${TAU_EFFECT}_seed${SEED}"
fi

mv logs/tmp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out logs/${NAME}.out
mv logs/tmp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err logs/${NAME}.err

echo "Config: $CONFIG | Seed: $SEED | Name: $NAME"
python -u run_expt.py --config=$CONFIG --seed=$SEED --name=$NAME