#!/bin/bash
#SBATCH --job-name=rnn_train
#SBATCH --mem=10g
#SBATCH --time=72:00:00
#SBATCH --array=0-999
#SBATCH -o logs/tmp_%A_%a.out
#SBATCH -e logs/tmp_%A_%a.err

source activate py312p9

mapfile -t CONFIGS < <(printf '%s\n' configs/cshl2026/autocorr*_ktau*_delay*_normal*.yaml \
                                     configs/cshl2026/autocorr*_ktau*_delay*_vanilla*.yaml \
                                     configs/cshl2026/autocorr*_ktau*_delay*_groups*.yaml | sort)
N_SEEDS=20
OFFSET=${1:-0}

CONFIG_IDX=$(( (SLURM_ARRAY_TASK_ID + OFFSET) / N_SEEDS ))
SEED=$(( (SLURM_ARRAY_TASK_ID + OFFSET) % N_SEEDS ))
CONFIG=${CONFIGS[$CONFIG_IDX]}

BASENAME=$(basename $CONFIG .yaml)
AUTOCORR=$(echo $BASENAME  | grep -oP 'autocorr\K[0-9]+\.?[0-9]*')
KERNEL_TAU=$(echo $BASENAME | grep -oP 'ktau\K[0-9]+\.?[0-9]*')
DELAY=$(echo $BASENAME      | grep -oP 'delay\K[0-9]+\.?[0-9]*')

if echo $BASENAME | grep -q 'normal'; then
    ETA=$(echo $BASENAME | grep -oP 'eta\K[0-9]+\.?[0-9]*')
    NAME="autocorr${AUTOCORR}_ktau${KERNEL_TAU}_delay${DELAY}_normal_eta${ETA}_seed${SEED}"
elif echo $BASENAME | grep -q 'groups'; then
    GROUPS_TAU=$(echo $BASENAME | grep -oP 'groups\K[0-9]+\.?[0-9]*')
    PROP=$(echo $BASENAME       | grep -oP 'prop\K[0-9]+\.?[0-9]*')
    NAME="autocorr${AUTOCORR}_ktau${KERNEL_TAU}_delay${DELAY}_groups${GROUPS_TAU}_prop${PROP}_seed${SEED}"
else
    VANILLA_TAU=$(echo $BASENAME | grep -oP 'vanilla\K[0-9]+\.?[0-9]*')
    NAME="autocorr${AUTOCORR}_ktau${KERNEL_TAU}_delay${DELAY}_vanilla${VANILLA_TAU}_seed${SEED}"
fi

mv logs/tmp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out logs/${NAME}.out
mv logs/tmp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err logs/${NAME}.err

echo "Config: $CONFIG | Seed: $SEED | Name: $NAME"
python -u run_expt.py --config=$CONFIG --seed=$SEED --name=$NAME

# sbatch submit_expts.sh 0
# sbatch submit_expts.sh 1000
# sbatch submit_expts.sh 2000
# sbatch submit_expts.sh 3000