#!/bin/bash
#SBATCH --job-name=rnn_train

# No longer using GPU
## SBATCH --partition=gpu
## SBATCH --gres=gpu:v100x:1
## SBATCH --cpus-per-task=14

#SBATCH --mem=32g
#SBATCH --time=24:00:00
#SBATCH --array=0-799
#SBATCH -o logs/tmp_%A_%a.out
#SBATCH -e logs/tmp_%A_%a.err

source activate py312p9

CONFIGS=(configs/cshl2026/kerneltau*.yaml)
N_SEEDS=100

CONFIG_IDX=$((SLURM_ARRAY_TASK_ID / N_SEEDS))
SEED=$((SLURM_ARRAY_TASK_ID % N_SEEDS))
CONFIG=${CONFIGS[$CONFIG_IDX]}

BASENAME=$(basename $CONFIG .yaml)
DELAY=$(echo $BASENAME | grep -oP 'kerneltau\K[0-9]*\.?[0-9]+')
TAU=$(echo $BASENAME | grep -oP 'taueffect\K(decay|all)')
DIST=$(echo $BASENAME | grep -oP 'dist\K(uniform|groups)')
NAME="kerneltau${DELAY}_taueffect${TAU}_dist${DIST}_seed${SEED}"

mv logs/tmp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out logs/${NAME}.out
mv logs/tmp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err logs/${NAME}.err

echo "Config: $CONFIG | Seed: $SEED | Name: $NAME"
python -u run_expt_pep8.py --config=$CONFIG --seed=$SEED --name=$NAME