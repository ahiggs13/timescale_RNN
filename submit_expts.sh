#!/bin/bash
#SBATCH --job-name=rnn_train
#SBATCH --mem=10g
#SBATCH --time=120:00:00
#SBATCH --array=0-999
#SBATCH -o logs/tmp_%A_%a.out
#SBATCH -e logs/tmp_%A_%a.err

source activate py312p9

OFFSET=0
DIR="debugging"

while [[ $# -gt 0 ]]; do
    case $1 in
        --offset) OFFSET=$2; shift 2 ;;
        --dir)    DIR=$2;    shift 2 ;;
        *) shift ;;
    esac
done

mapfile -t CONFIGS < <(printf '%s\n' configs/${DIR}/autocorr*_ktau*_delay*_normal*.yaml \
                                     configs/${DIR}/autocorr*_ktau*_delay*_vanilla*.yaml \
                                     configs/${DIR}/autocorr*_ktau*_delay*_groups*.yaml | sort)
N_SEEDS=1

CONFIG_IDX=$(( (SLURM_ARRAY_TASK_ID + OFFSET) / N_SEEDS ))
SEED=$(( (SLURM_ARRAY_TASK_ID + OFFSET) % N_SEEDS + 20 ))
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

mkdir -p logs/${DIR}
mv logs/tmp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out logs/${DIR}/${NAME}.out
mv logs/tmp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err logs/${DIR}/${NAME}.err

echo "Config: $CONFIG | Seed: $SEED | Name: $NAME"
python -u run_expt.py --config=$CONFIG --seed=$SEED --name=$NAME --output_dir=$DIR

# sbatch submit_expts.sh --offset 0 --dir debugging
# sbatch submit_expts.sh --offset 1000 --dir debugging
# sbatch submit_expts.sh --offset 2000 --dir debugging
# sbatch --array=0-599 submit_expts.sh --offset 3000 --dir debugging
# i.e. for 3600 total jobs