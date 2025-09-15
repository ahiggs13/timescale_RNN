#!/bin/bash
mkdir -p logs

for cfg in real_expt3_configs/*.yaml; do
    BASENAME=$(basename $cfg .yaml)

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$BASENAME
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/${BASENAME}_%j.out
#SBATCH --error=logs/${BASENAME}_%j.err

# Activate conda environment
source activate py312p9

# Run experiment
python -u run_expt.py --config="$cfg" --seed=0 --name="$BASENAME"
EOT

done