#!/bin/bash
mkdir -p logs

# Grab the first config file
cfg=$(ls real_expt3_configs/*.yaml | head -n 1)
BASENAME=$(basename "$cfg" .yaml)

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${BASENAME}_test
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=01:00:00
#SBATCH --output=logs/${BASENAME}_test_%j.out
#SBATCH --error=logs/${BASENAME}_test_%j.err

# Activate conda environment
source activate py312p9

# Run experiment (shorter test run)
python -u run_expt.py --config="$cfg" --seed=0 --name="${BASENAME}_test"
EOT