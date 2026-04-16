#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --gpus-per-node=2080ti:1
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --output=logs_output/supconloss_%j.out
#SBATCH --error=logs_error/supconloss_%j.err

# Load necessary modules (optional, if needed)
# module load cuda/11.7

echo 'sourcing bashrc'
source ~/.bashrc
conda activate contrastive_ml

# Run your command or script here
echo "Job started on $(hostname)"
# For example, run a Python script:
# python my_script.py

# Change to project directory
cd /gscratch/cheme/agolan/saffron

python scripts/supconloss_test_old.py \
    --epochs=100 --batch_size=128
