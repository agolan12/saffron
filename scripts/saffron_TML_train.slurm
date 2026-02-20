#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --gpus-per-node=2080ti:1
#SBATCH --mem=20G
#SBATCH --time=2:00:00
#SBATCH --output=output_balanced_data.out
#SBATCH --error=error_balanced_data.err

# Load necessary modules (optional, if needed)
# module load cuda/11.7

echo 'sourcing bashrc'
source ~/.bashrc

# Run your command or script here
echo "Job started on $(hostname)"
# For example, run a Python script:
# python my_script.py

# Change to project directory
cd /gscratch/cheme/agolan/saffron

export PYTHONPATH="/gscratch/cheme/agolan/saffron/src:$PYTHONPATH"

/gscratch/cheme/agolan/phys_417_env/bin/python scripts/test_model_local.py \
    --epochs=10 --batch_size=32