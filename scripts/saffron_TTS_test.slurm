#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --gpus-per-node=2080ti:1
#SBATCH --mem=10G
#SBATCH --time=2:00:00
#SBATCH --output=output_1.out
#SBATCH --error=error_1.err

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

/gscratch/cheme/agolan/phys_417_env/bin/python scripts/train_test_split.py \
       	--input_dir	/gscratch/cheme/agolan/data/rabbit_preprocessed \
	--output_dir /gscratch/cheme/agolan/data/rabbit_output

