#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --gpus-per-node=2080ti:1
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --output=logs_output/supconloss_%A_%a.out
#SBATCH --error=logs_error/supconloss_%A_%a.err
#SBATCH --array=0-10                  # <-- defines 5 tasks (indices 0,1,2,3,4)

echo 'sourcing bashrc'
source ~/.bashrc
conda activate contrastive_ml

echo "Job started on $(hostname) | Array Task ID: $SLURM_ARRAY_TASK_ID"

cd /gscratch/cheme/agolan/saffron

# Example: sweep over different learning rates using the task ID
LR_VALUES=(0.0003 0.0002 0.0001 0.00009 0.00008 0.00007 0.00006 0.00005 0.00004 0.00003)
LR=${LR_VALUES[$SLURM_ARRAY_TASK_ID]}

python scripts/supconloss_test_old.py \
    --epochs=100 \
    --batch_size=128 \
    --l=$LR