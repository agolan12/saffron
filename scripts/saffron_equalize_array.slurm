#!/bin/bash
#SBATCH --job-name=equalize
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2080ti:1
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --output=logs_output/equalize_%A_%a.out
#SBATCH --error=logs_error/equalize_%A_%a.err
#SBATCH --array=0-1                 # 2 tasks, one per line of RUNS below (indices 0,1)

# Does the model still tell the classes apart when brightness cannot help?
# Two runs that differ in one thing only: task 1 equalises every image first.
# Submit from scripts/:
#     sbatch saffron_equalize_array.slurm
#
# When both tasks have finished:
#     python plot_runs.py logs_error/equalize_<jobid>_*.err -o equalize_<jobid>.png
# That writes equalize_<jobid>.png and equalize_<jobid>.txt (the summary table).

echo 'sourcing bashrc'
source ~/.bashrc
conda activate contrastive_ml

echo "Job started on $(hostname) | Array Task ID: $SLURM_ARRAY_TASK_ID"

cd /gscratch/cheme/agolan/saffron

# One run per line. The task ID picks the line.
RUNS=(
    "--ce_weight=0.5"                 # 0: images as they are (the comparison run)
    "--ce_weight=0.5 --equalize"      # 1: every image given the same brightness distribution
)
ARGS=${RUNS[$SLURM_ARRAY_TASK_ID]}
echo "Run settings: $ARGS"

python scripts/supconloss_test_old.py \
    --stage1_epochs=60 \
    --epochs=15 \
    --batch_size=128 \
    $ARGS
