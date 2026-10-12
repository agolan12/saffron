#!/bin/bash
#SBATCH --job-name=followup
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2080ti:1
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --output=logs_output/followup_%A_%a.out
#SBATCH --error=logs_error/followup_%A_%a.err
#SBATCH --array=0-4                 # 5 tasks, one per line of RUNS below (indices 0,1,2,3,4)

# Follow-up to the ce_weight sweep (job 34737231 and later). Submit from scripts/:
#     sbatch saffron_followup_array.slurm
#
# When all five tasks have finished, make the comparison figure and results file:
#     python plot_runs.py logs_error/followup_<jobid>_*.err -o followup_<jobid>.png
# That writes followup_<jobid>.png and followup_<jobid>.txt (the summary table).
#
# Every run uses the shorter schedule: 60 stage 1 epochs + 15 stage 2 epochs.
# The model now has 3 outputs (it had 6), so compare these runs with each other,
# not with the older sweep.

echo 'sourcing bashrc'
source ~/.bashrc
conda activate contrastive_ml

echo "Job started on $(hostname) | Array Task ID: $SLURM_ARRAY_TASK_ID"

cd /gscratch/cheme/agolan/saffron

# One run per line. The task ID picks the line.
RUNS=(
    "--ce_weight=0.5"                                                       # 0: new reference
    "--ce_weight=1.0"                                                       # 1: cross-entropy only, no SupCon
    "--ce_weight=0.5 --stage2_head_only"                                    # 2: stage 2 trains the last layer only
    "--ce_weight=0.5 --lr_drop_epoch=40"                                    # 3: learning rate / 10 after stage 1 epoch 40
    "--ce_weight=0.0 --stage2_head_only --stage2_lr=0.001 --epochs=40"      # 4: SupCon only, then last layer only
)
ARGS=${RUNS[$SLURM_ARRAY_TASK_ID]}
echo "Run settings: $ARGS"

# $ARGS comes last so a line above can override the shared settings (run 4 sets --epochs=40)
python scripts/supconloss_test_old.py \
    --stage1_epochs=60 \
    --epochs=15 \
    --batch_size=128 \
    $ARGS
