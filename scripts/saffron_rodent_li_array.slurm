#!/bin/bash
#SBATCH --job-name=rodent_li
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2080ti:1
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --output=logs_output/rodent_li_%A_%a.out
#SBATCH --error=logs_error/rodent_li_%A_%a.err
#SBATCH --array=0-3                 # 4 tasks, one per line of RUNS below (indices 0,1,2,3)

# Two questions:
#   A. Mice and rat only (the higher-quality images): how well are they told apart
#      as stored, with brightness equalised, and as Li-threshold masks? (tasks 0, 1, 2)
#   B. All three classes as Li-threshold masks. (task 3)
# Submit from scripts/:
#     sbatch saffron_rodent_li_array.slurm
#
# When all four tasks have finished:
#     python plot_runs.py logs_error/rodent_li_<jobid>_*.err -o rodent_li_<jobid>.png
# That writes rodent_li_<jobid>.png and rodent_li_<jobid>.txt (the summary table).
#
# Stage 1 is 100 epochs here (it was 60): the equalised runs were still improving at
# epoch 60. Tasks 0-2 share every setting except the image preparation, so compare
# those three with each other. Compare task 3 with the earlier 3-class runs only
# loosely, because the stage 1 length differs.
#
# With two classes the numbers to keep in mind change:
#     always guessing mice      accuracy 0.53   (0.41 with three classes)
#     SupCon loss, batch 128    4.84 = nothing learned, about 4.14 = perfect
#                               (about 3.75 with three classes)

echo 'sourcing bashrc'
source ~/.bashrc
conda activate contrastive_ml

echo "Job started on $(hostname) | Array Task ID: $SLURM_ARRAY_TASK_ID"

cd /gscratch/cheme/agolan/saffron

# One run per line. The task ID picks the line.
RUNS=(
    "--classes=rodent"                    # 0: mice vs rat, images as stored (the comparison run)
    "--classes=rodent --equalize"         # 1: mice vs rat, brightness equalised
    "--classes=rodent --li_threshold"     # 2: mice vs rat, Li-threshold masks
    "--li_threshold"                      # 3: all three classes, Li-threshold masks
)
ARGS=${RUNS[$SLURM_ARRAY_TASK_ID]}
echo "Run settings: $ARGS"

python scripts/supconloss_test_old.py \
    --stage1_epochs=100 \
    --epochs=15 \
    --batch_size=128 \
    --ce_weight=0.5 \
    $ARGS
