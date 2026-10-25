#!/bin/bash
#SBATCH --job-name=fluor_train
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2080ti:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs_output/fluor_train_%A_%a.out
#SBATCH --error=logs_error/fluor_train_%A_%a.err
#SBATCH --array=0-3                 # 4 tasks, one per line of PREPS below (indices 0..3)

# Train on the fluorescent images, once per image preparation, to compare what each learns
# (saliency maps) and how far each gets above its brightness-only score (shortcut_fluorescent/).
# Submit from scripts/:
#     sbatch saffron_fluorescent_train_array.slurm
#
# Each task first writes its preparation to $OUT_ROOT/<prep>/<condition>/*.npy with
# export_fluorescent.py (skipped if that folder already exists; delete it to redo):
# 1024 x 1024 images become four 512 x 512 pieces, kept on the same side of the split.
# Every setting except the preparation is shared, so the four runs compare directly.
#
# Weights: weights/job<jobid>_task<N>_final.pt, where N is the line of PREPS below.
# Summary when all four have finished:
#     python plot_runs.py logs_error/fluor_train_<jobid>_*.err -o fluor_train_<jobid>.png

DATA_DIR=/gscratch/cheme/agolan/data/fluorescent_micro_images    # <-- folder holding the condition folders and li_thresh/
OUT_ROOT=/gscratch/cheme/agolan/data/fluorescent_prepped

echo 'sourcing bashrc'
source ~/.bashrc
conda activate contrastive_ml

echo "Job started on $(hostname) | Array Task ID: $SLURM_ARRAY_TASK_ID"

cd /gscratch/cheme/agolan/saffron

PREPS=(
    denoised       # 0: background removal, then normalised
    li_raw         # 1: Li's threshold, no noise reduction
    li_denoised    # 2: Li with background removal and clean-up (image_prep.li_mask defaults)
    li_saved       # 3: the masks already in li_thresh/
)
PREP=${PREPS[$SLURM_ARRAY_TASK_ID]}
PREP_DIR=$OUT_ROOT/$PREP
echo "Prep: $PREP   data: $PREP_DIR"

if [ ! -d "$PREP_DIR" ]; then
    python scripts/export_fluorescent.py "$DATA_DIR" --prep="$PREP" --out_dir="$PREP_DIR" || exit 1
fi

python scripts/supconloss_test_old.py \
    --classes=fluorescent \
    --data_dir="$PREP_DIR" \
    --stage1_epochs=60 \
    --epochs=15 \
    --batch_size=128 \
    --ce_weight=0.5
