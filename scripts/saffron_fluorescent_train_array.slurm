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
#SBATCH --array=0-3                 # 4 tasks, one per line of PREPS below (indices 0..3);
                                    # with SPLIT=sample submit with --array=0-$((4 * N_FOLDS - 1)), see below

# Train on the fluorescent images, once per image preparation, to compare what each learns
# (saliency maps) and how far each gets above its brightness-only score (shortcut_fluorescent/).
# Submit from scripts/:
#     sbatch saffron_fluorescent_train_array.slurm
#
# Each task first writes its preparation to $OUT_ROOT/<prep>_<PIECE>/<condition>/*.npy with
# export_fluorescent.py (skipped if that folder already exists; delete it to redo): images are
# cut into PIECE x PIECE pieces, all pieces of one image on the same side of the split, and the
# CNN is trained at that size. With PIECE=256 a 1024 image gives 16 pieces and a 512 image 4.
# Change PIECE below, or set it at submission: sbatch --export=ALL,PIECE=128 <this file>
# Every setting except the preparation is shared, so the four runs compare directly.
#
# Weights: weights/job<jobid>_task<N>_final.pt, where N is the line of PREPS below.
# Validation predictions at the end of each stage: predictions/job<jobid>_task<N>_stage<1|2>.csv
# Summary when all four have finished:
#     python scripts/plot_runs.py logs_error/fluor_train_<jobid>_*.err -o fluor_train_<jobid>.png
#
# Cross-validation by sample (SPLIT=sample): every prep is trained N_FOLDS times, each time
# validating on different samples, so there are 4 * N_FOLDS tasks. Task T trains prep
# PREPS[T / N_FOLDS] on fold T % N_FOLDS (tasks 0-3 = denoised folds 0-3, 4-7 = li_raw, ...).
#     sbatch --array=0-15 --export=ALL,SPLIT=sample,N_FOLDS=4,REVIEW_CSV=...,CLASSES=... <this file>
# Then pool the folds of each prep (one result per prep and stage):
#     python scripts/pool_folds.py predictions/job<jobid>_task*_stage2.csv

DATA_DIR=${DATA_DIR:-/gscratch/cheme/agolan/data/fluorescent_microglia_images}    # <-- folder holding the condition folders and li_thresh/
OUT_ROOT=/gscratch/cheme/agolan/data/fluorescent_prepped
PIECE=${PIECE:-256}                                                 # piece side in pixels: 512, 256 or 128
# Clean images only: set REVIEW_CSV to the review_images.py labels (copy fluorescent_review.csv to
# Hyak first; it is gitignored). Only "good" images are exported, to <prep>_<PIECE>_good/:
#     sbatch --export=ALL,REVIEW_CSV=/gscratch/cheme/agolan/saffron/fluorescent_review.csv <this file>
REVIEW_CSV=${REVIEW_CSV:-}
# Class set and run settings; set any of them at submission, e.g.
#     sbatch --export=ALL,CLASSES=fluorescent_ogd_3_vs_control,CE=0.3,STAGE1=200,EPOCHS=50 <this file>
# CLASSES: fluorescent (all six), fluorescent_ogd_vs_control, fluorescent_ogd_3_vs_control
CLASSES=${CLASSES:-fluorescent}
CE=${CE:-0.5}               # --ce_weight
LR=${LR:-0.0003}            # --learning_rate
STAGE1=${STAGE1:-60}        # --stage1_epochs
EPOCHS=${EPOCHS:-15}        # --epochs (stage 2)
LR_DROP=${LR_DROP:-0}       # --lr_drop_epoch: stage 1 learning rate x 0.1 after this epoch (0 = never)
SPLIT=${SPLIT:-image}       # --split_by: image (one 80/20 split) or sample (cross-validation, see above)
N_FOLDS=${N_FOLDS:-4}       # --n_folds, for SPLIT=sample

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
if [ "$SPLIT" = sample ]; then
    PREP=${PREPS[$((SLURM_ARRAY_TASK_ID / N_FOLDS))]}
    FOLD=$((SLURM_ARRAY_TASK_ID % N_FOLDS))
    if [ "${SLURM_ARRAY_TASK_COUNT:-0}" -ne $((4 * N_FOLDS)) ]; then
        echo "WARNING: SPLIT=sample with N_FOLDS=$N_FOLDS needs --array=0-$((4 * N_FOLDS - 1)); this array has ${SLURM_ARRAY_TASK_COUNT:-?} tasks"
    fi
else
    PREP=${PREPS[$SLURM_ARRAY_TASK_ID]}
    FOLD=0
fi
[ -n "$PREP" ] || { echo "no prep for task $SLURM_ARRAY_TASK_ID"; exit 1; }
PREP_DIR=$OUT_ROOT/${PREP}_${PIECE}
REVIEW_ARGS=()
if [ -n "$REVIEW_CSV" ]; then
    [ -f "$REVIEW_CSV" ] || { echo "REVIEW_CSV not found: $REVIEW_CSV"; exit 1; }
    PREP_DIR=${PREP_DIR}_good
    REVIEW_ARGS=(--review_csv="$REVIEW_CSV" --keep good)
fi
echo "Prep: $PREP   piece size: $PIECE   data: $PREP_DIR   review filter: ${REVIEW_CSV:-none (all images)}"
echo "Classes: $CLASSES   ce_weight: $CE   lr: $LR   lr drop: $LR_DROP   epochs: $STAGE1 + $EPOCHS   split: $SPLIT fold $FOLD of $N_FOLDS"

# Export once per prep. Several tasks share a prep folder (both class sets, every fold), so the
# first task holds a lock while it exports to a temporary folder and renames it when complete;
# the others wait for the lock and then find the finished folder.
(
    flock 9
    if [ ! -d "$PREP_DIR" ]; then
        TMP_DIR="$PREP_DIR.tmp.$SLURM_JOB_ID"
        rm -rf "$TMP_DIR"
        python scripts/export_fluorescent.py "$DATA_DIR" --prep="$PREP" --piece_size="$PIECE" --out_dir="$TMP_DIR" \
            "${REVIEW_ARGS[@]}" || { rm -rf "$TMP_DIR"; exit 1; }
        mv "$TMP_DIR" "$PREP_DIR"
    fi
) 9>"$PREP_DIR.lock" || { echo "export failed"; exit 1; }

python scripts/supconloss_test_old.py \
    --classes="$CLASSES" \
    --data_dir="$PREP_DIR" \
    --image_size="$PIECE" \
    --stage1_epochs="$STAGE1" \
    --epochs="$EPOCHS" \
    --batch_size=128 \
    --learning_rate="$LR" \
    --lr_drop_epoch="$LR_DROP" \
    --split_by="$SPLIT" \
    --n_folds="$N_FOLDS" \
    --fold="$FOLD" \
    --ce_weight="$CE"
