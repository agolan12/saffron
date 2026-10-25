#!/bin/bash
#SBATCH --job-name=fluor_shortcut
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs_output/fluor_shortcut_%A_%a.out
#SBATCH --error=logs_error/fluor_shortcut_%A_%a.err
#SBATCH --array=0-5                 # 6 tasks, one per line of PREPS below (indices 0..5)

# Shortcut test (brightness alone, no cell shape) on the fluorescent images, once per image
# preparation. No GPU: only a random forest on a few numbers per image.
# Submit from scripts/:
#     sbatch saffron_fluorescent_shortcut_array.slurm
#
# Each task writes shortcut_fluorescent/shortcut_fluorescent_<prep>.txt in the saffron folder.
# When all six have finished, compare the "balanced" column across them:
#     grep -A4 "^version" shortcut_fluorescent/*.txt

DATA_DIR=/gscratch/cheme/agolan/data/fluorescent_micro_images    # <-- folder holding the condition folders and li_thresh/

echo 'sourcing bashrc'
source ~/.bashrc
conda activate contrastive_ml

echo "Job started on $(hostname) | Array Task ID: $SLURM_ARRAY_TASK_ID"

cd /gscratch/cheme/agolan/saffron

PREPS=(
    raw            # 0: nothing, values as stored
    normalised     # 1: 1st-99.5th percentile stretched to 0-1
    denoised       # 2: background removal, then normalised
    li_raw         # 3: Li's threshold, no noise reduction
    li_denoised    # 4: Li with background removal and clean-up (image_prep.li_mask defaults)
    li_saved       # 5: the masks already in li_thresh/
)
PREP=${PREPS[$SLURM_ARRAY_TASK_ID]}
echo "Prep: $PREP"

python scripts/shortcut_baseline_fluorescent.py "$DATA_DIR" --prep="$PREP"
