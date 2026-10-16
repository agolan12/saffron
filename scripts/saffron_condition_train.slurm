#!/bin/bash
#SBATCH --job-name=condition
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2080ti:1
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --output=logs_output/condition_%j.out
#SBATCH --error=logs_error/condition_%j.err

# HC vs OGD vs ROT, trained on TIFFs that are already processed: no --equalize, no
# --li_threshold. Images are only resized to 512x512, augmented (training split only)
# and scaled to -1..1, as in every other run.
#
# The TIFFs are label masks: background 0, each cell stored as its own object number
# (1, 2, 3, ... wrapping at 255). --binarize turns every nonzero pixel into 1 = cell, so
# the model sees cell shapes and not the order in which cells were numbered.
#
# DATA_DIR must hold the three folders HC/, OGD/ and ROT/ (names are case-sensitive).
# Files in them can be .tif, .tiff or .npy, in subfolders too.
#
# Submit from scripts/:
#     sbatch saffron_condition_train.slurm
# Then:
#     python plot_runs.py logs_error/condition_<jobid>.err -o condition_<jobid>.png
#
# Check the .out file first: for each folder it prints the file count and the shape and
# value range of the first file as loaded.

DATA_DIR=/gscratch/cheme/agolan/data/Training    # <-- set to the folder holding HC, OGD, ROT

echo 'sourcing bashrc'
source ~/.bashrc
conda activate contrastive_ml

echo "Job started on $(hostname)"

cd /gscratch/cheme/agolan/saffron

python scripts/supconloss_test_old.py \
    --classes=condition \
    --binarize \
    --data_dir="$DATA_DIR" \
    --stage1_epochs=60 \
    --epochs=15 \
    --batch_size=128 \
    --ce_weight=0.5
