"""
Script to perform train/test split on preprocessed .npy files.

Usage:
    python run_train_test_split.py --input_dir <path> --output_dir <path>
"""

import sys
from pathlib import Path
import argparse
import shutil
from typing import List
from src.saffron.data.data_processing import train_test_split_by_animal


def get_npy_files(directory: str) -> List[str]:
    """Get all .npy files from directory (including subdirectories)."""
    input_path = Path(directory)
    npy_files = list(input_path.rglob("*.npy"))  # rglob searches recursively
    return [str(f) for f in npy_files]


def copy_files_to_split(file_list: List[str], output_dir: str, split_name: str):
    """Copy files to their split directory."""
    split_dir = Path(output_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    for filepath in file_list:
        src = Path(filepath)
        dst = split_dir / src.name
        shutil.copy2(src, dst)
    
    print(f"Copied {len(file_list)} files to {split_dir}")


def main():
    parser = argparse.ArgumentParser(description="Split preprocessed .npy files into train/val/test")
    
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with .npy files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for splits')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set proportion')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Import the split function
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Get all .npy files
    print(f"Loading files from {args.input_dir}...")
    all_files = get_npy_files(args.input_dir)
    print(f"Found {len(all_files)} .npy files")
    
    # Perform split
    print(f"\nPerforming split (test={args.test_size}, val={args.val_size})...")
    # In run_train_test_split.py or your training script:
    train_files, val_files, test_files = train_test_split_by_animal(
        all_files, 
        test_size=0.2,
        val_size=0.1,
        random_seed=42
)
    
    # Copy files to split directories
    print("\nCopying files to split directories...")
    copy_files_to_split(train_files, args.output_dir, 'train')
    copy_files_to_split(val_files, args.output_dir, 'val')
    copy_files_to_split(test_files, args.output_dir, 'test')
    
    print(f"\n{'='*60}")
    print("SPLIT COMPLETE")
    print(f"{'='*60}")
    print(f"Train: {len(train_files)} files")
    print(f"Val:   {len(val_files)} files")
    print(f"Test:  {len(test_files)} files")
    print(f"Total: {len(all_files)} files")


if __name__ == "__main__":
    main()
