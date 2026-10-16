"""
Pipeline Component 2: Train/Test Split

Ensures no data leakage between splits when using quadrants.
Splits at the base image level, then assigns all quadrants together.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict


def extract_base_name(filepath: str) -> str:
    """
    Extract base image name from filepath, removing quadrant suffix.
    
    Examples:
        'image_01_TL.npy' -> 'image_01'
        'image_01.npy' -> 'image_01'
        'data/image_01_BR.npy' -> 'image_01'
        'image_01_bottom_right_top_right.tif' -> 'image_01'   (tiff_quadrant_split.py, run twice)
    """
    # Get filename without directory
    filename = Path(filepath).stem  # Removes .npy extension too

    # Remove the quadrant suffix (_TL, _TR, _BL, _BR) if present
    for suffix in ['_TL', '_TR', '_BL', '_BR']:
        if filename.endswith(suffix):
            return filename[:-len(suffix)]

    # tiff_quadrant_split.py names quadrants _top_left etc.; images split more than once
    # carry several of these, and all of them belong to the same original image
    long_suffixes = ['_top_left', '_top_right', '_bottom_left', '_bottom_right']
    stripped = True
    while stripped:
        stripped = False
        for suffix in long_suffixes:
            if filename.endswith(suffix):
                filename = filename[:-len(suffix)]
                stripped = True

    return filename


def extract_animal_id(filepath: str) -> str:
    """
    Extract animal ID from filepath for cross-species data.
    
    Examples:
        Ferret: '20230308_ferret_085B_..._BR.npy' -> 'ferret_085B'
        Human: 'UWA-7606_A2_..._TR.npy' -> 'UWA-7606_A2'
        Mouse: '20230327_mouse_saline_p12_female_fs3_..._TR.npy' -> 'mouse_fs3'
        Pig: '20230312_pig_sow2_D_..._BR.npy' -> 'pig_sow2_D'
        Rabbit: '20230417_rabbit_22183C2_..._TL.npy' -> 'rabbit_22183C2'
        Rat: '20230417_rat_nontreated_p70_male_m3_..._TR.npy' -> 'rat_m3'
    """
    filename = Path(filepath).stem
    
    # Remove quadrant suffix first (_TL, _TR, _BL, _BR)
    for suffix in ['_TL', '_TR', '_BL', '_BR']:
        if filename.endswith(suffix):
            filename = filename[:-3]
    
    parts = filename.split('_')
    
    # Ferret: date_ferret_ID_...
    if 'ferret' in parts:
        idx = parts.index('ferret')
        if idx + 1 < len(parts):
            return f"ferret_{parts[idx + 1]}"  # e.g., 'ferret_085B'
    
    # Mouse: date_mouse_treatment_age_sex_ID_...
    elif 'mouse' in parts:
        idx = parts.index('mouse')
        # Find the 'fs' ID (e.g., fs3)
        for i in range(idx, len(parts)):
            if parts[i].startswith('fs'):
                return f"mouse_{parts[i]}"  # e.g., 'mouse_fs3'
    
    # Pig: date_pig_ID_...
    elif 'pig' in parts:
        idx = parts.index('pig')
        if idx + 2 < len(parts):
            return f"pig_{parts[idx + 1]}_{parts[idx + 2]}"  # e.g., 'pig_sow2_D'
    
    # Rabbit: date_rabbit_ID_...
    elif 'rabbit' in parts:
        idx = parts.index('rabbit')
        if idx + 1 < len(parts):
            return f"rabbit_{parts[idx + 1]}"  # e.g., 'rabbit_22183C2'
    
    # Rat: date_rat_treatment_age_sex_ID_...
    elif 'rat' in parts:
        idx = parts.index('rat')
        # Find the 'm' ID (e.g., m3)
        for i in range(idx, len(parts)):
            if parts[i].startswith('m') and len(parts[i]) <= 3:
                return f"rat_{parts[i]}"  # e.g., 'rat_m3'
    
    # Human: UWA-ID_subID_...
    elif parts[0].startswith('UWA'):
        return f"{parts[0]}_{parts[1]}"  # e.g., 'UWA-7606_A2'
    
    # Fallback: return first 3 parts joined
    return '_'.join(parts[:3])


def group_by_animal(file_paths: List[str]) -> Dict[str, List[str]]:
    """
    Group file paths by their base image name.
    
    Returns dict mapping base_name -> list of file paths
    """
    groups = defaultdict(list)
    
    for filepath in file_paths:
        base_name = extract_base_name(filepath)
        groups[base_name].append(filepath)
    
    return dict(groups)


def train_test_split(file_paths: List[str], 
                     test_size: float = 0.2, 
                     val_size: float = 0.1,
                     random_seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Split files into train/val/test ensuring no quadrant leakage.
    
    Returns (train_files, val_files, test_files)
    """
    np.random.seed(random_seed)
    
    # Group files by base image
    groups = group_by_base_image(file_paths)
    base_names = list(groups.keys())
    n_images = len(base_names)
    
    # Calculate split sizes at image level
    n_test = max(1, int(n_images * test_size))
    n_val = max(1, int(n_images * val_size))
    n_train = n_images - n_test - n_val
    
    # Shuffle base names
    np.random.shuffle(base_names)
    
    # Split base names (not quadrants!)
    test_bases = base_names[:n_test]
    val_bases = base_names[n_test:n_test + n_val]
    train_bases = base_names[n_test + n_val:]
    
    # Collect all files (including quadrants) for each split
    train_files = [f for base in train_bases for f in groups[base]]
    val_files = [f for base in val_bases for f in groups[base]]
    test_files = [f for base in test_bases for f in groups[base]]
    
    return train_files, val_files, test_files


if __name__ == "__main__":
    # Test with sample filenames
    print("Testing train/test split...")
    
    # Simulate quadrant files from 10 images
    test_files = []
    for i in range(10):
        base = f"image_{i:02d}"
        test_files.extend([
            f"{base}_TL.npy",
            f"{base}_TR.npy",
            f"{base}_BL.npy",
            f"{base}_BR.npy"
        ])
    
    print(f"\nTotal files: {len(test_files)}")
    print(f"Total base images: {len(set(extract_base_name(f) for f in test_files))}")
    
    # Perform split
    train, val, test = train_test_split(test_files, test_size=0.2, val_size=0.1)
    
    print(f"\nSplit results:")
    print(f"  Train: {len(train)} files from {len(set(extract_base_name(f) for f in train))} images")
    print(f"  Val:   {len(val)} files from {len(set(extract_base_name(f) for f in val))} images")
    print(f"  Test:  {len(test)} files from {len(set(extract_base_name(f) for f in test))} images")
    
    # Verify no leakage
    train_bases = set(extract_base_name(f) for f in train)
    val_bases = set(extract_base_name(f) for f in val)
    test_bases = set(extract_base_name(f) for f in test)
    
    print(f"\n✓ No leakage check:")
    print(f"  Train ∩ Val: {len(train_bases & val_bases)} (should be 0)")
    print(f"  Train ∩ Test: {len(train_bases & test_bases)} (should be 0)")
    print(f"  Val ∩ Test: {len(val_bases & test_bases)} (should be 0)")











def train_test_split_by_animal(file_paths: List[str], 
                                test_size: float = 0.2, 
                                val_size: float = 0.1,
                                random_seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Split files into train/val/test by ANIMAL to prevent animal leakage.
    
    This ensures no animal appears in multiple splits.
    
    Returns (train_files, val_files, test_files)
    """
    np.random.seed(random_seed)
    
    # Group files by animal
    groups = group_by_animal(file_paths)
    animal_ids = list(groups.keys())
    n_animals = len(animal_ids)
    
    print(f"Found {n_animals} unique animals")
    
    # Calculate split sizes at animal level
    n_test = max(1, int(n_animals * test_size))
    n_val = max(1, int(n_animals * val_size))
    n_train = n_animals - n_test - n_val
    
    print(f"Split: {n_train} train animals, {n_val} val animals, {n_test} test animals")
    
    # Shuffle animal IDs
    np.random.shuffle(animal_ids)
    
    # Split animal IDs (not files!)
    test_animals = animal_ids[:n_test]
    val_animals = animal_ids[n_test:n_test + n_val]
    train_animals = animal_ids[n_test + n_val:]
    
    # Collect all files (including all images and quadrants) for each split
    train_files = [f for animal in train_animals for f in groups[animal]]
    val_files = [f for animal in val_animals for f in groups[animal]]
    test_files = [f for animal in test_animals for f in groups[animal]]
    
    print(f"Result: {len(train_files)} train files, {len(val_files)} val files, {len(test_files)} test files")
    
    return train_files, val_files, test_files