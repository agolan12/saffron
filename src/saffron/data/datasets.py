import zlib
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from pathlib import Path

from .data_processing import extract_base_name

def get_random_patch_position(image_shape: Tuple[int, int], patch_size: int) -> Tuple[int, int]:
    """
    Generate random top-left position for a patch.
    
    Returns (row, col) ensuring patch fits within image.
    """
    h, w = image_shape
    max_row = h - patch_size
    max_col = w - patch_size
    
    row = np.random.randint(0, max_row + 1)
    col = np.random.randint(0, max_col + 1)
    
    return (row, col)


def extract_patch(image: np.ndarray, position: Tuple[int, int], patch_size: int) -> np.ndarray:
    """
    Extract a patch from an image.
    
    Returns patch of shape (patch_size, patch_size)
    """
    row, col = position
    patch = image[row:row + patch_size, col:col + patch_size]
    return patch


def create_masked_image(image: np.ndarray, position: Tuple[int, int], patch_size: int) -> np.ndarray:
    """
    Create masked image by zeroing out a patch.
    
    Returns copy of image with patch region set to 0.
    """
    masked = image.copy()
    row, col = position
    masked[row:row + patch_size, col:col + patch_size] = 0.0
    return masked


class PatchPairDataset(Dataset):
    """Dataset that generates masked images and positive/negative patch pairs."""
    
    def __init__(self, file_paths: List[str], patch_size: int = 64, n_negatives: int = 3):
        self.file_paths = file_paths
        self.patch_size = patch_size
        self.n_negatives = n_negatives
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx: int):
        # Load image
        image_path = self.file_paths[idx]
        image = np.load(image_path)  # Shape: (512, 512), float32, [0, 1]
        
        # Generate random position for positive patch
        pos_position = get_random_patch_position(image.shape, self.patch_size)
        
        # Extract positive patch
        positive_patch = extract_patch(image, pos_position, self.patch_size)
        
        # Create masked image
        masked_image = create_masked_image(image, pos_position, self.patch_size)
        
        # Generate negative patches from random positions
        negative_patches = []
        for _ in range(self.n_negatives):
            neg_position = get_random_patch_position(image.shape, self.patch_size)
            neg_patch = extract_patch(image, neg_position, self.patch_size)
            negative_patches.append(neg_patch)
        
        # Convert to tensors and add channel dimension
        masked_image = torch.from_numpy(masked_image).unsqueeze(0)  # (1, H, W)
        positive_patch = torch.from_numpy(positive_patch).unsqueeze(0)  # (1, pH, pW)
        negative_patches = torch.stack([torch.from_numpy(p).unsqueeze(0) for p in negative_patches])  # (N, 1, pH, pW)
        
        return {
            'masked_image': masked_image,
            'positive_patch': positive_patch,
            'negative_patches': negative_patches
        }

class MicrogliaDataset(Dataset):
    """Supervised dataset for microglia classification."""
    
    def __init__(
        self,
        data_dir,
        labels=['HC', 'OGD', 'ROT'],
        transform=None,
        merge_map=None,
    ):
        """
        Args:
            data_dir: Path to directory containing subdirectories for each label
            train: Not used, kept for compatibility
            labels: Logical class names in output label order (indices 0..C-1)
            transform: Optional transforms to apply
            merge_map: Optional dict mapping subdirectory name -> logical label name.
                Files under that subdirectory use the target label's index. Example:
                ``{'human': 'gyrified', 'ferret': 'gyrified'}`` loads ``human/*`` as class
                ``gyrified`` if ``'gyrified'`` is in ``labels``.
        """
        self.data_dir = Path(data_dir)
        self.labels = list(labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.transform = transform
        self.merge_map = merge_map or {}
        self.samples = []

        # Subdirectories to scan: explicit label folders plus any merge_map sources
        subdirs_to_scan = set(self.labels)
        subdirs_to_scan.update(self.merge_map.keys())

        for subdir_name in sorted(subdirs_to_scan):
            label_dir = self.data_dir / subdir_name
            if not label_dir.exists():
                continue
            logical_label = self.merge_map.get(subdir_name, subdir_name)
            if logical_label not in self.label_to_idx:
                print(f"Warning: Mapped label {logical_label!r} not in labels, skipping {label_dir}")
                continue
            label_idx = self.label_to_idx[logical_label]
            for filepath in label_dir.rglob("*.npy"):
                self.samples.append((str(filepath), label_idx))
        
        self.length = len(self.samples)
        print(f"Loaded {self.length} images from {data_dir}")
        print(f"Class distribution: {self._get_class_distribution()}")
    
    def _get_class_distribution(self):
        """Count samples per class."""
        counts = {label: 0 for label in self.labels}
        for _, label_idx in self.samples:
            label_name = self.labels[label_idx]
            counts[label_name] += 1
        return counts
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        # Load .npy file
        image = np.load(image_path)  # Shape: (H, W), float32, already normalized
        
        # Convert to tensor and add channel dimension
        image = torch.from_numpy(image).unsqueeze(0).float()  # (1, H, W)
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label


def split_indices_by_image(dataset, val_split=0.2, seed=42):
    """
    Split sample indices into train/val so that every quadrant of an image
    lands on the same side.

    Files are grouped by original image (filename without the _TL/_TR/_BL/_BR
    suffix). Each top-level folder under ``dataset.data_dir`` is split
    separately, so every folder contributes about ``val_split`` of its images
    to validation.

    The split depends only on folder names, filenames and ``seed`` -- not on
    the dataset's labels or merge_map -- so two MicrogliaDatasets built over
    the same files get the same validation images.

    Returns:
        train_indices, val_indices
    """
    # folder -> image -> indices of that image's quadrants
    images_by_folder = defaultdict(lambda: defaultdict(list))
    for idx, (filepath, _) in enumerate(dataset.samples):
        path = Path(filepath)
        folder = path.relative_to(dataset.data_dir).parts[0]
        image = str(path.parent / extract_base_name(filepath))
        images_by_folder[folder][image].append(idx)

    train_indices, val_indices = [], []
    for folder in sorted(images_by_folder):
        images = sorted(images_by_folder[folder])
        # Seeded per folder, so a folder's split doesn't depend on which other folders are loaded
        rng = np.random.default_rng([seed, zlib.crc32(folder.encode())])
        rng.shuffle(images)
        n_val = round(len(images) * val_split)
        if len(images) > 1:
            n_val = min(max(n_val, 1), len(images) - 1)  # at least one image on each side
        for i, image in enumerate(images):
            (val_indices if i < n_val else train_indices).extend(images_by_folder[folder][image])

    # Files are listed folder by folder, so in file order a validation batch would
    # hold a single class, and batch-level metrics (SupConLoss returns 0 for a
    # batch with no negatives) would be meaningless. Fixed shuffle: mixed batches,
    # same order every epoch and every run.
    val_indices = sorted(val_indices)
    np.random.default_rng(seed).shuffle(val_indices)
    return sorted(train_indices), val_indices


class TransformedSubset(Dataset):
    """
    The samples of ``dataset`` at ``indices``, with an extra transform applied
    to each image. Lets the train and val halves of one dataset use different
    transforms (augmentation for train, none for val).
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label


def generate_dataloaders(dataset, num_workers=2, batch_size=32, val_split=0.2,
                         train_transform=None, val_transform=None):
    """
    Split dataset into train/val by image and create DataLoaders.
    
    Args:
        dataset: MicrogliaDataset instance
        num_workers: Number of worker processes
        batch_size: Batch size
        val_split: Fraction of images (per folder) for validation
        train_transform: Transforms for training images only (put random
            augmentation here)
        val_transform: Transforms for validation images only (resize/normalize,
            nothing random)

        Both are applied after the dataset's own ``transform``, which runs on
        every image. Leave that one as None when using these, otherwise its
        augmentation reaches the validation images too.
    
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import WeightedRandomSampler
    
    # Split by image, not by file: quadrants of one image must not straddle train/val
    train_indices, val_indices = split_indices_by_image(dataset, val_split=val_split, seed=42)
    train_dataset = TransformedSubset(dataset, train_indices, train_transform)
    val_dataset = TransformedSubset(dataset, val_indices, val_transform)
    
    print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}")
    
    # Get class counts for training set
    train_labels = [dataset.samples[i][1] for i in train_dataset.indices]
    class_counts = np.bincount(train_labels)
    
    print(f"\nClass distribution in training set:")
    for idx, label in enumerate(dataset.labels):
        print(f"  {label}: {class_counts[idx]} samples")
    
    # Calculate weights for each sample (inverse of class frequency)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler, # weighted sampler instead of shuffle
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader



def create_dataloaders(train_files: List[str],
                       val_files: List[str], 
                       test_files: List[str],
                       batch_size: int = 16,
                       patch_size: int = 64,
                       n_negatives: int = 3,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test DataLoaders."""
    # Create datasets
    train_dataset = PatchPairDataset(train_files, patch_size=patch_size, n_negatives=n_negatives)
    val_dataset = PatchPairDataset(val_files, patch_size=patch_size, n_negatives=n_negatives)
    test_dataset = PatchPairDataset(test_files, patch_size=patch_size, n_negatives=n_negatives)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def load_split_files(split_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Load train/val/test file lists from split directories.
    
    Expects structure:
        split_dir/
            train/*.npy
            val/*.npy
            test/*.npy
    """
    split_path = Path(split_dir)
    
    train_files = [str(f) for f in (split_path / 'train').glob('*.npy')]
    val_files = [str(f) for f in (split_path / 'val').glob('*.npy')]
    test_files = [str(f) for f in (split_path / 'test').glob('*.npy')]
    
    return train_files, val_files, test_files


if __name__ == "__main__":
    # Test the dataset
    print("Testing PatchPairDataset...")
    
    # Create a fake .npy file for testing
    test_image = np.random.rand(512, 512).astype(np.float32)
    np.save('/tmp/test_image.npy', test_image)
    
    # Create dataset
    dataset = PatchPairDataset(['/tmp/test_image.npy'], patch_size=64, n_negatives=3)
    
    print(f"Dataset length: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample shapes:")
    print(f"  Masked image: {sample['masked_image'].shape}")
    print(f"  Positive patch: {sample['positive_patch'].shape}")
    print(f"  Negative patches: {sample['negative_patches'].shape}")
    
    print("\n✓ Dataset test passed!")
