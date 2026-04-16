import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from pathlib import Path

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

# test class added by assaf
class MicrogliaDataset(Dataset):
    """Supervised dataset for microglia classification."""
    
    def __init__(self, data_dir, train=True, labels=['HC', 'OGD', 'ROT'], 
    transform=None, merge_map=None):
        """
        Args:
            data_dir: Path to directory containing subdirectories for each label
            train: Not used, kept for compatibility
            labels: List of class labels (should match subdirectory names)
            transform: Optional transforms to apply
            merge_map: Option map to merge labeled groups
        """
        self.data_dir = Path(data_dir)
        self.labels = labels
        self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        self.transform = transform
        self.merge_map = merge_map or {}
        self.samples = []
        
        # Load files from each label subdirectory
        dirs_to_scan = []
        for label in labels:
            dirs_to_scan.append((label, label))  # (dir_name, label_name)
        for dir_name, label_name in self.merge_map.items():
            dirs_to_scan.append((dir_name, label_name))

        for dir_name, label_name in dirs_to_scan:
            label_dir = self.data_dir / dir_name
            if not label_dir.exists():
                print(f"Warning: Directory {label_dir} does not exist, skipping...")
                continue
            for filepath in label_dir.rglob("*.npy"):
                self.samples.append((str(filepath), self.label_to_idx[label_name]))
        
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


def generate_dataloaders(dataset, num_workers=2, batch_size=32, val_split=0.2):
    """
    Split dataset into train/val and create DataLoaders.
    
    Args:
        dataset: MicrogliaDataset instance
        num_workers: Number of worker processes
        batch_size: Batch size
        val_split: Fraction of data for validation
    
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import random_split, WeightedRandomSampler
    
    # Calculate split sizes
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    # Split dataset
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"Train size: {train_size}, Val size: {val_size}")
    
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
