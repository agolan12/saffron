import os
from saffron.data import load_split_files, create_dataloaders
from saffron.models.loss_functions import ContrastiveLoss
from saffron.models.torch_models import ContrastiveModel
import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path
import time


def train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        masked_images = batch['masked_image'].to(device)
        positive_patches = batch['positive_patch'].to(device)
        negative_patches = batch['negative_patches'].to(device)
        
        # Forward pass
        loss = loss_fn(model, masked_images, positive_patches, negative_patches)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
        num_batches += 1
        
        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} | Loss: {avg_loss:.4f}")
    
    return total_loss / num_batches


def validate(model, val_loader, loss_fn, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            masked_images = batch['masked_image'].to(device)
            positive_patches = batch['positive_patch'].to(device)
            negative_patches = batch['negative_patches'].to(device)
            
            loss = loss_fn(model, masked_images, positive_patches, negative_patches)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def train_model(model, train_loader, val_loader, loss_fn, optimizer, 
                num_epochs, device, save_dir='checkpoints', 
                patience=7, use_scheduler=True):
    """
    Main training function with learning rate scheduling and early stopping.
    
    Args:
        patience: Number of epochs without improvement before stopping
        use_scheduler: Whether to use CosineAnnealingLR scheduler
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    
    # Create learning rate scheduler
    scheduler = None
    if use_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        print(f"✓ Using CosineAnnealingLR scheduler")
    
    print(f"✓ Early stopping enabled (patience={patience})")
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        if scheduler:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, loss_fn, device)
        print(f"Val Loss: {val_loss:.4f}")
        
        # Track losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path / 'best_model.pt')
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
        
        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"\n{'='*60}")
            print(f"Early stopping triggered after {epoch} epochs")
            print(f"Best val loss: {best_val_loss:.4f}")
            print(f"{'='*60}")
            break
        
        # Learning rate scheduler step
        if scheduler:
            scheduler.step()
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save(model.state_dict(), save_path / f'checkpoint_epoch_{epoch}.pt')
    
    print(f"\n{'='*60}")
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Total epochs: {epoch}")
    print(f"{'='*60}")
    
    return train_losses, val_losses


def main():

    print("starting script")

    train_files, val_files, test_files = load_split_files(
        split_dir="/gscratch/cheme/agolan/data/preprocessed_data/split_data"
        )

    train_loader, val_loader, test_loader = create_dataloaders(
        train_files=train_files,
        val_files=val_files, test_files=test_files,
        num_workers=4,
        batch_size=32,
        n_negatives=5,
        patch_size=128)


    
    print(f"Made data loaders: {len(train_loader)} train batches")
    
    # QUICK TEST - Get one batch and check shapes
    print("\n" + "="*60)
    print("TESTING FIRST BATCH")
    print("="*60)
    
    batch = next(iter(train_loader))
    
    print(f"Batch keys: {batch.keys()}")
    print(f"Masked image shape: {batch['masked_image'].shape}")
    print(f"Positive patch shape: {batch['positive_patch'].shape}")
    print(f"Negative patches shape: {batch['negative_patches'].shape}")
    
    print(f"\nMasked image dtype: {batch['masked_image'].dtype}")
    print(f"Masked image range: [{batch['masked_image'].min():.3f}, {batch['masked_image'].max():.3f}]")
    
    print("\n✓ DataLoader test passed!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ContrastiveModel(backbone_type='resnet18', pretrained=False, input_channels=1).to(device)
    loss_fn = ContrastiveLoss(temperature=0.5)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)  # Was 1e-5

    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Model on GPU: {next(model.parameters()).is_cuda}")
    
    
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, loss_fn, optimizer,
        num_epochs=100,        # Train longer with regularization
        device=device,
        patience=25,           # Early stopping patience
        use_scheduler=True    # Use learning rate scheduler
    )

if __name__ == "__main__":
    main()

