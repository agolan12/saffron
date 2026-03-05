"""
Pipeline Component 5: Contrastive Loss Function

Loss function for training the contrastive learning model.
Pulls positive patches close, pushes negative patches away.

Also re-exports SupConLoss from pytorch-metric-learning for supervised
contrastive learning (use with embeddings and class labels).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_metric_learning.losses import SupConLoss as _SupConLoss

    def SupConLoss(temperature: float = 0.07, **kwargs):
        """Supervised contrastive loss from pytorch-metric-learning.

        Usage:
            loss_fn = SupConLoss(temperature=0.07)
            embeddings = model(images)  # (B, embedding_dim)
            loss = loss_fn(embeddings, labels)  # labels: (B,) class indices
        """
        return _SupConLoss(temperature=temperature, **kwargs)

except ImportError:
    SupConLoss = None  # type: ignore[misc, assignment]


class ContrastiveLoss(nn.Module):
    """Contrastive loss for masked image and patch pairs."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, model, masked_images, positive_patches, negative_patches):
        """
        Compute contrastive loss.
        
        Args:
            model: Encoder model
            masked_images: (B, 1, H, W)
            positive_patches: (B, 1, pH, pW)
            negative_patches: (B, N, 1, pH, pW)
        """
        batch_size = masked_images.shape[0]
        n_negatives = negative_patches.shape[1]
        
        # Encode masked images
        masked_embeddings = model(masked_images)  # (B, embedding_dim)
        
        # Encode positive patches
        positive_embeddings = model(positive_patches)  # (B, embedding_dim)
        
        # Encode negative patches - reshape to (B*N, 1, pH, pW)
        neg_reshaped = negative_patches.reshape(batch_size * n_negatives, 1, *negative_patches.shape[-2:])
        negative_embeddings = model(neg_reshaped)  # (B*N, embedding_dim)
        negative_embeddings = negative_embeddings.reshape(batch_size, n_negatives, -1)  # (B, N, embedding_dim)
        
        # Normalize embeddings
        masked_embeddings = F.normalize(masked_embeddings, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, dim=1)
        negative_embeddings = F.normalize(negative_embeddings, dim=2)
        
        # Compute similarity scores with positive (higher is better)
        pos_similarity = (masked_embeddings * positive_embeddings).sum(dim=1) / self.temperature  # (B,)
        
        # Compute similarity scores with negatives
        # masked_embeddings: (B, embedding_dim) -> (B, 1, embedding_dim)
        # negative_embeddings: (B, N, embedding_dim)
        neg_similarity = (masked_embeddings.unsqueeze(1) * negative_embeddings).sum(dim=2) / self.temperature  # (B, N)
        
        # Contrastive loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # Concatenate positive and negative similarities
        all_similarities = torch.cat([pos_similarity.unsqueeze(1), neg_similarity], dim=1)  # (B, 1+N)
        
        # Target is index 0 (the positive)
        targets = torch.zeros(batch_size, dtype=torch.long, device=masked_images.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(all_similarities, targets)
        
        return loss


if __name__ == "__main__":
    print("Testing ContrastiveLoss...")

    # Create dummy model
    class DummyModel(nn.Module):
        def forward(self, x):
            # Simple flatten and linear projection
            b = x.shape[0]
            return torch.randn(b, 128)  # Return random embeddings

    model = DummyModel()
    loss_fn = ContrastiveLoss(temperature=0.07)

    # Create dummy batch
    masked_images = torch.randn(4, 1, 256, 256)
    positive_patches = torch.randn(4, 1, 64, 64)
    negative_patches = torch.randn(4, 3, 1, 64, 64)

    # Compute loss
    loss = loss_fn(model, masked_images, positive_patches, negative_patches)

    print(f"Loss value: {loss.item():.4f}")
    print(f"Loss is scalar: {loss.shape == torch.Size([])}")
    print("\n✓ Loss function test passed!")