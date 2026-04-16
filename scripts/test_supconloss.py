import os
import argparse
import logging
import torch
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import umap

# Add src to path so we can import saffron (script lives in scripts/, package in src/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from saffron.io import data_io
from saffron.data.datasets import MicrogliaDataset, generate_dataloaders
from saffron.data import data_processing
from saffron.models.torch_models import MicrogliaCNN
from saffron.models.loss_functions import SupConLoss
from torchvision.transforms import v2
import torch.nn as nn
import torch.nn.functional as F

class MicrogliaEmbeddingWrapper(nn.Module):
    """Wraps MicrogliaCNN to return L2-normalized 500-d embeddings for SupConLoss (penultimate layer)."""
    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn

    def forward(self, x):
        out = self.cnn.cnn1(x)
        out = self.cnn.cnn2(out)
        out = out.view(out.size(0), -1)
        out = self.cnn.fc1[0](out)
        out = self.cnn.fc1[1](out)
        return F.normalize(out, p=2, dim=1)


def device_check(req_dev):
	"""
	Decides the device being used for training.
	Returns:
		req_dev: [None, "cpu", "cuda", "mps"]
	"""
	device = None

	if req_dev:
		# Attempt manual device selection
		if req_dev == "mps" and torch.backends.mps.is_available():
			device = torch.device("mps")  # Apple Silicon GPU
		elif req_dev == "cuda" and torch.cuda.is_available():
			device = torch.device("cuda")  # NVIDIA GPU
		else:
			device = torch.device("cpu")  # Defaults to CPU
	else:
		# Automatic device detection
		if torch.backends.mps.is_available():
			# Check and use Apple Silicon GPU
			# https://pytorch.org/docs/stable/notes/mps.html
			device = torch.device("mps")
		elif torch.cuda.is_available():
			# The provided code for CUDA
			device = torch.device("cuda")
		else:
			# Default to CPU if no accelerator available
			device = torch.device("cpu")

	return device


def make_dataloaders(batch_size=32):

    transforms = v2.Compose([
        v2.Resize(size=(512, 512)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.Grayscale(num_output_channels=1),  # ensure single channel
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5]),  # 1-channel grayscale
        ])
    
    images_path = Path("/gscratch/cheme/agolan/data/preprocessed_data")
    # images = data_io.load_images_from_directory(images_path)
    
    # DEBUG: Check directory structure
    print(f"\n{'='*60}")
    print(f"DEBUG: Checking directory structure")
    print(f"{'='*60}")
    print(f"Base path: {images_path}")
    print(f"Path exists: {images_path.exists()}")
    print(f"Is directory: {images_path.is_dir()}")
    
    # Check subdirectories
    for label in ['mice', 'rat', 'ferret', 'human', 'pig']:
        subdir = images_path / label
        print(f"\n{label} directory:")
        print(f"  Path: {subdir}")
        print(f"  Exists: {subdir.exists()}")
        if subdir.exists():
            npy_files = list(subdir.rglob("*.npy"))
            print(f"  .npy files found: {len(npy_files)}")
            if npy_files:
                print(f"  Example files: {[f.name for f in npy_files[:3]]}")
    print(f"{'='*60}\n")

    microglia_dataset = MicrogliaDataset(images_path,
                                         train=True,
                                         labels=['mice', 'rat', 'gyrified'],
                                         transform=transforms,
                                         merge_map={
                                         'human' : 'gyrified',
                                         'ferret' : 'gyrified',
                                         'pig' : 'gyrified',
                                        })
    print(microglia_dataset.length)

    data_train, data_val = generate_dataloaders(microglia_dataset, num_workers=2, batch_size=batch_size)

    return data_train, data_val
    
    
def visualize_embeddings(embedding_model, dataloader, device, epoch, method="both"):
    """
    Collect embeddings from the val set and plot UMAP and/or t-SNE.
    method: "umap", "tsne", or "both"
    """
    embedding_model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for _, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            embs = embedding_model(images).cpu().numpy()
            all_embeddings.append(embs)
            all_labels.extend(labels.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.array(all_labels)
    label_names = {0: "mice", 1: "rat", 2: "gyrified"}
    colors = ["steelblue", "tomato", "yellow"]

    Path("fig").mkdir(exist_ok=True)

    def _scatter(reduced, title, filepath):
        plt.figure(figsize=(7, 6))
        for cls_idx, cls_name in label_names.items():
            mask = all_labels == cls_idx
            plt.scatter(
                reduced[mask, 0], reduced[mask, 1],
                label=cls_name, alpha=0.6, s=18,
                color=colors[cls_idx]
            )
        plt.title(f"{title} — Epoch {epoch}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"[+] Saved {filepath}")

    if (method in ("tsne", "both")):
        perplexity = min(30, len(all_embeddings) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        reduced_tsne = tsne.fit_transform(all_embeddings)
        _scatter(reduced_tsne, "t-SNE of Embeddings", f"fig/tsne_epoch{epoch:03d}.png")

    if (method in ("umap", "both")):
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced_umap = reducer.fit_transform(all_embeddings)
        _scatter(reduced_umap, "UMAP of Embeddings", f"fig/umap_epoch_gyrified_lr003_{epoch}.png")
    """ elif method in ("umap", "both") and not UMAP_AVAILABLE:
        print("[!] umap-learn not installed. Run: pip install umap-learn") """


def freeze_backbone(model):
    """Freeze everything except the last two FC layers."""
    for param in model.cnn.cnn1.parameters():
        param.requires_grad = False
    for param in model.cnn.cnn2.parameters():
        param.requires_grad = False
    # fc1 and fc2 remain trainable
    print("[+] Backbone frozen. Only fc1 and fc2 are trainable.")

def train(model, embedding_model, weights, epochs, data, device, loss_func, optimizer, ce_weight=0.5, stage=1):
    """Train with SupConLoss on embeddings + CrossEntropy on logits to train classifier head.
    ce_weight controls the blend: total_loss = (1 - ce_weight) * supcon + ce_weight * ce
    """
    ce_loss_func = nn.CrossEntropyLoss()
    train_loss_epoch = []
    val_loss_epoch = []
    val_accuracy_epoch = []

    model.to(device)
    embedding_model.to(device)
    data_train, data_val = data

    pbar_epoch = tqdm.tqdm(iterable=range(epochs), colour="green", desc="Epoch")
    for epoch in pbar_epoch:
        train_loss_batch = []
        model.train()
        embedding_model.train()

        pbar_batch = tqdm.tqdm(total=len(data_train), colour="blue", desc="Batch", leave=False)
        for _, (images, labels) in enumerate(data_train):
            images = images.to(device)
            labels = labels.to(device)

            # CrossEntropy on logits trains the classifier head
            logits = model(images)
            ce_loss = ce_loss_func(logits, labels)

            # Stage 1: SupCon + CE. Stage 2: CE only
            if stage == 1:
                embeddings = embedding_model(images)
                supcon_loss = loss_func(embeddings, labels)
                loss = (1 - ce_weight) * supcon_loss + ce_weight * ce_loss
            else:
                loss = ce_loss
                
            train_loss_batch.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar_batch.set_postfix({"Loss": train_loss_batch[-1]})
            pbar_batch.update(1)

        pbar_batch.close()

        # Save weights once per epoch, not every batch
        if weights:
            torch.save(model.state_dict(), weights)

        # Validation once per epoch (not every batch)
        model.eval()
        embedding_model.eval()
        val_pred = []
        val_lbls = []
        val_loss_list = []
        with torch.no_grad():
            for _, (images, labels) in enumerate(data_val):
                images = images.to(device)
                labels = labels.to(device)
                val_embeddings = embedding_model(images)
                val_loss_list.append(loss_func(val_embeddings, labels).item())
                # Accuracy from full model logits
                logits = model(images)
                val_pred += logits.argmax(dim=1).cpu().tolist()
                val_lbls += labels.cpu().tolist()

        val_pred = np.array(val_pred)
        val_lbls = np.array(val_lbls)
        val_accuracy_epoch.append((val_pred == val_lbls).mean())
        val_loss_epoch.append(np.mean(val_loss_list))
        train_loss_epoch.append(np.mean(train_loss_batch))
        
        # Visualize embeddings every 5 epochs (or every epoch if you prefer)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            visualize_embeddings(
                embedding_model, data_val, device,
                epoch=epoch + 1,
                method="umap"  # "tsne", "umap", or "both"
            )

        pbar_epoch.set_postfix({
            "Loss": val_loss_epoch[-1],
            "Acc": val_accuracy_epoch[-1],
        })

        # Plot training curves
        Path("fig").mkdir(exist_ok=True)
        epoch_axis = range(1, len(val_accuracy_epoch) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epoch_axis, val_accuracy_epoch, marker='o', color='blue', label='Validation Accuracy')
        plt.title('Validation Accuracy per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.grid(True)
        plt.legend()
        plt.savefig('fig/validation.png')
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epoch_axis, val_loss_epoch, marker='o', color='blue', label='Validation Loss')
        plt.plot(epoch_axis, train_loss_epoch, marker='o', color='red', label='Training Loss')
        plt.title('Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig('fig/training.png')
        plt.close()

    return val_accuracy_epoch, val_loss_epoch, train_loss_epoch


if __name__ == "__main__":

    msg = "Microglia CNN training script"
    parser = argparse.ArgumentParser(description=msg)

    # Adding arguments
    parser.add_argument("-d", "--dev", "--device", 
					default=None, 
					choices=["cpu", "cuda", "mps"],
					type=str,
					help="Device to use for training [cpu, gpu, mps], defaults to automatic detection.")
    parser.add_argument("-e", "--epochs",
					default=10, 
					type=int, 
					help="Number of epochs to train the model.")
    parser.add_argument("--split",
					default=0.8,
					type=float,
					help="Train-test split ratio, defaults to 0.8 (80%% train, 20%% validation).")
    parser.add_argument("-b", "--batch_size",
					default=64, 
					type=int, 
					help="Batch size for training.")
    parser.add_argument("-l", "--learning_rate",
					default=0.005, 
					type=float, 
					help="Learning rate for the optimizer.")
    parser.add_argument("--decay",
					default=0.0001, 
					type=float, 
					help="Weight decay for the optimizer.")
    parser.add_argument("-w", "--weights",
					default=None, 
					type=str, 
					help="Path to store or load the model weights file, if any.")
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

	# Set up logging
    logging.basicConfig(format="[+] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET if args.verbose else logging.WARNING)

    ### Figure out the device to use for training
    device = device_check(args.dev)
    logger.info(f"Using device: {device}")
	

    data_train, data_val = make_dataloaders(batch_size=args.batch_size)
    model = MicrogliaCNN()
    embedding_model = MicrogliaEmbeddingWrapper(model)  # 500-d normalized embeddings for SupConLoss
    if SupConLoss is None:
        raise ImportError("Install pytorch-metric-learning for SupConLoss: pip install -e '.[torch]'")
    loss_func = SupConLoss(temperature=0.07)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(embedding_model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.decay
    )

    # --- Stage 1: SupCon + CE, all layers trainable ---
    print("[+] Stage 1: Supervised contrastive learning")
    val_accuracy, val_loss, train_loss = train(
        model=model,
        embedding_model=embedding_model,
        weights=args.weights,
        epochs=100,
        device=device,
        data=[data_train, data_val],
        loss_func=loss_func,
        optimizer=optimizer,
        ce_weight=0.3,
        stage=1,
    )

    torch.save(model.state_dict(), "weights_stage1.pt")
    print("[+] Stage 1 weights saved to weights_stage1.pt")

    # --- Stage 2: Freeze backbone, fine-tune head ---
    print("[+] Stage 2: Fine-tuning classifier head")
    freeze_backbone(model)

    optimizer_ft = torch.optim.Adam(
        filter(lambda p: p.requires_grad,
               list(model.parameters()) + list(embedding_model.parameters())),
        lr=args.learning_rate / 10,
        weight_decay=args.decay
    )

    val_accuracy_ft, val_loss_ft, train_loss_ft = train(
        model=model,
        embedding_model=embedding_model,
        weights=args.weights,
        epochs=args.epochs,
        device=device,
        data=[data_train, data_val],
        loss_func=loss_func,
        optimizer=optimizer_ft,
        ce_weight=1.0,
        stage=2,
    )

    # --- Final embeddings ---
    print("[+] Generating final embeddings")
    visualize_embeddings(embedding_model, data_val, device, epoch="final", method="umap")