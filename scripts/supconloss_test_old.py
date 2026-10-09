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

# fc1[0] is Linear(flattened_size, 500) and holds almost all of the weights. Above this
# many inputs it (plus its gradient and Adam state) no longer fits on an 11 GB GPU.
MAX_FLATTENED_SIZE = 600_000


class ConfigurableMicrogliaCNN(MicrogliaCNN):
    """
    MicrogliaCNN with its convolution and pooling settings exposed, for architecture sweeps.

    The defaults rebuild MicrogliaCNN exactly: 12x12 convolutions with stride 2, average
    pooling after the first convolution and max pooling after the second ("mixed").

    kernel_size, stride: used by both convolutions (padding stays 2, as in MicrogliaCNN)
    pool: "mixed" (avg then max), "avg" (both average) or "max" (both max)
    pool_size: pooling window, used by both pooling layers (pooling stride stays 2)
    """

    def __init__(self, kernel_size=12, stride=2, pool="mixed", pool_size=12, input_size=512, num_classes=6):
        nn.Module.__init__(self)  # the layers are built here instead of in MicrogliaCNN.__init__

        pool_padding = min(2, pool_size // 2)  # PyTorch requires pooling padding <= half the window

        def pool_layer(kind):
            layer = nn.AvgPool2d if kind == "avg" else nn.MaxPool2d
            return layer(kernel_size=pool_size, stride=2, padding=pool_padding)

        first_pool, second_pool = ("avg", "max") if pool == "mixed" else (pool, pool)

        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=kernel_size, stride=stride, padding=2),
            nn.ReLU(),
            pool_layer(first_pool),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=128, kernel_size=kernel_size, stride=stride, padding=2),
            nn.ReLU(),
            pool_layer(second_pool),
        )

        try:
            self.flattened_size = self._get_flattened_size(input_size)
        except RuntimeError as error:
            raise SystemExit(
                f"kernel_size={kernel_size}, stride={stride}, pool_size={pool_size} shrink a "
                f"{input_size}x{input_size} image to nothing ({error}). Use a smaller kernel, stride or pool size."
            )
        if self.flattened_size > MAX_FLATTENED_SIZE:
            raise SystemExit(
                f"kernel_size={kernel_size}, stride={stride}, pool_size={pool_size} leave {self.flattened_size:,} "
                f"features after the convolutions (limit {MAX_FLATTENED_SIZE:,}); the next layer would not fit "
                f"in GPU memory. Use a larger stride or pool size."
            )

        self.fc1 = nn.Sequential(
            nn.Linear(self.flattened_size, 500),
            nn.ReLU(),
            nn.Linear(500, num_classes),
        )


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

    # Training: random augmentation
    train_transforms = v2.Compose([
        v2.Resize(size=(512, 512)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.Grayscale(num_output_channels=1),  # ensure single channel
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5]),  # 1-channel grayscale
        ])

    # Validation: same preprocessing, nothing random, so metrics and UMAPs are repeatable
    val_transforms = v2.Compose([
        v2.Resize(size=(512, 512)),
        v2.Grayscale(num_output_channels=1),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5]),
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

    microglia_dataset = MicrogliaDataset(
        images_path,
        labels=["mice", "rat", "gyrified"],
        merge_map={"human": "gyrified", "ferret": "gyrified", "pig": "gyrified"},
    )

    # Dataset with all 5 species — used only for the species UMAP
    species_dataset = MicrogliaDataset(
        images_path,
        labels=["mice", "rat", "human", "ferret", "pig"],
    )

    print(microglia_dataset.length)

    # Transforms are applied after the split, so only the training half is augmented
    data_train, data_val = generate_dataloaders(
        microglia_dataset, num_workers=2, batch_size=batch_size,
        train_transform=train_transforms, val_transform=val_transforms,
    )
    _, species_val = generate_dataloaders(
        species_dataset, num_workers=2, batch_size=batch_size,
        train_transform=train_transforms, val_transform=val_transforms,
    )

    return data_train, data_val, species_val


def visualize_embeddings(
    embedding_model, dataloader, species_dataloader, device, epoch, job_id="local", task_id="0"
):
    """
    Produces two UMAPs saved to fig/:
      1. Gyrified vs non-gyrified (merged labels: mice, rat, gyrified)
      2. All 5 species separately (human, ferret, pig, rat, mice)

    Filenames include the SLURM job/task ID.
    """
    embedding_model.eval()
    Path("fig").mkdir(exist_ok=True)

    def collect_embeddings(loader):
        all_embs, all_lbls = [], []
        with torch.no_grad():
            for _, (images, labels) in enumerate(loader):
                images = images.to(device)
                embs = embedding_model(images).cpu().numpy()
                all_embs.append(embs)
                if isinstance(labels, torch.Tensor):
                    lbl = labels.detach().cpu().numpy()
                else:
                    lbl = np.asarray(labels)
                all_lbls.extend(lbl)
        return np.concatenate(all_embs, axis=0), np.array(all_lbls)

    def run_umap(embeddings):
        reducer = umap.UMAP(n_components=2, random_state=42)
        return reducer.fit_transform(embeddings)

    def scatter(reduced, labels_arr, label_names, colors, title, filepath):
        plt.figure(figsize=(7, 6))
        for cls_idx, cls_name in label_names.items():
            mask = labels_arr == cls_idx
            plt.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                label=cls_name,
                alpha=0.6,
                s=18,
                color=colors[cls_idx],
            )
        plt.title(f"{title} — Epoch {epoch}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"[+] Saved {filepath}")

    # UMAP 1: gyrified vs non-gyrified
    embs_merged, lbls_merged = collect_embeddings(dataloader)
    reduced_merged = run_umap(embs_merged)
    scatter(
        reduced_merged,
        lbls_merged,
        label_names={0: "mice", 1: "rat", 2: "gyrified"},
        colors={0: "steelblue", 1: "tomato", 2: "gold"},
        title="UMAP — Gyrified vs Non-Gyrified",
        filepath=f"fig/umap_gyrified_job{job_id}_task{task_id}_epoch{epoch:03d}.png",
    )

    # UMAP 2: all 5 species
    embs_species, lbls_species = collect_embeddings(species_dataloader)
    reduced_species = run_umap(embs_species)
    scatter(
        reduced_species,
        lbls_species,
        label_names={0: "mice", 1: "rat", 2: "human", 3: "ferret", 4: "pig"},
        colors={
            0: "steelblue",
            1: "tomato",
            2: "gold",
            3: "mediumseagreen",
            4: "mediumpurple",
        },
        title="UMAP — All Species",
        filepath=f"fig/umap_species_job{job_id}_task{task_id}_epoch{epoch:03d}.png",
    )


def freeze_backbone(model):
    """Freeze everything except the last two FC layers."""
    for param in model.cnn1.parameters():
        param.requires_grad = False
    for param in model.cnn2.parameters():
        param.requires_grad = False
    # fc trainable
    print("[+] Backbone frozen. Only fc stays trainable.")

def train(
    model,
    embedding_model,
    weights,
    epochs,
    data,
    species_val,
    device,
    loss_func,
    optimizer,
    ce_weight=0.5,
    stage=1,
    job_id="local",
    task_id="0",
):
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
                embedding_model,
                data_val,
                species_val,
                device,
                epoch=epoch + 1,
                job_id=job_id,
                task_id=task_id,
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
					default=0.003, 
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
    # training recipe
    parser.add_argument("--stage1_epochs",
					default=30,
					type=int,
					help="Number of epochs for stage 1 (--epochs sets stage 2).")
    parser.add_argument("--ce_weight",
					default=0.3,
					type=float,
					help="Stage 1 loss = (1 - ce_weight) * SupCon + ce_weight * cross-entropy. "
					     "0 trains stage 1 on SupCon only; accuracy is then meaningless until stage 2.")
    # architecture (defaults are the original MicrogliaCNN)
    parser.add_argument("--kernel_size",
					default=12,
					type=int,
					help="Kernel size of both convolutions.")
    parser.add_argument("--stride",
					default=2,
					type=int,
					help="Stride of both convolutions.")
    parser.add_argument("--pool",
					default="mixed",
					choices=["mixed", "avg", "max"],
					help="Pooling after the convolutions: mixed = average then max (the original), "
					     "avg = both average, max = both max.")
    parser.add_argument("--pool_size",
					default=12,
					type=int,
					help="Window size of both pooling layers.")

    args = parser.parse_args()

    # Print all parameters to stdout (shows up in .out file)
    print("=" * 60)
    print("RUN PARAMETERS")
    print("=" * 60)
    for arg, val in vars(args).items():
        print(f"  {arg:<20} = {val}")
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID", "local")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    print(f"  {'slurm_job_id':<20} = {job_id}")
    print(f"  {'slurm_task_id':<20} = {task_id}")
    print("=" * 60)

    # Set up logging
    logging.basicConfig(format="[+] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET if args.verbose else logging.WARNING)

    ### Figure out the device to use for training
    device = device_check(args.dev)
    logger.info(f"Using device: {device}")
	

    model = ConfigurableMicrogliaCNN(
        kernel_size=args.kernel_size,
        stride=args.stride,
        pool=args.pool,
        pool_size=args.pool_size,
    )
    print(f"[+] Model: {model.flattened_size:,} features after the convolutions, "
          f"{sum(p.numel() for p in model.parameters()):,} weights")

    data_train, data_val, species_val = make_dataloaders(batch_size=args.batch_size)
    embedding_model = MicrogliaEmbeddingWrapper(model)  # 500-d normalized embeddings for SupConLoss
    if SupConLoss is None:
        raise ImportError("Install pytorch-metric-learning for SupConLoss: pip install -e '.[torch]'")
    loss_func = SupConLoss(temperature=0.07)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.decay
    )

    # --- Stage 1: SupCon + CE, all layers trainable ---
    print(f"[+] Stage 1: Supervised contrastive learning ({args.stage1_epochs} epochs, ce_weight={args.ce_weight})")
    val_accuracy, val_loss, train_loss = train(
        model=model,
        embedding_model=embedding_model,
        weights=args.weights,
        epochs=args.stage1_epochs,
        device=device,
        data=[data_train, data_val],
        species_val=species_val,
        loss_func=loss_func,
        optimizer=optimizer,
        ce_weight=args.ce_weight,
        stage=1,
        job_id=job_id,
        task_id=task_id,
    )
    torch.save(model.state_dict(), "weights_stage1.pt")
    print("[+] Stage 1 weights saved to weights_stage1.pt")

    # --- Stage 2: Freeze backbone, fine-tune head ---
    print("[+] Stage 2: Fine-tuning classifier head")
    freeze_backbone(model)

    optimizer_ft = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
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
        species_val=species_val,
        loss_func=loss_func,
        optimizer=optimizer_ft,
        ce_weight=1.0,
        stage=2,
        job_id=job_id,
        task_id=task_id,
    )

    # --- Final embeddings ---
    print("[+] Generating final embeddings")
    visualize_embeddings(
        embedding_model,
        data_val,
        species_val,
        device,
        epoch=999,
        job_id=job_id,
        task_id=task_id,
    )