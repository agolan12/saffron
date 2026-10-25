"""
Saliency maps: which parts of an image does a trained model use for its decision?

For a few validation images per class it draws three panels:
    0. (only with --equalize or --li_threshold) the original image, before that preprocessing
    1. the image as the model sees it
    2. gradient map:  how much the decision changes when each single pixel changes
                      (fine detail; averaged over 20 slightly noised copies, "SmoothGrad")
    3. occlusion map: how much the model's confidence drops when a 64x64 square of the
                      image is covered up (coarse, but the most direct evidence)
and prints one number per image: the share of the gradient map that falls on cell pixels
(the brightest 15% of the image). If the model ignored the difference between cells and
background that share would be 0.15; well above it means the model looks at the cells.

Usage (on Hyak, from the saffron folder), with weights saved by supconloss_test_old.py:
    python scripts/saliency_map.py --weights weights/job12345_task0_final.pt
    python scripts/saliency_map.py --weights weights/job12345_task1_final.pt --equalize

IMPORTANT: pass the same --equalize and architecture options the run was trained with
(they are printed at the top of its logs_output/*.out file).
Writes saliency_<weights name>.png and prints a summary.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display on the cluster
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import v2

# Add src to path so we can import saffron (script lives in scripts/, package in src/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from saffron.data.datasets import MicrogliaDataset, split_indices_by_image
from supconloss_test_old import (CLASS_SETS, DEFAULT_DATA_DIR, ConfigurableMicrogliaCNN, EqualizeHistogram,
                                 LiThreshold, device_check)
from image_prep import LI_BACKGROUND_SIGMA, LI_MIN_SIZE, LI_MAX_HOLE

CELL_SHARE = 0.15   # the brightest 15% of pixels are treated as "cell"


def gradient_map(model, image, target, samples=20, noise=0.1):
    """SmoothGrad: average of |d(score of the predicted class) / d(pixel)| over noised copies."""
    total = torch.zeros_like(image)
    for _ in range(samples):
        noisy = (image + noise * torch.randn_like(image)).requires_grad_(True)
        score = model(noisy.unsqueeze(0))[0, target]
        total += torch.autograd.grad(score, noisy)[0].abs()
    return (total / samples).squeeze(0)


def occlusion_map(model, image, target, patch=64, step=32, batch_size=32):
    """Drop in the predicted class's probability when each patch is replaced by the image's median value."""
    _, height, width = image.shape
    fill = image.median()
    with torch.no_grad():
        base = torch.softmax(model(image.unsqueeze(0)), dim=1)[0, target]
        positions = [(top, left) for top in range(0, height - patch + 1, step)
                     for left in range(0, width - patch + 1, step)]
        drop_sum = torch.zeros(height, width, device=image.device)
        covered = torch.zeros(height, width, device=image.device)
        for start in range(0, len(positions), batch_size):
            chunk = positions[start:start + batch_size]
            batch = image.unsqueeze(0).repeat(len(chunk), 1, 1, 1)
            for i, (top, left) in enumerate(chunk):
                batch[i, :, top:top + patch, left:left + patch] = fill
            drops = base - torch.softmax(model(batch), dim=1)[:, target]
            for (top, left), drop in zip(chunk, drops):
                drop_sum[top:top + patch, left:left + patch] += drop
                covered[top:top + patch, left:left + patch] += 1
    return drop_sum / covered.clamp(min=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="model weights (.pt) saved by supconloss_test_old.py")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="folder that holds the species folders")
    parser.add_argument("--equalize", action="store_true", help="use if the run was trained with --equalize")
    parser.add_argument("--li_threshold", action="store_true", help="use if the run was trained with --li_threshold")
    parser.add_argument("--li_background_sigma", default=LI_BACKGROUND_SIGMA, type=float)
    parser.add_argument("--li_min_size", default=LI_MIN_SIZE, type=int)
    parser.add_argument("--li_max_hole", default=LI_MAX_HOLE, type=int)
    parser.add_argument("--classes", default="all", choices=sorted(CLASS_SETS),
                        help="the --classes value the run was trained with")
    parser.add_argument("--per_class", default=4, type=int, help="validation images to show per class")
    parser.add_argument("--seed", default=0, type=int, help="which random validation images are picked")
    parser.add_argument("--dev", default=None, choices=["cpu", "cuda", "mps"], help="device, default automatic")
    parser.add_argument("-o", "--output", default=None, help="image file to write")
    # architecture: must match the trained run (defaults are the original MicrogliaCNN)
    parser.add_argument("--kernel_size", default=12, type=int)
    parser.add_argument("--stride", default=2, type=int)
    parser.add_argument("--pool", default="mixed", choices=["mixed", "avg", "max"])
    parser.add_argument("--pool_size", default=12, type=int)
    parser.add_argument("--image_size", default=512, type=int, help="the --image_size value the run was trained with")
    args = parser.parse_args()
    output = args.output or f"saliency_{Path(args.weights).stem}.png"
    LABELS, MERGE_MAP = CLASS_SETS[args.classes]

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = device_check(args.dev)

    model = ConfigurableMicrogliaCNN(kernel_size=args.kernel_size, stride=args.stride, pool=args.pool,
                                     pool_size=args.pool_size, input_size=args.image_size,
                                     num_classes=len(LABELS))
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.to(device).eval()

    # Same preprocessing as validation in the training script
    steps = [EqualizeHistogram()] if args.equalize else []
    if args.li_threshold:
        steps.append(LiThreshold(args.li_background_sigma, args.li_min_size, args.li_max_hole))
    transform = v2.Compose(steps + [
        v2.Resize(size=(args.image_size, args.image_size)),
        v2.Grayscale(num_output_channels=1),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5]),
    ])
    dataset = MicrogliaDataset(args.data_dir, labels=LABELS, merge_map=MERGE_MAP, transform=transform)
    # With preprocessing, also load each image without it, to show next to what the model sees.
    # Same folder and labels, so its samples are in the same order as `dataset`.
    show_original = args.equalize or args.li_threshold
    if show_original:
        originals = MicrogliaDataset(args.data_dir, labels=LABELS, merge_map=MERGE_MAP, transform=v2.Compose([
            v2.Resize(size=(args.image_size, args.image_size)),
            v2.Grayscale(num_output_channels=1),
            v2.ToDtype(torch.float32, scale=True),
        ]))
    _, val_indices = split_indices_by_image(dataset, val_split=0.2, seed=42)

    rng = np.random.default_rng(args.seed)
    chosen = []
    for class_index in range(len(LABELS)):
        of_class = [i for i in val_indices if dataset.samples[i][1] == class_index]
        chosen += list(rng.choice(of_class, size=min(args.per_class, len(of_class)), replace=False))

    columns = 4 if show_original else 3
    fig, axes = plt.subplots(len(chosen), columns, figsize=(3.5 * columns, 3.6 * len(chosen)), squeeze=False)
    print(f"\n{'true':<10}{'predicted':<11}{'confidence':>11}{'gradient on cells':>19}{'largest occlusion drop':>24}   file")
    shares = {label: [] for label in LABELS}
    for row, index in zip(axes, chosen):
        image, label = dataset[int(index)]
        image = image.to(device)
        with torch.no_grad():
            probabilities = torch.softmax(model(image.unsqueeze(0)), dim=1)[0]
        predicted = int(probabilities.argmax())

        gradients = gradient_map(model, image, predicted).cpu().numpy()
        occlusion = occlusion_map(model, image, predicted).cpu().numpy()
        shown = (image.squeeze(0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)   # undo Normalize for display

        # "cell" pixels: the white part of a Li mask, otherwise the brightest 15% of the image
        cells = shown > 0.5 if args.li_threshold else shown >= np.quantile(shown, 1 - CELL_SHARE)
        chance = cells.mean()                          # the share expected if the model ignored cells
        on_cells = gradients[cells].sum() / gradients.sum()
        shares[LABELS[label]].append((on_cells, chance))
        path = Path(dataset.samples[int(index)][0])
        species = path.relative_to(dataset.data_dir).parts[0]
        print(f"{LABELS[label]:<10}{LABELS[predicted]:<11}{probabilities[predicted].item():>11.2f}{on_cells:>19.2f}"
              f"{occlusion.max():>24.2f}   {species}/{path.name[:60]}")

        title = (f"{species}: true {LABELS[label]}, predicted {LABELS[predicted]} "
                 f"({probabilities[predicted].item():.2f})")
        if show_original:
            # true brightness (no contrast stretch), so differences between species stay visible
            original = originals[int(index)][0].squeeze(0).numpy()
            row[0].imshow(original, cmap="gray", vmin=0, vmax=1)
            row[0].set_title(f"{title}\noriginal", fontsize=8, loc="left")
            row = row[1:]
            steps_name = " + ".join(name for name, on in [("equalized", args.equalize),
                                                           ("Li mask", args.li_threshold)] if on)
            row[0].imshow(shown, cmap="gray", vmin=0, vmax=1)
            row[0].set_title(f"model input ({steps_name})", fontsize=8, loc="left")
        else:
            row[0].imshow(shown, cmap="gray", vmin=0, vmax=1)
            row[0].set_title(title, fontsize=8, loc="left")
        row[1].imshow(shown, cmap="gray", vmin=0, vmax=1)
        row[1].imshow(gradients, cmap="magma", alpha=0.75, vmax=np.quantile(gradients, 0.995))
        row[1].set_title(f"gradient map: {on_cells:.0%} on cell pixels (chance {chance:.0%})", fontsize=8, loc="left")
        limit = max(abs(occlusion).max(), 0.05)   # drops under 0.05 stay pale, so noise does not look like signal
        row[2].imshow(shown, cmap="gray", vmin=0, vmax=1)
        row[2].imshow(occlusion, cmap="coolwarm", alpha=0.6, vmin=-limit, vmax=limit)
        row[2].set_title(f"occlusion: red = covering lowers confidence (max drop {occlusion.max():.2f})",
                         fontsize=8, loc="left")
    for ax in axes.flat:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output, dpi=110)
    print(f"\nwrote {output}")
    print("\nShare of the gradient map on cell pixels, next to the share of the image those pixels cover.\n"
          "Equal numbers mean no preference for cells; a higher first number means the model looks at cells.")
    for label, values in shares.items():
        if values:
            on, expected = np.array(values).T
            print(f"  {label:<10} on cells {on.mean():.2f} (range {on.min():.2f} to {on.max():.2f})"
                  f"   cells cover {expected.mean():.2f} of the image")


if __name__ == "__main__":
    main()
