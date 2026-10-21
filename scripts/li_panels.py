"""
Show, for a few images per condition: the original image, the brightness-normalised image, and
Li's threshold with no noise reduction. Made for the fluorescent_micro_images layout:
    <data_dir>/<condition>/<name>.tif                      raw images, (3, H, W)
    <data_dir>/li_thresh/<condition>/<name>_li_thresh.npy  saved masks (shown when found)

One row per image, columns:
    original       the raw channel on a plain linear scale, its own min to max. Looks dark:
                   a few very bright pixels set the top of the scale.
    normalised     1st percentile -> 0, 99.5th percentile -> 1, clipped. Every image ends up on
                   the same 0-1 scale, whatever the microscope brightness was.
    Li, raw        image_prep.li_mask with background removal and clean-up switched off: only
                   Li's threshold, one cutoff per image. Computed on the original values.
    saved mask     the mask already in li_thresh/, for comparison with the raw threshold.
Plus a first row with all three channels of the first image, to check the channel order.

Channels (datadescriptions.xlsx): DAPI (blue), Iba1 (green), PI (red). Green is the middle
channel whether the file is stored blue-green-red or red-green-blue, so --channel 1 is Iba1.

Usage (from the saffron folder):
    python scripts/li_panels.py /path/to/fluorescent_micro_images
    python scripts/li_panels.py /path/to/fluorescent_micro_images --channel 0 --n 4 --out_dir li_panels_dapi

Writes one .png per condition and all_conditions.pdf. Nothing in the data folder is modified.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display on the cluster
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import tifffile

from image_prep import li_mask

LOW_PERCENTILE, HIGH_PERCENTILE = 1, 99.5
MAX_SIDE = 1024  # larger images (the one 3072 x 3072 tile scan) are skipped


def normalise(image):
    """Values scaled to 0-1: the 1st percentile becomes 0, the 99.5th becomes 1, the rest clipped."""
    image = image.astype(np.float64)
    low, high = np.percentile(image, [LOW_PERCENTILE, HIGH_PERCENTILE])
    if high <= low:
        return np.zeros_like(image)                    # flat image: nothing to stretch
    return np.clip((image - low) / (high - low), 0, 1)


def raw_li(image):
    """Li's threshold alone: no background removal, no removal of specks, no hole filling."""
    return li_mask(image, background_sigma=0, min_size=0, max_hole=0)


def show(ax, image, title, **imshow_args):
    ax.imshow(image, cmap="gray", interpolation="nearest", **imshow_args)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def plot_condition(condition, tif_paths, mask_dir, channel):
    stacks = [tifffile.imread(p) for p in tif_paths]
    figure = plt.figure(figsize=(11, 2.9 * (len(stacks) + 1) + 0.4), layout="constrained")
    figure.suptitle(f"{condition}   channel {channel}", fontsize=13)
    rows = figure.subfigures(len(stacks) + 1, 1)

    # Row 0: every channel of the first image, normalised, to check which channel is which
    first = stacks[0]
    axes = rows[0].subplots(1, 4)
    for c in range(4):
        if c < first.shape[0]:
            show(axes[c], normalise(first[c]), f"channel {c} (normalised)")
        else:
            axes[c].axis("off")
    rows[0].suptitle(f"channel check: {tif_paths[0].name}", fontsize=8, x=0.01, ha="left")

    for row, (path, stack) in zip(rows[1:], zip(tif_paths, stacks)):
        image = stack[channel]
        mask = raw_li(image)
        saved = mask_dir / f"{path.stem}_li_thresh.npy"
        axes = row.subplots(1, 4)
        show(axes[0], image, f"original  {image.dtype}\nmin {image.min()}  max {image.max()}  (linear)",
             vmin=image.min(), vmax=image.max())
        show(axes[1], normalise(image), f"normalised\n{LOW_PERCENTILE}st-{HIGH_PERCENTILE}th percentile -> 0-1",
             vmin=0, vmax=1)
        show(axes[2], mask, f"Li, no noise reduction\n{mask.mean():.1%} of pixels above the cutoff",
             vmin=0, vmax=1)
        if saved.exists():
            saved_mask = np.load(saved) > 0
            show(axes[3], saved_mask, f"saved mask (li_thresh/)\n{saved_mask.mean():.1%} of pixels",
                 vmin=0, vmax=1)
        else:
            axes[3].axis("off")
            axes[3].set_title("no saved mask found", fontsize=8)
        row.suptitle(path.name, fontsize=8, x=0.01, ha="left")
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", help="the fluorescent_micro_images folder")
    parser.add_argument("--channel", default=1, type=int, help="channel to show: 1 = Iba1 (microglia), the default")
    parser.add_argument("--n", default=3, type=int, help="images per condition")
    parser.add_argument("--seed", default=0, type=int, help="change to see different images")
    parser.add_argument("--out_dir", default="li_panels", help="folder to save the pictures in")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"not a folder: {data_dir}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with PdfPages(out_dir / "all_conditions.pdf") as pdf:
        for condition_dir in sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name != "li_thresh"):
            tifs = sorted(p for p in condition_dir.glob("*.tif") if not p.name.startswith("."))
            # Read only the header to skip the tile scan without loading it
            usable = []
            for p in tifs:
                with tifffile.TiffFile(p) as tif:
                    if max(tif.series[0].shape[-2:]) <= MAX_SIDE:
                        usable.append(p)
            print(f"{condition_dir.name}: {len(tifs)} .tif, {len(tifs) - len(usable)} skipped as too large")
            if not usable:
                continue
            picked = sorted(rng.choice(len(usable), size=min(args.n, len(usable)), replace=False))
            figure = plot_condition(condition_dir.name, [usable[i] for i in picked],
                                    data_dir / "li_thresh" / condition_dir.name, args.channel)
            figure.savefig(out_dir / f"{condition_dir.name}.png", dpi=110)
            pdf.savefig(figure)
            plt.close(figure)

    print(f"\nsaved one .png per condition and all_conditions.pdf in {out_dir}/")


if __name__ == "__main__":
    main()
