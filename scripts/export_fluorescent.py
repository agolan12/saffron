"""
Write one image preparation of the fluorescent_micro_images data set as ready-to-train files:
    <out_dir>/<condition>/<name>.npy                 2-D float32, 0..1 (masks: 0 or 1)
so supconloss_test_old.py and saliency_map.py can read it with --data_dir <out_dir>.

The preparation (--prep, same choices as shortcut_baseline_fluorescent.py and the same code) is
applied to the whole image first, so Li's cutoff and the normalisation are chosen from the whole
image, exactly as in the shortcut test. Then every image is cut into quarters, and quarters into
quarters, until the pieces are --piece_size on a side: with --piece_size 256 a 1024 x 1024 image
gives 16 pieces and a 512 x 512 image 4. Pieces are named like tiff_quadrant_split.py does
(<name>_top_left, <name>_top_left_bottom_right, ...), so split_indices_by_image keeps every piece
of one image on the same side of the train/validation split. Images that do not halve down to
--piece_size are written whole (the training script resizes them) and counted in the printout.
Images larger than 1024 px are skipped (the tile scan), and for li_saved so are images with no
saved mask.

Usage (on Hyak, from the saffron folder; CPU, a few minutes):
    python scripts/export_fluorescent.py /path/to/fluorescent_micro_images --prep li_raw \
        --piece_size 256 --out_dir /gscratch/cheme/agolan/data/fluorescent_prepped/li_raw_256
Train on the result with --image_size equal to --piece_size.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile

from shortcut_baseline_fluorescent import PREPS, MAX_SIDE, prepare

QUADRANTS = {"top_left": (0, 0), "top_right": (0, 1), "bottom_left": (1, 0), "bottom_right": (1, 1)}


def cut(image, name, size):
    """[(name, piece)]: `image` cut into quarters repeatedly until the pieces are size x size."""
    if image.shape[0] == size:
        return [(name, image)]
    half = image.shape[0] // 2
    pieces = []
    for suffix, (row, col) in QUADRANTS.items():
        pieces += cut(image[row * half:(row + 1) * half, col * half:(col + 1) * half], f"{name}_{suffix}", size)
    return pieces


def halves_down_to(shape, size):
    """True when a square image of this shape reaches size x size by halving."""
    side = shape[0]
    if shape[0] != shape[1] or side < size:
        return False
    while side > size and side % 2 == 0:
        side //= 2
    return side == size


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", help="the fluorescent_micro_images folder")
    parser.add_argument("--prep", required=True, choices=PREPS, help="image preparation")
    parser.add_argument("--out_dir", required=True, help="folder to write <condition>/<name>.npy into")
    parser.add_argument("--piece_size", default=512, type=int,
                        help="side of the pieces in pixels (512, 256, 128, ...); train with the same --image_size")
    parser.add_argument("--channel", default=1, type=int, help="channel to use: 1 = Iba1 (microglia), the default")
    args = parser.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"not a folder: {data_dir}")
    conditions = sorted(p.name for p in data_dir.iterdir()
                        if p.is_dir() and p.name != "li_thresh" and not p.name.startswith("."))

    for condition in conditions:
        (out_dir / condition).mkdir(parents=True, exist_ok=True)
        sizes, written, whole, too_large, no_mask = Counter(), 0, 0, 0, 0
        for path in sorted((data_dir / condition).glob("*.tif")):
            if path.name.startswith("."):
                continue
            with tifffile.TiffFile(path) as tif:       # header only: skip the tile scan without loading it
                if max(tif.series[0].shape[-2:]) > MAX_SIDE:
                    too_large += 1
                    continue
            image = prepare(path, args.prep, args.channel, data_dir / "li_thresh")
            if image is None:
                no_mask += 1
                continue
            image = image.astype(np.float32)
            sizes[image.shape] += 1
            if halves_down_to(image.shape, args.piece_size):
                for name, piece in cut(image, path.stem, args.piece_size):
                    np.save(out_dir / condition / f"{name}.npy", piece)
                    written += 1
            else:
                np.save(out_dir / condition / f"{path.stem}.npy", image)
                written += 1
                whole += 1
        size_text = ", ".join(f"{h}x{w}: {n}" for (h, w), n in sorted(sizes.items()))
        print(f"{condition:<14} {sum(sizes.values()):>4} images ({size_text}) -> {written} files"
              f"   ({whole} written whole)   skipped: {too_large} too large" + (f", {no_mask} no saved mask" if args.prep == "li_saved" else ""))

    print(f"\nwrote {args.prep} to {out_dir}/")


if __name__ == "__main__":
    main()
