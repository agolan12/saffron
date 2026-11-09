"""
Shortcut test for the fluorescent_micro_images data set: how well can the conditions be told
apart from brightness alone, after each of six ways of preparing the images?

Same test as shortcut_baseline.py (brightness-only numbers, a random forest, the
train/validation split of split_indices_by_image), but for this data set's layout:
    <data_dir>/<condition>/<name>.tif                      raw images, (3, H, W)
    <data_dir>/li_thresh/<condition>/<name>_li_thresh.npy  saved masks
Every folder except li_thresh/ is a condition (class). One channel is used, Iba1 by default.
Images larger than 1024 pixels on a side (the tile scan) are skipped, as in li_panels.py.

--prep, the image preparation (one per array task in saffron_fluorescent_shortcut_array.slurm):
    raw              nothing: the stored values, divided by the largest value of their type
    normalised       1st percentile -> 0, 99.5th -> 1, clipped (as in li_panels.py)
    denoised         background removal (the image minus a Gaussian blur of width 16 px, step 1
                     of image_prep.li_mask), then normalised
    li_raw           Li's threshold only: no background removal, no speck removal, no hole filling
    li_denoised      image_prep.li_mask with its defaults: background removal, Li, then specks
                     under 30 px removed and holes under 30 px filled
    li_saved         the masks already in li_thresh/. Images with no saved mask are left out
                     (the count is printed).

How to read it: as in shortcut_baseline.py, use the "balanced" column. Near 1 / number of
conditions = brightness says nothing; well above it = that much needs no cell shape.
For the masks the only brightness left is the share of the image marked as cell.

Usage (on Hyak, from the saffron folder; CPU only):
    python scripts/shortcut_baseline_fluorescent.py /path/to/fluorescent_micro_images --prep li_raw
Writes <out_dir>/shortcut_fluorescent_<prep>.txt (the same text that is printed).

Clean images only: --review_csv fluorescent_review.csv keeps only the images labelled "good" there
(--keep good unsure to keep both); images missing from the CSV are left out too. The result
then goes to shortcut_fluorescent_<prep>_good.txt.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

# Add src to path so we can import saffron (script lives in scripts/, package in src/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from image_prep import li_mask, LI_BACKGROUND_SIGMA
from saffron.data.datasets import split_indices, sample_id
from pool_folds import summarize

PREPS = ["raw", "normalised", "denoised", "li_raw", "li_denoised", "li_saved"]
BINS = 32                                        # same numbers as shortcut_baseline.py
PERCENTILES = [1, 5, 25, 50, 75, 95, 99, 99.9]
LOW_PERCENTILE, HIGH_PERCENTILE = 1, 99.5        # same stretch as li_panels.py
MAX_SIDE = 1024

# Class sets for this data set: name -> (class names in label order, folder -> class it is merged
# into). Only the folders named here are read. supconloss_test_old.py imports these for --classes,
# so the CNN and this shortcut test use the same classes.
FLUORESCENT_CLASS_SETS = {
    "fluorescent": (["nt_control", "ogd_0.5", "ogd_1.5", "ogd_3", "ogd_azo_1.5", "ogd_sod_3"], {}),
    # OGD (all three durations, no drug) vs untreated control; the drug folders are left out
    "fluorescent_ogd_vs_control": (["nt_control", "ogd"], {"ogd_0.5": "ogd", "ogd_1.5": "ogd", "ogd_3": "ogd"}),
    "fluorescent_ogd_3_vs_control": (["nt_control", "ogd_3"], {}),
}


def normalise(image):
    """Values scaled to 0-1: the 1st percentile becomes 0, the 99.5th becomes 1, the rest clipped."""
    low, high = np.percentile(image, [LOW_PERCENTILE, HIGH_PERCENTILE])
    if high <= low:
        return np.zeros_like(image)                    # flat image: nothing to stretch
    return np.clip((image - low) / (high - low), 0, 1)


def prepare(path, prep, channel, mask_dir):
    """One image as a 2-D float array after `prep`, or None for li_saved when no mask was saved."""
    if prep == "li_saved":
        saved = mask_dir / path.parent.name / f"{path.stem}_li_thresh.npy"
        return (np.load(saved).squeeze() > 0).astype(np.float64) if saved.exists() else None
    stack = tifffile.imread(path)
    image = stack[channel].astype(np.float64)
    if prep == "raw":
        return image / np.iinfo(stack.dtype).max if np.issubdtype(stack.dtype, np.integer) else image
    if prep == "normalised":
        return normalise(image)
    if prep == "denoised":
        return normalise(image - ndimage.gaussian_filter(image, LI_BACKGROUND_SIGMA))
    if prep == "li_raw":
        return li_mask(image, background_sigma=0, min_size=0, max_hole=0).astype(np.float64)
    if prep == "li_denoised":
        return li_mask(image).astype(np.float64)
    raise ValueError(prep)


def load_review(csv_path):
    """{'<condition>/<name>.tif': label} from a review_images.py CSV (file paths relative to data_dir).

    The header row may be anywhere: a CSV re-sorted in a spreadsheet has it among the data rows."""
    import csv
    with open(csv_path, newline="") as f:
        rows = [row for row in csv.reader(f) if row]
    header = next((row for row in rows if "file" in row and "label" in row), None)
    if header is None:
        raise SystemExit(f"{csv_path}: no header row with 'file' and 'label' columns")
    file_col, label_col = header.index("file"), header.index("label")
    return {row[file_col]: row[label_col] for row in rows if row is not header}


def review_label(review, path):
    """The review label of an image at <data_dir>/<condition>/<name>.tif, or 'not reviewed'."""
    return review.get(f"{path.parent.name}/{path.name}", "not reviewed")


def check_review_matches(review, data_dir, conditions):
    """Stop if no image under data_dir is in the review CSV (wrong CSV or wrong data_dir)."""
    on_disk = {f"{c}/{p.name}" for c in conditions for p in (data_dir / c).glob("*.tif")}
    if review and not on_disk & review.keys():
        raise SystemExit(f"none of the {len(review)} files in the review CSV are under {data_dir}: "
                         "is it the CSV for this data set?")


def brightness_numbers(image, top):
    """[mean, std, 32 histogram shares over 0..top, 8 percentiles], as in shortcut_baseline.py."""
    image = image.ravel()
    histogram = np.histogram(image, bins=BINS, range=(0.0, top))[0] / image.size
    return np.concatenate([[image.mean(), image.std()], histogram, np.percentile(image, PERCENTILES)])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", help="the fluorescent_micro_images folder")
    parser.add_argument("--prep", required=True, choices=PREPS, help="image preparation, see above")
    parser.add_argument("--channel", default=1, type=int, help="channel to use: 1 = Iba1 (microglia), the default")
    parser.add_argument("--out_dir", default="shortcut_fluorescent", help="folder for the results text file")
    parser.add_argument("--review_csv", help="review_images.py labels; only images labelled --keep are used")
    parser.add_argument("--keep", nargs="+", default=["good"],
                        help="review labels to keep with --review_csv (default: good)")
    parser.add_argument("--split_by", default="image", choices=["image", "sample"],
                        help="image: one 80/20 split by image (the original); sample: cross-validation by "
                             "sample, as supconloss_test_old.py --split_by sample (all folds, results pooled)")
    parser.add_argument("--n_folds", default=4, type=int, help="--split_by sample: number of folds")
    parser.add_argument("--classes", choices=sorted(FLUORESCENT_CLASS_SETS),
                        help="class set, as --classes in supconloss_test_old.py "
                             "(default: every folder is its own class)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"not a folder: {data_dir}")
    mask_dir = data_dir / "li_thresh"
    conditions = sorted(p.name for p in data_dir.iterdir()
                        if p.is_dir() and p.name != "li_thresh" and not p.name.startswith("."))
    # conditions: the folders read; class_names: the classes they are sorted into
    if args.classes:
        class_names, merge_map = FLUORESCENT_CLASS_SETS[args.classes]
        wanted = [name for name in class_names if name not in merge_map.values()] + list(merge_map)
        missing = [name for name in wanted if name not in conditions]
        if missing:
            raise SystemExit(f"--classes {args.classes} needs folders that are not in {data_dir}: {missing}")
        conditions = sorted(wanted)
    else:
        class_names, merge_map = conditions, {}
    review = load_review(args.review_csv) if args.review_csv else None
    if review is not None:
        check_review_matches(review, data_dir, conditions)

    # Same files for every prep (li_saved then drops the ones with no mask)
    samples, too_large, dropped = [], 0, Counter()
    for condition in conditions:
        label = class_names.index(merge_map.get(condition, condition))
        for path in sorted((data_dir / condition).glob("*.tif")):
            if path.name.startswith("."):
                continue
            if review is not None and review_label(review, path) not in args.keep:
                dropped[review_label(review, path)] += 1
                continue
            with tifffile.TiffFile(path) as tif:       # header only: skip the tile scan without loading it
                if max(tif.series[0].shape[-2:]) > MAX_SIDE:
                    too_large += 1
                    continue
            samples.append((str(path), label))

    print(f"reading {len(samples)} images, prep '{args.prep}' ...", flush=True)
    images, kept, no_mask = [], [], 0
    for path, label in samples:
        image = prepare(Path(path), args.prep, args.channel, mask_dir)
        if image is None:
            no_mask += 1
            continue
        images.append(image)
        kept.append((path, label))
    # Histogram range: 0..1 for masks and normalised images. Raw values only use a small part
    # of their type's range, so their bands span 0 to the brightest value in the whole data set.
    top = max(image.max() for image in images) if args.prep == "raw" else 1.0
    features = np.stack([brightness_numbers(image, top) for image in images])
    del images

    dataset = SimpleNamespace(samples=kept, data_dir=data_dir)
    labels = np.array([label for _, label in kept])
    folders = np.array([Path(path).parent.name for path, _ in kept])
    # One (train, validation) pair for --split_by image; one per fold for --split_by sample, the same
    # folds as supconloss_test_old.py --split_by sample. Every image is validated at most once, so
    # the predictions of all folds are pooled and scored together.
    n_folds = args.n_folds if args.split_by == "sample" else 1
    folds = [split_indices(dataset, split=args.split_by, val_split=0.2, n_folds=args.n_folds, fold=k, seed=42)
             for k in range(n_folds)]
    val_indices = np.concatenate([val for _, val in folds])
    y_val = labels[val_indices]

    lines = []

    def out(text=""):
        print(text)
        lines.append(text)

    out("=" * 70)
    out(f"SHORTCUT TEST, fluorescent images: brightness alone   [prep: {args.prep}, channel {args.channel}]")
    out("=" * 70)
    out(f"data: {data_dir}")
    out(f"conditions: {', '.join(conditions)}")
    if args.classes:
        out(f"class set {args.classes}: " + "; ".join(
            f"{name} = {' + '.join(c for c in conditions if merge_map.get(c, c) == name)}" for name in class_names))
    out(f"skipped as larger than {MAX_SIDE} px: {too_large}" + (f"   no saved mask: {no_mask}" if args.prep == "li_saved" else ""))
    if review is not None:
        out(f"review filter: {args.review_csv}, kept {'/'.join(args.keep)}; left out: "
            + (", ".join(f"{n} {label}" for label, n in sorted(dropped.items())) or "none"))
    if args.split_by == "sample":
        out(f"split: {n_folds}-fold cross-validation by sample; every image validated once, results pooled")
        for k, (train, val) in enumerate(folds):
            out(f"  fold {k}: train {len(train)} images, validation {len(val)} images, samples "
                + ", ".join(sorted({sample_id(kept[i][0]) for i in val})))
    else:
        out(f"train images: {len(folds[0][0])}   validation images: {len(val_indices)}")
    out(f"always guessing the largest class: accuracy {np.bincount(y_val).max() / len(y_val):.3f}")

    versions = [("mean", slice(0, 1)), ("mean + spread", slice(0, 2)), ("histogram", slice(2, None))]
    out(f"\n{'version':<16}{'numbers':>8}{'accuracy':>10}{'balanced':>10}   recall per class: "
        + "  ".join(class_names))
    predictions = {}
    for name, columns in versions:
        print(f"training '{name}' ...", flush=True)
        # Same classifier as shortcut_baseline.py (n_jobs=1: no OpenMP, which hung on Hyak there)
        predicted = []
        for train, val in folds:
            model = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                           n_jobs=1, random_state=0)
            model.fit(features[train][:, columns], labels[train])
            predicted.append(model.predict(features[val][:, columns]))
        predicted = np.concatenate(predicted)
        predictions[name] = predicted
        recall = [np.mean(predicted[y_val == c] == c) if np.any(y_val == c) else np.nan
                  for c in range(len(class_names))]
        out(f"{name:<16}{features[:, columns].shape[1]:>8}{np.mean(predicted == y_val):>10.3f}"
            f"{balanced_accuracy_score(y_val, predicted):>10.3f}   "
            + "  ".join(f"{r:>{max(len(c), 5)}.3f}" for r, c in zip(recall, class_names)))
    out(f"\n'balanced' is the average of the per-class recalls; pure guessing gives {1 / len(class_names):.3f}.")

    out("\nConfusion matrix, histogram version (rows = true class, columns = predicted class)")
    matrix = confusion_matrix(y_val, predictions["histogram"], labels=range(len(class_names)))
    width = max(10, max(len(c) for c in class_names + conditions) + 2)
    out(f"{'':<{width}}" + "".join(f"{c:>{width}}" for c in class_names))
    for condition, row in zip(class_names, matrix):
        out(f"{condition:<{width}}" + "".join(f"{n:>{width}}" for n in row))

    if args.split_by == "sample":
        out("\nHistogram version, per fold and per sample:")
        fold_of = np.concatenate([[k] * len(val) for k, (_, val) in enumerate(folds)])
        rows = [{"file": kept[i][0], "sample": sample_id(kept[i][0]), "folder": folders[i],
                 "true": class_names[labels[i]], "predicted": class_names[p], "fold": str(k)}
                for i, p, k in zip(val_indices, predictions["histogram"], fold_of)]
        summary = summarize(rows, class_names)
        start = next(n for n, line in enumerate(summary) if "per fold:" in line or "per sample" in line)
        for line in summary[start:]:          # the overall numbers and matrix are already above
            out(line.strip("\n") if line.startswith("\n") else line)

    out("\nAverage value per condition (all images; for masks, the share of pixels marked as cell)")
    for condition in conditions:
        means = features[folders == condition, 0]
        if len(means):
            out(f"  {condition:<{width}}{len(means):>5} images   mean {means.mean():.4f}   middle half "
                f"{np.percentile(means, 25):.4f} to {np.percentile(means, 75):.4f}")

    suffix = f"_{args.classes}" if args.classes else ""
    suffix += f"_sample{n_folds}fold" if args.split_by == "sample" else ""
    suffix += f"_{'+'.join(args.keep)}" if review is not None else ""   # don't overwrite the all-images result
    out_txt = Path(args.out_dir) / f"shortcut_fluorescent_{args.prep}{suffix}.txt"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_txt}")


if __name__ == "__main__":
    main()
