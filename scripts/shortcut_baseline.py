"""
Shortcut test: how well can the classes be told apart from brightness alone?

Every image is reduced to a few numbers that describe how bright its pixels are and
nothing about where they are. Cell shape cannot survive that: scrambling the pixels of
an image leaves all of these numbers unchanged. A simple classifier is then trained on
those numbers, using the same classes and the same train/validation split as
supconloss_test_old.py, so its validation accuracy can be set directly beside the CNN's.

Three versions, from least to most information:
    mean            1 number:  the average pixel value
    mean + spread   2 numbers: average and standard deviation
    histogram       40 numbers: the share of pixels in each of 32 brightness bands,
                    plus 8 percentiles. This is everything brightness can tell you.

Usage (on Hyak, from the saffron folder; no GPU needed, a few minutes):
    python scripts/shortcut_baseline.py
Writes shortcut_baseline.txt (the same text that is printed).

How to read it (use the "balanced" column: the average of the three per-class recalls):
    near 0.33                    brightness carries no class information (pure guessing)
    well above 0.33              that much of the task needs no cell shape at all
    close to the CNN's result    the CNN may be doing little more than this
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

# Add src to path so we can import saffron (script lives in scripts/, package in src/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from saffron.data.datasets import MicrogliaDataset, split_indices_by_image

DEFAULT_DATA_DIR = "/gscratch/cheme/agolan/data/preprocessed_data"
LABELS = ["mice", "rat", "gyrified"]
MERGE_MAP = {"human": "gyrified", "ferret": "gyrified", "pig": "gyrified"}
BINS = 32
PERCENTILES = [1, 5, 25, 50, 75, 95, 99, 99.9]


def brightness_numbers(path):
    """All the brightness-only numbers for one image: [mean, std, 32 histogram shares, 8 percentiles]."""
    image = np.load(path).astype(np.float64).ravel()
    histogram = np.histogram(image, bins=BINS, range=(0.0, 1.0))[0] / image.size
    return np.concatenate([[image.mean(), image.std()], histogram, np.percentile(image, PERCENTILES)])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="folder that holds the species folders")
    parser.add_argument("--out_txt", default="shortcut_baseline.txt", help="results text file to write")
    args = parser.parse_args()

    lines = []

    def out(text=""):
        print(text)
        lines.append(text)

    # Same classes and same split as the training script
    dataset = MicrogliaDataset(args.data_dir, labels=LABELS, merge_map=MERGE_MAP)
    train_indices, val_indices = split_indices_by_image(dataset, val_split=0.2, seed=42)

    print(f"reading {len(dataset.samples)} images ...", flush=True)
    features = np.stack([brightness_numbers(path) for path, _ in dataset.samples])
    labels = np.array([label for _, label in dataset.samples])
    folders = np.array([Path(path).relative_to(dataset.data_dir).parts[0] for path, _ in dataset.samples])

    x_train, y_train = features[train_indices], labels[train_indices]
    x_val, y_val = features[val_indices], labels[val_indices]

    out("=" * 70)
    out("SHORTCUT TEST: classifying from brightness alone (no cell shape)")
    out("=" * 70)
    out(f"train images: {len(train_indices)}   validation images: {len(val_indices)}")
    majority = np.bincount(y_val).max() / len(y_val)
    out(f"always guessing the largest class: accuracy {majority:.3f}")

    versions = [
        ("mean", slice(0, 1)),
        ("mean + spread", slice(0, 2)),
        ("histogram", slice(2, None)),
    ]
    out(f"\n{'version':<16}{'numbers':>8}{'accuracy':>10}{'balanced':>10}   recall per class: "
        + "  ".join(LABELS))
    predictions = {}
    for name, columns in versions:
        print(f"training '{name}' ...", flush=True)
        # class_weight="balanced": each class counts equally, like the balanced sampler in the
        # training script. n_jobs=1 and a random forest (no OpenMP) on purpose: the gradient-boosting
        # classifier used here before hung on Hyak, where PyTorch is loaded in the same process.
        model = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                       n_jobs=1, random_state=0)
        model.fit(x_train[:, columns], y_train)
        predicted = model.predict(x_val[:, columns])
        predictions[name] = predicted
        recall = [np.mean(predicted[y_val == c] == c) for c in range(len(LABELS))]
        out(f"{name:<16}{x_train[:, columns].shape[1]:>8}{np.mean(predicted == y_val):>10.3f}"
            f"{balanced_accuracy_score(y_val, predicted):>10.3f}   "
            + "  ".join(f"{r:>{max(len(label), 5)}.3f}" for r, label in zip(recall, LABELS)))
    out("\n'balanced' is the average of the three recalls; pure guessing gives 0.333.")

    predicted = predictions["histogram"]
    out("\nConfusion matrix, histogram version (rows = true class, columns = predicted class)")
    matrix = confusion_matrix(y_val, predicted, labels=range(len(LABELS)))
    out(f"{'':<10}" + "".join(f"{label:>10}" for label in LABELS))
    for label, row in zip(LABELS, matrix):
        out(f"{label:<10}" + "".join(f"{n:>10}" for n in row))

    out("\nBy species folder, histogram version (share of validation images given the right class)")
    val_folders = folders[val_indices]
    for folder in sorted(set(val_folders)):
        mask = val_folders == folder
        out(f"  {folder:<8}{mask.sum():>6} images   correct {np.mean(predicted[mask] == y_val[mask]):.3f}")

    out("\nAverage brightness per species folder (all images)")
    for folder in sorted(set(folders)):
        means = features[folders == folder, 0]
        out(f"  {folder:<8} mean {means.mean():.3f}   middle half of images {np.percentile(means, 25):.3f}"
            f" to {np.percentile(means, 75):.3f}")

    Path(args.out_txt).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out_txt}")


if __name__ == "__main__":
    main()
