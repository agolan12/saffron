"""
Pool the validation predictions of cross-validation folds into one result.

With --split_by sample, every fold of supconloss_test_old.py validates on different samples, and
over all folds every sample is validated once. Each run writes its validation predictions at
the end of each stage to
    predictions/job<id>_task<n>_stage<1|2>.csv      (one row per validation piece)
This script reads any number of those files, groups them by data folder, class set and stage
(so one call can take every task of an array job), and for each group prints:
    accuracy and balanced accuracy over all folds together, the confusion matrix,
    each fold's balanced accuracy, and the share of each sample's pieces predicted correctly.

Balanced accuracy is the mean of the per-class recalls: 1 / number of classes for guessing,
whatever the class sizes, so it can be compared with the "balanced" column of
shortcut_baseline_fluorescent.py.

Usage (from the folder that holds predictions/):
    python scripts/pool_folds.py predictions/job12345_task*_stage2.csv
    python scripts/pool_folds.py predictions/job12345_*.csv -o pooled_job12345.txt
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

COLUMNS = ["file", "sample", "folder", "true", "predicted", "fold", "stage", "classes", "data_dir", "run"]


def write_predictions(path, rows):
    """Write prediction rows (dicts with the keys in COLUMNS) to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in COLUMNS})


def read_predictions(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def balanced_accuracy(true, predicted):
    """Mean recall over the classes that occur in `true` (sklearn's balanced_accuracy_score)."""
    true, predicted = np.asarray(true), np.asarray(predicted)
    if len(true) == 0:
        return float("nan")
    return float(np.mean([np.mean(predicted[true == c] == c) for c in np.unique(true)]))


def summarize(rows, class_names=None):
    """Lines of text: overall scores, recall per class, confusion matrix, per-fold and per-sample scores."""
    if not rows:
        return ["no predictions"]
    true = np.array([r["true"] for r in rows])
    predicted = np.array([r["predicted"] for r in rows])
    if class_names is None:
        class_names = sorted(set(true) | set(predicted))
    lines = []
    samples = sorted({r["sample"] for r in rows})
    largest = max(np.mean(true == c) for c in class_names)
    lines.append(f"pieces: {len(rows)}   samples: {len(samples)}   folds: {len({r['fold'] for r in rows})}")
    lines.append(f"accuracy {np.mean(true == predicted):.3f}   (always guessing the largest class: {largest:.3f})")
    lines.append(f"balanced accuracy {balanced_accuracy(true, predicted):.3f}   (guessing: {1 / len(class_names):.3f})")
    lines.append("recall per class: " + "   ".join(
        f"{c} {np.mean(predicted[true == c] == c):.3f}" if np.any(true == c) else f"{c} -" for c in class_names))

    width = max(12, max(len(c) for c in class_names) + 2)
    lines.append("\nconfusion matrix (rows = true class, columns = predicted class, counts of pieces)")
    lines.append(f"{'':<{width}}" + "".join(f"{c:>{width}}" for c in class_names))
    for c in class_names:
        lines.append(f"{c:<{width}}" + "".join(f"{np.sum((true == c) & (predicted == p)):>{width}}"
                                               for p in class_names))

    folds = sorted({str(r["fold"]) for r in rows}, key=lambda f: (len(f), f))
    if len(folds) > 1:
        lines.append("\nper fold:")
        for fold in folds:
            in_fold = np.array([str(r["fold"]) == fold for r in rows])
            fold_samples = sorted({r["sample"] for r, keep in zip(rows, in_fold) if keep})
            lines.append(f"  fold {fold}: balanced accuracy {balanced_accuracy(true[in_fold], predicted[in_fold]):.3f}"
                         f"   accuracy {np.mean(true[in_fold] == predicted[in_fold]):.3f}"
                         f"   samples {', '.join(fold_samples)}")

    lines.append("\nper sample (share of its pieces predicted correctly):")
    by_sample = defaultdict(list)
    for r in rows:
        by_sample[r["sample"]].append(r)
    for sample in sorted(by_sample, key=lambda s: (by_sample[s][0]["true"], s)):
        group = by_sample[sample]
        correct = np.mean([r["true"] == r["predicted"] for r in group])
        lines.append(f"  {sample:<12} {group[0]['folder']:<14} true {group[0]['true']:<12} pieces {len(group):>4}"
                     f"   correct {correct:.2f}" + ("   <- mostly wrong" if correct < 0.5 else ""))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="prediction CSV files written by supconloss_test_old.py")
    parser.add_argument("-o", "--output", help="also write the text to this file")
    args = parser.parse_args()

    groups = defaultdict(list)
    for path in args.files:
        for row in read_predictions(path):
            groups[(row["data_dir"], row["classes"], row["stage"])].append(row)

    lines = []
    for (data_dir, classes, stage), rows in sorted(groups.items()):
        lines.append("=" * 78)
        lines.append(f"{Path(data_dir).name}   classes {classes}   stage {stage}   runs: "
                     + ", ".join(sorted({r['run'] for r in rows})))
        lines.append("=" * 78)
        seen = defaultdict(set)
        for r in rows:
            seen[r["file"]].add(r["run"])
        repeated = [f for f, runs in seen.items() if len(runs) > 1]
        if repeated:
            lines.append(f"WARNING: {len(repeated)} pieces are in the validation set of more than one run "
                         "(the same fold given twice, or runs with different splits)")
        lines += summarize(rows)
        lines.append("")
    print("\n".join(lines))
    if args.output:
        Path(args.output).write_text("\n".join(lines) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
