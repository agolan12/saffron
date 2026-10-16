"""
Compare training runs by reading their SLURM error logs (the tqdm progress bars).

For each run it plots, per epoch:
    - training loss (mean of the batch losses shown on the Batch bar)
    - validation SupCon loss   (Loss= on the Epoch bar)
    - validation accuracy      (Acc=  on the Epoch bar)
and prints a summary table, which is also saved next to the image (same name, .txt).

Usage (from the folder that holds logs_error/):
    python plot_runs.py logs_error/supconloss_12345678_*.err
    python plot_runs.py logs_error/a.err logs_error/b.err -o my_comparison.png

Each run is labelled with its settings (learning rate, ce_weight, and the stage 2 /
learning-rate-drop options when they are switched on), read from the matching
logs_output/*.out file when it exists, otherwise with the file name.
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display on the cluster
import matplotlib.pyplot as plt
import numpy as np

# Fixed colour per run, in order. Up to 8 runs; more than that is unreadable anyway.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MAJORITY_BASELINE = 0.41  # accuracy of always guessing mice (2,916 of 7,076 files)
MAJORITY_BASELINE_RODENT = 0.53  # the same with mice and rat only (2,916 of 5,544 files)

ANSI = re.compile(r"\x1b\[[0-9;]*m")
# One regex for both bars, so matches come back in the order they were printed.
#   Batch:  44%|████▍     | 20/45 [00:39<00:51,  2.05s/it, Loss=3.59]
#   Epoch:   1%|          | 1/100 [02:58<4:53:59, 178.18s/it, Loss=1.22, Acc=0.648]
BAR = re.compile(
    r"(Batch|Epoch):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+) \[[^\]]*?Loss=([-+.\dnaife]+)(?:, Acc=([-+.\dnaife]+))?\s*\]"
)


def parse_log(path):
    """
    Returns a list of stages; each stage is a dict of per-epoch lists:
    {"train_loss": [...], "val_loss": [...], "val_acc": [...]}.
    Only epochs that finished (have a validation result) are kept.
    """
    text = ANSI.sub("", Path(path).read_text(errors="ignore"))

    stages = []
    stage = None
    last_epoch_count = None   # last value of n in "Epoch: n/total"
    last_batch_count = None   # last value of n in "Batch: n/total"
    batch_losses = []         # batch losses of the epoch being trained
    pending_train_loss = None

    for kind, n, _total, loss, acc in BAR.findall(text):
        n = int(n)
        if kind == "Batch":
            if last_batch_count is not None and n < last_batch_count:
                batch_losses = []  # counter went back to the start: a new epoch began
            # tqdm prints each value twice (once before and once after the counter
            # advances); the first time a count appears is the one that belongs to it.
            if n >= 1 and n != last_batch_count:
                batch_losses.append(float(loss))
            last_batch_count = n
            pending_train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        else:
            if acc == "":
                continue
            if stage is None or (last_epoch_count is not None and n < last_epoch_count):
                stage = {"train_loss": [], "val_loss": [], "val_acc": []}  # epoch counter restarted: next stage
                stages.append(stage)
                last_epoch_count = None
            if n >= 1 and n != last_epoch_count:
                stage["train_loss"].append(pending_train_loss if pending_train_loss is not None else float("nan"))
                stage["val_loss"].append(float(loss))
                stage["val_acc"].append(float(acc))
            last_epoch_count = n

    return [s for s in stages if s["val_acc"]]


def run_label(err_path):
    """'lr 0.0003, ce 0.5, head only' from the matching .out file, else the file name."""
    err_path = Path(err_path)
    candidates = [
        Path(str(err_path).replace("logs_error", "logs_output")).with_suffix(".out"),
        err_path.with_suffix(".out"),
    ]
    for out_path in candidates:
        if out_path.exists():
            text = out_path.read_text(errors="ignore")
            parts = []
            for name, short in [("learning_rate", "lr"), ("ce_weight", "ce")]:
                match = re.search(rf"{name}\s*=\s*(\S+)", text)
                if match:
                    parts.append(f"{short} {float(match.group(1)):g}")
            # options that only appear in the label when they differ from the original recipe
            if re.search(r"classes\s*=\s*rodent", text):
                parts.append("mice vs rat")
            if re.search(r"classes\s*=\s*condition", text):
                parts.append("HC/OGD/ROT")
            if re.search(r"binarize\s*=\s*True", text):
                parts.append("binarized")
            if re.search(r"equalize\s*=\s*True", text):
                parts.append("equalized")
            if re.search(r"li_threshold\s*=\s*True", text):
                parts.append("Li mask")
            if re.search(r"stage2_head_only\s*=\s*True", text):
                parts.append("head only")
            match = re.search(r"lr_drop_epoch\s*=\s*(\d+)", text)
            if match and int(match.group(1)) > 0:
                parts.append(f"lr drop {match.group(1)}")
            match = re.search(r"stage2_lr\s*=\s*([-+.\de]+)", text)
            if match:
                parts.append(f"s2 lr {float(match.group(1)):g}")
            if parts:
                return ", ".join(parts)
    return err_path.stem


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", help="error log files (.err), one per run")
    parser.add_argument("-o", "--output", default="runs_comparison.png", help="image file to write")
    args = parser.parse_args()

    runs = []
    for path in args.logs:
        stages = parse_log(path)
        if not stages:
            print(f"skipping {path}: no finished epochs found")
            continue
        runs.append({"label": run_label(path), "path": path, "stages": stages})
    if not runs:
        raise SystemExit("nothing to plot")
    if len(runs) > len(COLORS):
        raise SystemExit(f"{len(runs)} runs given; plot at most {len(COLORS)} at a time")

    panels = [
        ("train_loss", "Training loss (mean per epoch)", "loss"),
        ("val_loss", "Validation SupCon loss", "loss"),
        ("val_acc", "Validation accuracy", "accuracy"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

    for run, color in zip(runs, COLORS):
        for ax, (key, _, _) in zip(axes, panels):
            values = [v for stage in run["stages"] for v in stage[key]]
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, color=color, lw=2, label=run["label"])

    # where stage 1 ends, for runs that reached stage 2
    boundaries = sorted({len(run["stages"][0]["val_acc"]) for run in runs if len(run["stages"]) > 1})
    for ax, (key, title, ylabel) in zip(axes, panels):
        for b in boundaries:
            ax.axvline(b + 0.5, color="#8a8983", lw=1, ls="--")
            ax.annotate("stage 2 →", (b + 0.5, 1), xycoords=("data", "axes fraction"), xytext=(4, -12),
                        textcoords="offset points", fontsize=8, color="#52514e")
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.03)

    # a line for each class set that appears among the runs
    baselines = sorted({MAJORITY_BASELINE_RODENT if "mice vs rat" in run["label"] else MAJORITY_BASELINE for run in runs})
    for baseline in baselines:
        axes[2].axhline(baseline, color="#8a8983", lw=1, ls=":")
        axes[2].annotate(f"always guess mice ({baseline})", (1, baseline), xycoords=("axes fraction", "data"),
                         xytext=(-4, -11), textcoords="offset points", ha="right", fontsize=8, color="#52514e")
    axes[2].set_ylim(0, 1)
    # the bottom of the accuracy panel is the one area the lines rarely reach
    axes[2].legend(loc="lower right", frameon=False, fontsize=9, title="run")
    if boundaries:
        fig.suptitle("Training loss changes definition at stage 2 (cross-entropy only), so its drop there is not an improvement",
                     fontsize=9, color="#52514e", x=0.01, ha="left")

    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}\n")

    # summary table, printed and saved next to the image
    width = max(len(run["label"]) for run in runs) + 2
    table = [f"{'run':<{width}}{'stage':>6}{'epochs':>8}{'best acc':>10}{'at epoch':>10}{'last acc':>10}{'last val loss':>15}{'last train loss':>17}"]
    for run in runs:
        for i, stage in enumerate(run["stages"], start=1):
            acc = stage["val_acc"]
            best = int(np.argmax(acc))
            table.append(f"{run['label']:<{width}}{i:>6}{len(acc):>8}{acc[best]:>10.3f}{best + 1:>10}{acc[-1]:>10.3f}"
                         f"{stage['val_loss'][-1]:>15.3g}{stage['train_loss'][-1]:>17.3g}")
    print("\n".join(table))
    table_path = Path(args.output).with_suffix(".txt")
    table_path.write_text("\n".join(table) + "\n")
    print(f"\nwrote {table_path}")


if __name__ == "__main__":
    main()
