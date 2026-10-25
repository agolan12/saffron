"""
One figure from the results of shortcut_baseline_fluorescent.py (one .txt per image preparation):
    top     a table: for each preparation, validation accuracy and balanced accuracy of the
            three versions (mean, mean + spread, histogram), plus image counts
    bottom  the histogram version's confusion matrix for each preparation. Each cell is
            coloured by the share of that row's (true condition's) images it holds, so the
            diagonal is the recall of each condition: dark diagonal = condition recognised
            from brightness alone. Cells show the count and the share.

Usage (from the saffron folder, after saffron_fluorescent_shortcut_array.slurm has finished):
    python scripts/plot_shortcut_fluorescent.py shortcut_fluorescent
    python scripts/plot_shortcut_fluorescent.py shortcut_fluorescent -o fluorescent_shortcut.png
Writes the .png and a .pdf beside it.
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display on the cluster
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Same order and wording as the array job
PREPS = {
    "raw": "nothing (raw)",
    "normalised": "normalised",
    "denoised": "denoised + normalised",
    "li_raw": "Li, no denoising",
    "li_denoised": "Li, denoised",
    "li_saved": "Li, saved masks",
}
VERSIONS = ["mean", "mean + spread", "histogram"]
# One-hue sequential ramp, light (share 0) to dark (share 1)
SHARE_CMAP = LinearSegmentedColormap.from_list(
    "share", ["#f7fbff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
VERSION_LINE = re.compile(r"^(mean \+ spread|mean|histogram)\s+(\d+)\s+([\d.]+)\s+([\d.]+)")


def parse(path):
    """Everything the figure needs from one results file."""
    lines = Path(path).read_text().splitlines()
    result = {"versions": {}, "no_mask": None}
    for i, line in enumerate(lines):
        if line.startswith("conditions:"):
            result["conditions"] = [c.strip() for c in line.split(":", 1)[1].split(",")]
        elif line.startswith("train images:"):
            result["train"], result["val"] = (int(n) for n in re.findall(r"\d+", line))
        elif "no saved mask:" in line:
            result["no_mask"] = int(line.rsplit(":", 1)[1])
        elif "pure guessing gives" in line:
            result["chance"] = float(re.search(r"gives (\d+\.\d+)", line).group(1))
        elif (match := VERSION_LINE.match(line)):
            result["versions"][match.group(1)] = (float(match.group(3)), float(match.group(4)))
        elif line.startswith("Confusion matrix"):
            n = len(result["conditions"])
            rows = lines[i + 2:i + 2 + n]               # skip the column-name line
            result["matrix"] = np.array([[int(v) for v in row.split()[1:]] for row in rows])
    return result


def draw_table(ax, results):
    ax.axis("off")
    columns = ["preparation"] + [f"{v}\nacc. / balanced" for v in VERSIONS] + ["train / val\nimages"]
    cells = []
    for prep, r in results.items():
        row = [PREPS[prep]]
        row += [f"{r['versions'][v][0]:.3f} / {r['versions'][v][1]:.3f}" for v in VERSIONS]
        counts = f"{r['train']} / {r['val']}"
        if r["no_mask"]:
            counts += f"  ({r['no_mask']} no mask)"
        row.append(counts)
        cells.append(row)
    table = ax.table(cellText=cells, colLabels=columns, loc="center", cellLoc="center",
                     colWidths=[0.22, 0.19, 0.19, 0.19, 0.21])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.7)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeeeee")
            cell.set_height(cell.get_height() * 1.4)
        elif col == 0:
            cell.set_text_props(ha="left")
    chance = next(iter(results.values()))["chance"]
    ax.set_title(f"Validation accuracy from brightness alone (balanced: pure guessing = {chance:.3f})",
                 fontsize=11, loc="left")


def draw_matrix(ax, prep, r):
    matrix = r["matrix"]
    totals = matrix.sum(axis=1, keepdims=True)
    shares = np.divide(matrix, totals, out=np.zeros(matrix.shape), where=totals > 0)
    image = ax.imshow(shares, cmap=SHARE_CMAP, vmin=0, vmax=1)
    conditions = r["conditions"]
    for i in range(len(conditions)):
        for j in range(len(conditions)):
            ax.text(j, i, f"{matrix[i, j]}\n{shares[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if shares[i, j] > 0.55 else "#222222")
    ax.set_xticks(range(len(conditions)), conditions, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(conditions)), conditions, fontsize=8)
    ax.set_xlabel("predicted", fontsize=8)
    ax.set_ylabel("true", fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    balanced = r["versions"]["histogram"][1]
    ax.set_title(f"{PREPS[prep]}\nbalanced {balanced:.3f}", fontsize=9)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", help="folder holding shortcut_fluorescent_<prep>.txt")
    parser.add_argument("-o", "--output", default="fluorescent_shortcut.png", help="figure to write (.png)")
    args = parser.parse_args()

    results = {}
    for prep in PREPS:
        path = Path(args.results_dir) / f"shortcut_fluorescent_{prep}.txt"
        if path.exists():
            results[prep] = parse(path)
        else:
            print(f"missing, left out: {path}")
    if not results:
        raise SystemExit(f"no results found in {args.results_dir}")

    n = len(results)
    n_cols = min(3, n)
    n_rows = -(-n // n_cols)
    figure = plt.figure(figsize=(12, 3.2 + 3.6 * n_rows), layout="constrained")
    top, bottom = figure.subfigures(2, 1, height_ratios=[3.2, 3.6 * n_rows])
    draw_table(top.subplots(), results)

    axes = np.atleast_1d(bottom.subplots(n_rows, n_cols)).ravel()
    for ax, (prep, r) in zip(axes, results.items()):
        image = draw_matrix(ax, prep, r)
    for ax in axes[n:]:
        ax.axis("off")
    bottom.suptitle("Confusion matrices, histogram version (colour = share of the true condition's "
                    "validation images; diagonal = recall)", fontsize=11, x=0.01, ha="left")
    bar = bottom.colorbar(image, ax=axes.tolist(), shrink=0.6, format=lambda x, _: f"{x:.0%}")
    bar.ax.tick_params(labelsize=8)
    bar.outline.set_visible(False)

    output = Path(args.output)
    figure.savefig(output, dpi=150)
    figure.savefig(output.with_suffix(".pdf"))
    print(f"wrote {output} and {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
