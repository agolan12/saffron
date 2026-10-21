"""
First look at a new data set: what files there are, how they are named, and what is inside them.

It works on any folder and does not assume a naming scheme. It prints, and writes to a text file:
    1. folders and file types (count and size per extension, files per subfolder)
    2. name patterns: each name with its numbers replaced by "#", so 400 files named
       20240101_mouse_fs1_60x.tif, 20240102_mouse_fs2_60x.tif, ... show up as one line
    3. the pieces names are built from (split at "_", "-", " ", "."), most common first
    4. what is inside a sample of files per type: shape, dtype and value range for
       images and arrays, columns for tables, keys for JSON/HDF5

Usage (from the saffron folder):
    python scripts/explore_dataset.py /path/to/new_data
    python scripts/explore_dataset.py /path/to/new_data --sample 20 --out_txt new_data_overview.txt

Nothing in the data folder is modified. Reading TIFFs works best with tifffile installed
(pip install tifffile); without it Pillow is used, which reads fewer TIFF layouts.
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Optional readers: the script still runs without them and says which files it could not open.
try:
    import tifffile
except ImportError:
    tifffile = None
try:
    from PIL import Image
except ImportError:
    Image = None
try:
    import h5py
except ImportError:
    h5py = None

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
TIFF_TYPES = {".tif", ".tiff"}
TABLE_TYPES = {".csv", ".tsv", ".txt"}
HDF5_TYPES = {".h5", ".hdf5"}


def human_size(n_bytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n_bytes < 1024 or unit == "TB":
            return f"{n_bytes:.1f} {unit}" if unit != "B" else f"{n_bytes} B"
        n_bytes /= 1024


def template(name):
    """'20230326_mouse_fs1_60x_TL' -> '#_mouse_fs#_#x_TL': the shape of a name without its numbers."""
    return re.sub(r"\d+", "#", name)


def describe_array(array):
    """Shape, dtype and value range of one array, as a dict."""
    info = {"shape": str(tuple(array.shape)), "dtype": str(array.dtype)}
    if array.size and np.issubdtype(array.dtype, np.number):
        info["min"] = float(array.min())
        info["max"] = float(array.max())
        info["mean"] = float(array.mean())
    return info


def read_tiff(path):
    if tifffile is not None:
        with tifffile.TiffFile(path) as tif:
            array = tif.asarray()
            info = describe_array(array)
            info["axes"] = tif.series[0].axes          # e.g. "ZCYX": which dimension is which
            info["pages"] = len(tif.pages)
            return info
    if Image is None:
        raise RuntimeError("install tifffile or Pillow to read TIFFs")
    with Image.open(path) as image:
        info = describe_array(np.asarray(image))
        info["mode"] = image.mode
        info["pages"] = getattr(image, "n_frames", 1)  # only the first page is described above
        return info


def read_image(path):
    if Image is None:
        raise RuntimeError("install Pillow to read images")
    with Image.open(path) as image:
        info = describe_array(np.asarray(image))
        info["mode"] = image.mode                      # "L" = grey, "RGB", "I;16" = 16-bit grey, ...
        return info


def read_table(path):
    with open(path, newline="", errors="replace") as handle:
        first_line = handle.readline()
        n_lines = 1 + sum(1 for _ in handle) if first_line else 0
    delimiter = "\t" if path.suffix == ".tsv" or "\t" in first_line else ","
    header = next(csv.reader([first_line], delimiter=delimiter), [])
    return {"lines": n_lines, "columns": len(header), "first line": first_line.strip()[:150]}


def read_hdf5(path):
    if h5py is None:
        raise RuntimeError("install h5py to read HDF5")
    found = {}
    with h5py.File(path, "r") as handle:
        handle.visititems(lambda name, item: found.__setitem__(name, f"{item.shape} {item.dtype}")
                          if isinstance(item, h5py.Dataset) else None)
    return {"datasets": "; ".join(f"{k}: {v}" for k, v in list(found.items())[:10])}


def look_inside(path):
    """A dict describing the contents of one file, or raises if the file cannot be read."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return describe_array(np.load(path, mmap_mode="r"))
    if suffix == ".npz":
        with np.load(path) as archive:
            return {"arrays": "; ".join(f"{k}: {archive[k].shape} {archive[k].dtype}" for k in archive.files)}
    if suffix in TIFF_TYPES:
        return read_tiff(path)
    if suffix in IMAGE_TYPES:
        return read_image(path)
    if suffix in TABLE_TYPES:
        return read_table(path)
    if suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return {"type": "dict", "keys": ", ".join(list(data)[:20])}
        return {"type": type(data).__name__, "length": len(data) if hasattr(data, "__len__") else ""}
    if suffix in HDF5_TYPES:
        return read_hdf5(path)
    return None                                        # a type this script does not know how to open


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", help="folder to look at")
    parser.add_argument("--sample", default=10, type=int,
                        help="how many files per file type to open (0 = only look at names)")
    parser.add_argument("--out_txt", default="dataset_overview.txt", help="summary text file to write")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"not a folder: {data_dir}")

    lines = []

    def out(text=""):
        print(text)
        lines.append(text)

    # Hidden files (.DS_Store, ._name from macOS copies) are skipped.
    files = sorted(p for p in data_dir.rglob("*")
                   if p.is_file() and not any(part.startswith(".") for part in p.relative_to(data_dir).parts))
    if not files:
        raise SystemExit("no files found")

    out("=" * 70)
    out(f"DATA SET OVERVIEW: {data_dir}")
    out("=" * 70)

    # ---- 1. folders and file types ---------------------------------------------
    out(f"\n1. FILES: {len(files)}, {human_size(sum(p.stat().st_size for p in files))} in total")
    out("\n  by type:")
    by_suffix = defaultdict(list)
    for p in files:
        by_suffix[p.suffix.lower() or "(none)"].append(p)
    for suffix, paths in sorted(by_suffix.items(), key=lambda item: -len(item[1])):
        sizes = [p.stat().st_size for p in paths]
        out(f"    {suffix:<10}{len(paths):>7} files   {human_size(sum(sizes)):>10} total, "
            f"{human_size(min(sizes))} to {human_size(max(sizes))} each")

    out("\n  by folder (files directly inside each folder):")
    by_folder = Counter(str(p.parent.relative_to(data_dir)) for p in files)
    for folder, n in sorted(by_folder.items())[:50]:
        out(f"    {n:>7}  {folder}")
    if len(by_folder) > 50:
        out(f"    ... and {len(by_folder) - 50} more folders")

    # ---- 2. name patterns ------------------------------------------------------
    out("\n2. NAME PATTERNS PER TYPE   (# stands for any number)")
    for suffix, paths in sorted(by_suffix.items(), key=lambda item: -len(item[1])):
        templates = Counter(template(p.stem) for p in paths)
        out(f"\n  {suffix}: {len(templates)} different patterns")
        for pattern, n in templates.most_common(15):
            out(f"    {n:>6}  {pattern}")
        if len(templates) > 15:
            out(f"    ... and {len(templates) - 15} more")

    # ---- 3. pieces of names ----------------------------------------------------
    out("\n3. PIECES THE NAMES ARE BUILT FROM   (piece, number of files that contain it)")
    out("   Pieces in almost every file are fixed labels; pieces in a few files are values")
    out("   (animal, date, region, ...). Pieces with numbers are also shown with their numbers as #.")
    pieces, piece_templates = Counter(), Counter()
    for p in files:
        tokens = {t for t in re.split(r"[_\-\s.]+", p.stem) if t}
        pieces.update(tokens)
        piece_templates.update({template(t) for t in tokens if re.search(r"\d", t)})
    out("\n  exact pieces:")
    for piece, n in pieces.most_common(40):
        out(f"    {n:>6}  {piece}")
    out("\n  pieces with numbers, as patterns:")
    for piece, n in piece_templates.most_common(20):
        examples = sorted({t for t in pieces if template(t) == piece})
        shown = ", ".join(examples[:8]) + (", ..." if len(examples) > 8 else "")
        out(f"    {n:>6}  {piece:<14} {len(examples)} values: {shown}")

    # ---- 4. inside the files ---------------------------------------------------
    if args.sample > 0:
        out(f"\n4. INSIDE THE FILES   (up to {args.sample} files per type, picked at random with a fixed seed)")
        rng = np.random.default_rng(0)
        for suffix, paths in sorted(by_suffix.items(), key=lambda item: -len(item[1])):
            picked = [paths[i] for i in rng.choice(len(paths), size=min(args.sample, len(paths)), replace=False)]
            results, errors = [], []
            for p in picked:
                try:
                    info = look_inside(p)
                except Exception as error:
                    errors.append(f"{p.name}: {error}")
                    continue
                if info is None:
                    break
                results.append(info)
            out(f"\n  {suffix}:")
            if not results and not errors:
                out("    (this script does not know how to open this type)")
                continue
            out(f"    opened {len(results)} of {len(picked)} sampled files")
            # Text fields (shape, dtype, mode, ...) are counted; number fields are given as a range.
            keys = list(dict.fromkeys(k for info in results for k in info))
            for key in keys:
                values = [info[key] for info in results if key in info]
                if all(isinstance(v, (int, float)) for v in values) and key not in ("pages", "columns"):
                    out(f"    {key:<11}{min(values):.4g} to {max(values):.4g}")
                else:
                    counts = Counter(str(v) for v in values)
                    out(f"    {key:<11}" + "; ".join(f"{v} ({n})" for v, n in counts.most_common(5))
                        + (f"; ... {len(counts) - 5} more" if len(counts) > 5 else ""))
            for message in errors[:5]:
                out(f"    could not open {message}")

    Path(args.out_txt).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out_txt}")


if __name__ == "__main__":
    main()
