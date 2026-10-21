"""
List every file in a data set in one CSV, with the columns needed to find duplicates.

One row per file, including the files inside .zip archives (listed, not unpacked), so a zip
that only repeats a folder shows up as duplicates of that folder's files.

Columns:
    path            where the file is, relative to the data folder (zip members: archive.zip!inner/path)
    in_zip          the archive the file is inside, or empty
    condition       first folder of the path (24R_OGD, HC, ...)
    subfolder       the folders between the condition and the file
    file            file name
    base_name       file name without its extensions: x.tif.npy and x.tif both give x
    extension       last extension (.npy, .tif, ...)
    size_bytes, size_mb
    shape, dtype    for .npy and .tif files on disk (read from the file header, the pixels are not loaded)
    n_same_name     how many rows share this base_name (the same image saved as .tif and .npy, or copied)
    content_group   rows with the same number here have identical contents (empty = no copy found)
    n_same_content  how many rows are in that group

Usage (from the saffron folder):
    python scripts/file_inventory.py /path/to/data
    python scripts/file_inventory.py /path/to/data --out_csv all_micro_data_files.csv

Nothing in the data folder is modified.

How identical contents are found: first by size (cheap, from the file system), then only files
whose size matches another file are read to compute a CRC32 checksum. Zip archives already store
a CRC32 for every member, so zip members are compared without unpacking them. Two files count as
identical when both size and checksum match.
"""

import argparse
import csv
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import tifffile
except ImportError:
    tifffile = None

KNOWN_EXTENSIONS = {".npy", ".npz", ".tif", ".tiff", ".png", ".jpg", ".csv", ".tsv", ".txt",
                    ".pickle", ".pkl", ".json", ".zip", ".h5", ".hdf5"}
COLUMNS = ["path", "in_zip", "condition", "subfolder", "file", "base_name", "extension", "size_bytes",
           "size_mb", "shape", "dtype", "n_same_name", "content_group", "n_same_content"]


def base_name(name):
    """'x_li_thresh.tif.npy' -> 'x_li_thresh': strip every known extension, keep dots inside the name."""
    path = Path(name)
    while path.suffix.lower() in KNOWN_EXTENSIONS:
        path = Path(path.stem)
    return path.name


def is_hidden(parts):
    """.DS_Store, ._name copies and __MACOSX folders that macOS adds to folders and zips."""
    return any(part.startswith(".") or part == "__MACOSX" for part in parts)


def make_row(relative_parts, size, in_zip=""):
    row = dict.fromkeys(COLUMNS, "")
    name = relative_parts[-1]
    row["in_zip"] = in_zip
    row["condition"] = relative_parts[0] if len(relative_parts) > 1 else "(top level)"
    row["subfolder"] = "/".join(relative_parts[1:-1])
    row["file"] = name
    row["base_name"] = base_name(name)
    row["extension"] = Path(name).suffix.lower()
    row["size_bytes"] = size
    row["size_mb"] = round(size / 1024 ** 2, 3)
    return row


def read_shape(path):
    """Shape and dtype from the file header only, or ("", "") if the type is not handled or unreadable."""
    try:
        if path.suffix.lower() == ".npy":
            array = np.load(path, mmap_mode="r")
            return str(tuple(array.shape)), str(array.dtype)
        if path.suffix.lower() in (".tif", ".tiff") and tifffile is not None:
            with tifffile.TiffFile(path) as tif:
                return str(tuple(tif.series[0].shape)), str(tif.series[0].dtype)
    except Exception:
        pass
    return "", ""


def crc32(path):
    checksum = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            checksum = zlib.crc32(chunk, checksum)
    return checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", help="folder to list")
    parser.add_argument("--out_csv", default="file_inventory.csv", help="CSV to write")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"not a folder: {data_dir}")

    rows, disk_paths, checksums = [], [], []
    for path in sorted(data_dir.rglob("*")):
        parts = path.relative_to(data_dir).parts
        if not path.is_file() or is_hidden(parts):
            continue
        row = make_row(parts, path.stat().st_size)
        row["path"] = "/".join(parts)
        row["shape"], row["dtype"] = read_shape(path)
        rows.append(row)
        disk_paths.append(path)
        checksums.append(None)                         # filled in below, only where needed

        if path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path) as archive:
                    members = [m for m in archive.infolist() if not m.is_dir()]
            except zipfile.BadZipFile:
                print(f"could not open {row['path']}")
                continue
            for member in members:
                inner = Path(member.filename).parts
                if is_hidden(inner):
                    continue
                member_row = make_row(inner, member.file_size, in_zip=row["path"])
                member_row["path"] = f"{row['path']}!{member.filename}"
                rows.append(member_row)
                disk_paths.append(None)
                checksums.append(member.CRC)

    if not rows:
        raise SystemExit("no files found")

    # Only files whose size matches another row can have a copy, so only those are read.
    size_counts = Counter(r["size_bytes"] for r in rows)
    to_read = [i for i, r in enumerate(rows) if checksums[i] is None and size_counts[r["size_bytes"]] > 1]
    print(f"{len(rows)} files listed; reading {len(to_read)} that share a size with another file ...")
    for done, i in enumerate(to_read, 1):
        checksums[i] = crc32(disk_paths[i])
        if done % 200 == 0:
            print(f"  {done}/{len(to_read)}")

    groups = defaultdict(list)
    for i, r in enumerate(rows):
        if checksums[i] is not None and size_counts[r["size_bytes"]] > 1:
            groups[(r["size_bytes"], checksums[i])].append(i)
    copies = [members for members in groups.values() if len(members) > 1]
    for number, members in enumerate(sorted(copies, key=lambda m: rows[m[0]]["path"]), 1):
        for i in members:
            rows[i]["content_group"] = number
            rows[i]["n_same_content"] = len(members)

    name_counts = Counter(r["base_name"] for r in rows)
    for r in rows:
        r["n_same_name"] = name_counts[r["base_name"]]

    with open(args.out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # ---- summary ---------------------------------------------------------------
    on_disk = [r for r in rows if not r["in_zip"]]
    print(f"\nwrote {args.out_csv}: {len(rows)} rows "
          f"({len(on_disk)} files on disk, {len(rows) - len(on_disk)} inside zips)")

    print(f"\n  files on disk per condition and type (number, MB):")
    table = defaultdict(lambda: [0, 0.0])
    for r in on_disk:
        table[(r["condition"], r["extension"])][0] += 1
        table[(r["condition"], r["extension"])][1] += r["size_bytes"] / 1024 ** 2
    for (condition, extension), (n, mb) in sorted(table.items()):
        print(f"    {condition:<16}{extension:<9}{n:>6}{mb:>11.1f}")

    disk_copies = [[i for i in m if not rows[i]["in_zip"]] for m in copies]
    disk_copies = [m for m in disk_copies if len(m) > 1]
    extra_mb = sum(rows[m[0]]["size_bytes"] * (len(m) - 1) for m in disk_copies) / 1024 ** 2
    print(f"\n  identical contents: {len(copies)} groups in total; {len(disk_copies)} groups among files "
          f"on disk, whose extra copies take {extra_mb:.1f} MB")
    zipped = [i for i, r in enumerate(rows) if r["in_zip"]]
    if zipped:
        matched = sum(1 for i in zipped if rows[i]["content_group"] != ""
                      and any(not rows[j]["in_zip"] for j in groups[(rows[i]["size_bytes"], checksums[i])]))
        print(f"  zip members with an identical file on disk: {matched} of {len(zipped)}")
    same_name = sum(1 for r in on_disk if r["n_same_name"] > 1)
    print(f"  files on disk whose base_name appears more than once: {same_name}  (see n_same_name)")


if __name__ == "__main__":
    main()
