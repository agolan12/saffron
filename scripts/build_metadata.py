"""
Inventory the preprocessed data: what files exist, how they are named, and what is in them.

It reads every file NAME under the data folder (and opens a small sample of files to
check array shape and value range), then writes:
    metadata.csv          one row per file, the file name split into columns
    data_inventory.txt    the summary that is also printed to the screen

Usage (on Hyak, from the saffron folder):
    python scripts/build_metadata.py
    python scripts/build_metadata.py --data_dir /some/other/folder --sample 50

Nothing is modified in the data folder. No GPU needed; it takes well under a minute.

How names are read: a name is cut at every "_" into tokens, and each token is matched
against known kinds (date, magnification, age, sex, ...). Tokens that match nothing are
kept in the "other" column and counted in the summary, so naming patterns this script
does not know about still show up instead of being silently dropped.
"""

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Add src to path so we can import saffron (script lives in scripts/, package in src/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from saffron.data.data_processing import extract_animal_id

DEFAULT_DATA_DIR = "/gscratch/cheme/agolan/data/preprocessed_data"
QUADRANTS = {"TL", "TR", "BL", "BR"}
SPECIES = {"mouse", "mice", "rat", "ferret", "pig", "human", "rabbit"}
SEXES = {"male", "female"}
# Only these two species have "treatment" right after the species word in the examples seen so far.
TREATMENT_AFTER_SPECIES = {"mouse", "rat"}

# Brain-region codes seen in the names, each optionally followed by a number (cc1, thal2, ...).
# Spelling variants are mapped onto one code.
REGION = re.compile(r"(wmcor|wm|hipca|hipdg|cblm|cbl|cb|thal|cortex|cor|cc|st|pl)\d*")
REGION_ALIASES = {"cortex": "cor", "cbl": "cblm", "cb": "cblm"}

COLUMNS = ["folder", "subfolder", "file", "species", "treatment", "age", "sex", "animal",
           "animal_by_package", "date", "slide", "slice", "magnification", "region", "stain",
           "quadrant", "other"]


def find_animal(row, tokens):
    """
    The animal a file comes from, or "" if the name does not say.

    Mouse and rat IDs include the age: the same ID (rat m1) is used for one animal at p13
    and a different one at p70. Pig names carry a sample letter after the animal
    (2187_B, 2187_D); both letters are the same animal, so the letter is left out.
    """
    lows = [t.lower() for t in tokens]
    species = row["species"]
    if species == "mouse":
        ids = [t for t in lows if re.fullmatch(r"(fs|ms)\d+", t)]
        return f"mouse_{row['age']}_{ids[0]}" if ids else ""
    if species == "rat":
        ids = [t for t in lows if re.fullmatch(r"[mf]\d+", t)]
        return f"rat_{row['age']}_{ids[0]}" if ids else ""
    if species in ("pig", "ferret") and species in lows[:-1]:
        return f"{species}_{tokens[lows.index(species) + 1]}"
    if species == "rabbit":
        match = re.search(r"rabbit_?(\d+C\d+)", "_".join(tokens))
        return f"rabbit_{match.group(1)}" if match else ""
    if species == "human":
        return tokens[0]                               # donor, e.g. UWA-7606 (the next piece is the tissue block)
    return ""


def parse_name(path, data_dir):
    """Split one file name into the COLUMNS above. Unknown pieces go to 'other'."""
    relative = path.relative_to(data_dir)
    row = dict.fromkeys(COLUMNS, "")
    row["folder"] = relative.parts[0] if len(relative.parts) > 1 else "(top level)"
    row["subfolder"] = "/".join(relative.parts[1:-1])
    row["file"] = path.name

    tokens = path.stem.split("_")
    if tokens[-1] in QUADRANTS:
        row["quadrant"] = tokens.pop()

    # Stain is written as "Alexa_488_antibody" or "Alx568"
    stain = re.search(r"(?:Alexa_|Alx)(\d+)", path.stem)
    if stain:
        row["stain"] = f"Alexa {stain.group(1)}"

    other = []
    for i, token in enumerate(tokens):
        low = token.lower()
        previous = tokens[i - 1].lower() if i > 0 else ""
        if re.fullmatch(r"(19|20)\d{6}", token) and not row["date"]:
            row["date"] = token
        elif low in SPECIES and not row["species"]:
            row["species"] = low
        elif re.fullmatch(r"rabbit\d+c\d+", low) and not row["species"]:
            row["species"] = "rabbit"                  # "rabbit22011C3": species and ID written as one piece
        elif i == 0 and token.startswith("UWA") and not row["species"]:
            row["species"] = "human"                   # human names start with the donor ID, no species word
        elif previous in TREATMENT_AFTER_SPECIES and previous == row["species"] and not row["treatment"]:
            row["treatment"] = token
        elif re.fullmatch(r"p\d+", low) and not row["age"]:
            row["age"] = low
        elif low in SEXES and not row["sex"]:
            row["sex"] = low
        elif REGION.fullmatch(low) and not row["region"]:
            code = REGION.fullmatch(low).group(1)
            row["region"] = REGION_ALIASES.get(code, code)
        elif re.fullmatch(r"slide\d+", low):
            row["slide"] = low
        elif re.fullmatch(r"slice\d+", low):
            row["slice"] = low
        elif re.fullmatch(r"\d+x", low):
            row["magnification"] = low
        else:
            other.append(token)
    row["other"] = "_".join(other)

    # Some folders say the age instead of the file name (pig/p1/..., pig/adult/...)
    first_subfolder = relative.parts[1].lower() if len(relative.parts) > 2 else ""
    if not row["age"] and re.fullmatch(r"p\d+|adult", first_subfolder):
        row["age"] = first_subfolder

    # What the training code's own function calls this animal, to compare with the column above.
    row["animal_by_package"] = extract_animal_id(str(path))
    row["animal"] = find_animal(row, tokens) or row["animal_by_package"]
    if not row["species"]:
        row["species"] = f"({row['folder']})"
    return row


def template(name):
    """'20230326_mouse_fs1_60x_TL' -> '#_mouse_fs#_#x_TL': the shape of a name without its numbers."""
    return re.sub(r"\d+", "#", name)


def table(rows, row_key, col_key, out, title):
    """Print a count table: one line per value of row_key, one column per value of col_key."""
    counts = Counter((r[row_key] or "-", r[col_key] or "-") for r in rows)
    row_values = sorted({k[0] for k in counts})
    col_values = sorted({k[1] for k in counts})
    out(f"\n{title}")
    if len(col_values) > 12:
        out(f"  ({len(col_values)} different values of {col_key}; showing the 12 most common)")
        totals = Counter()
        for (_, c), n in counts.items():
            totals[c] += n
        col_values = [c for c, _ in totals.most_common(12)]
    width = max(10, max(len(c) for c in col_values) + 2)
    first = max(len(row_key), max(len(v) for v in row_values)) + 2
    out("  " + f"{row_key:<{first}}" + "".join(f"{c:>{width}}" for c in col_values) + f"{'total':>{width}}")
    for rv in row_values:
        total = sum(n for (r, _), n in counts.items() if r == rv)
        out("  " + f"{rv:<{first}}" + "".join(f"{counts.get((rv, c), 0):>{width}}" for c in col_values)
            + f"{total:>{width}}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="folder that holds the species folders")
    parser.add_argument("--out_csv", default="metadata.csv", help="per-file table to write")
    parser.add_argument("--out_txt", default="data_inventory.txt", help="summary text file to write")
    parser.add_argument("--sample", default=20, type=int,
                        help="how many .npy files per folder to open for shape and value checks (0 = none)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"not a folder: {data_dir}")

    lines = []

    def out(text=""):
        print(text)
        lines.append(text)

    all_files = sorted(p for p in data_dir.rglob("*") if p.is_file())
    npy_files = [p for p in all_files if p.suffix == ".npy"]
    rows = [parse_name(p, data_dir) for p in npy_files]
    folders = sorted({r["folder"] for r in rows})

    out("=" * 70)
    out(f"DATA INVENTORY: {data_dir}")
    out("=" * 70)

    # ---- 1. what is on disk --------------------------------------------------
    out("\n1. FILES ON DISK")
    by_extension = Counter(p.suffix or "(none)" for p in all_files)
    out("  by extension: " + ", ".join(f"{ext} {n}" for ext, n in by_extension.most_common()))
    out(f"  .npy files: {len(npy_files)}")
    top_level = Counter((p.relative_to(data_dir).parts[0] if len(p.relative_to(data_dir).parts) > 1
                         else "(top level)") for p in all_files)
    for folder, n in sorted(top_level.items()):
        subfolders = {r["subfolder"] for r in rows if r["folder"] == folder and r["subfolder"]}
        note = f"   ({len(subfolders)} subfolders: {', '.join(sorted(subfolders)[:5])}" \
               f"{', ...' if len(subfolders) > 5 else ''})" if subfolders else ""
        out(f"  {folder:<14}{n:>7} files{note}")
    if not rows:
        raise SystemExit("no .npy files found")

    # ---- 2. how names are built ----------------------------------------------
    out("\n2. NAME PATTERNS PER FOLDER   (# stands for any number)")
    for folder in folders:
        # the quadrant ending is dropped here, otherwise every pattern would be listed four times
        templates = Counter(template(Path(r["file"]).stem[:-3] if r["quadrant"] else Path(r["file"]).stem)
                            for r in rows if r["folder"] == folder)
        out(f"\n  {folder}: {len(templates)} different patterns")
        for pattern, n in templates.most_common(12):
            out(f"    {n:>6}  {pattern}")
        if len(templates) > 12:
            out(f"    ... and {len(templates) - 12} more")

    out("\n3. PIECES OF THE NAMES THIS SCRIPT DID NOT RECOGNISE   (# stands for any number)")
    out("   Look here for new brain-region codes, channels, processing steps, ...")
    for folder in folders:
        unknown = Counter()
        for r in rows:
            if r["folder"] == folder:
                unknown.update({template(t) for t in r["other"].split("_") if t})
        n_folder = sum(r["folder"] == folder for r in rows)
        out(f"\n  {folder} ({n_folder} files): piece, number of files that contain it")
        for piece, n in unknown.most_common(40):
            out(f"    {n:>6}  {piece}")
        if len(unknown) > 40:
            out(f"    ... and {len(unknown) - 40} more")

    # ---- 4. counts -------------------------------------------------------------
    out("\n4. COUNTS OF FILES")
    for column in ["species", "magnification", "stain", "region", "treatment", "age", "sex", "quadrant"]:
        table(rows, "folder", column, out, f"folder x {column}")
    table(rows, "treatment", "magnification", out, "treatment x magnification")
    table(rows, "folder", "date", out, "folder x imaging date")

    # ---- 5. images and animals -------------------------------------------------
    out("\n5. ORIGINAL IMAGES AND ANIMALS")
    out("   An original image = the files that share a name apart from the _TL/_TR/_BL/_BR ending.")
    out(f"\n  {'folder':<14}{'files':>8}{'images':>8}{'animals':>9}   files per image")
    for folder in folders:
        folder_rows = [r for r in rows if r["folder"] == folder]
        images = Counter()
        for r in folder_rows:
            stem = Path(r["file"]).stem
            images[(r["subfolder"], stem[:-3] if r["quadrant"] else stem)] += 1
        per_image = Counter(images.values())
        animals = {r["animal"] for r in folder_rows}
        spread = ", ".join(f"{k} files: {v} images" for k, v in sorted(per_image.items()))
        out(f"  {folder:<14}{len(folder_rows):>8}{len(images):>8}{len(animals):>9}   {spread}")

    out("\n  Animals per folder and treatment (animals, not files):")
    groups = defaultdict(set)
    for r in rows:
        groups[(r["folder"], r["treatment"] or "-", r["age"] or "-")].add(r["animal"])
    out(f"  {'folder':<14}{'treatment':<16}{'age':<8}{'animals':>8}   animal IDs")
    for (folder, treatment, age), animals in sorted(groups.items()):
        shown = ", ".join(sorted(animals)[:10]) + (", ..." if len(animals) > 10 else "")
        out(f"  {folder:<14}{treatment:<16}{age:<8}{len(animals):>8}   {shown}")

    out("\n  Files per animal:")
    for folder in folders:
        per_animal = Counter(r["animal"] for r in rows if r["folder"] == folder)
        counts = sorted(per_animal.values())
        out(f"  {folder:<14}{len(per_animal):>4} animals; files per animal: min {counts[0]}, "
            f"median {counts[len(counts) // 2]}, max {counts[-1]}")

    # An ID like "fs1" could be reused for different animals in different cohorts.
    seen_with = defaultdict(set)
    for r in rows:
        seen_with[(r["folder"], r["animal"])].add((r["treatment"], r["age"], r["sex"]))
    reused = {k: v for k, v in seen_with.items() if len(v) > 1}
    out(f"\n  Animal IDs that appear with more than one treatment/age/sex: {len(reused)}")
    for (folder, animal), combos in sorted(reused.items())[:20]:
        out(f"    {folder} {animal}: " + "; ".join("/".join(x or "-" for x in c) for c in sorted(combos)))
    if reused:
        out("    -> these are probably different animals sharing an ID; the ID alone is not enough.")

    # Does the training package group files into the same animals as the name columns do?
    package_animals = {(r["folder"], r["animal_by_package"]) for r in rows}
    out(f"\n  Animals found from the names here: {len(seen_with)}.   "
        f"By saffron's extract_animal_id: {len(package_animals)}.")
    fallback = Counter((r["folder"], r["animal_by_package"]) for r in rows
                       if r["date"] and r["animal_by_package"].startswith(r["date"]))
    if fallback:
        out(f"  extract_animal_id fell back to a date-based ID for {sum(fallback.values())} files "
            f"({len(fallback)} IDs), for example:")
        for (folder, animal), n in fallback.most_common(5):
            out(f"    {folder}: {animal}  ({n} files)")
        out("    -> those files are grouped by imaging date, not by animal.")

    # ---- 6. what is inside the files -------------------------------------------
    if args.sample > 0:
        out(f"\n6. INSIDE THE FILES   (up to {args.sample} files per folder, picked at random with a fixed seed)")
        rng = np.random.default_rng(0)
        for folder in folders:
            paths = [p for p, r in zip(npy_files, rows) if r["folder"] == folder]
            picked = [paths[i] for i in rng.choice(len(paths), size=min(args.sample, len(paths)), replace=False)]
            shapes, dtypes, lows, highs, means, failed = Counter(), Counter(), [], [], [], 0
            for p in picked:
                try:
                    array = np.load(p, mmap_mode="r")
                except Exception as error:
                    failed += 1
                    out(f"    could not open {p.name}: {error}")
                    continue
                shapes[str(tuple(array.shape))] += 1
                dtypes[str(array.dtype)] += 1
                lows.append(float(array.min()))
                highs.append(float(array.max()))
                means.append(float(array.mean()))
            out(f"\n  {folder}: opened {len(picked) - failed} of {len(paths)} files")
            out("    shape (files): " + ", ".join(f"{s} ({n})" for s, n in shapes.most_common()))
            out("    dtype (files): " + ", ".join(f"{d} ({n})" for d, n in dtypes.most_common()))
            if lows:
                out(f"    values: lowest {min(lows):g}, highest {max(highs):g}, "
                    f"mean of file means {np.mean(means):.4g} (file means range {min(means):.4g} to {max(means):.4g})")

    # ---- write the outputs -----------------------------------------------------
    with open(args.out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    Path(args.out_txt).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out_csv} ({len(rows)} rows) and {args.out_txt}")


if __name__ == "__main__":
    main()
