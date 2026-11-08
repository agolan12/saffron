"""Tests for the review-label filter (--review_csv/--keep) in export_fluorescent.py and
shortcut_baseline_fluorescent.py."""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import export_fluorescent as ef  # noqa: E402
import shortcut_baseline_fluorescent as sbf  # noqa: E402


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "folder", "label", "labelled_at"])
        for file, label in rows:
            writer.writerow([file, file.split("/")[0], label, "2026-11-05T12:00:00"])


@pytest.fixture
def data_dir(tmp_path):
    """nt_control/ and ogd_3/ with 64x64 3-channel images; li_thresh/ must be ignored."""
    rng = np.random.default_rng(0)
    for condition, names in {"nt_control": ["a", "b", "c"], "ogd_3": ["d", "e", "f"]}.items():
        (tmp_path / "data" / condition).mkdir(parents=True)
        for name in names:
            tifffile.imwrite(tmp_path / "data" / condition / f"{name}.tif",
                             rng.integers(0, 4000, (3, 64, 64), dtype=np.uint16))
    (tmp_path / "data" / "li_thresh" / "ogd_3").mkdir(parents=True)
    write_csv(tmp_path / "review.csv", [
        ("nt_control/a.tif", "good"), ("nt_control/b.tif", "bad"), ("nt_control/c.tif", "unsure"),
        ("ogd_3/d.tif", "good"), ("ogd_3/e.tif", "bad"),            # ogd_3/f.tif not reviewed
    ])
    return tmp_path


def images(files):
    """'<condition>/<image>' for each written piece (pieces are <image>_<quadrant>.npy)."""
    return {f"{Path(f).parent}/{Path(f).name.split('_')[0]}" for f in files}


def run_export(monkeypatch, data_dir, out, *extra):
    monkeypatch.setattr(sys, "argv", ["export_fluorescent.py", str(data_dir / "data"), "--prep", "normalised",
                                      "--piece_size", "32", "--out_dir", str(out), *extra])
    ef.main()
    return sorted(p.relative_to(out).as_posix() for p in out.rglob("*.npy"))


def test_review_label_uses_condition_and_file_name(data_dir):
    review = sbf.load_review(data_dir / "review.csv")
    assert sbf.review_label(review, data_dir / "data" / "nt_control" / "b.tif") == "bad"
    assert sbf.review_label(review, data_dir / "data" / "ogd_3" / "f.tif") == "not reviewed"
    # same name in another condition folder is a different image
    assert sbf.review_label(review, data_dir / "data" / "ogd_3" / "a.tif") == "not reviewed"


def test_load_review_finds_header_anywhere(tmp_path):
    # A CSV sorted in a spreadsheet: header row in the middle, Windows line endings
    (tmp_path / "sorted.csv").write_text(
        "nt_control/a.tif,nt_control,bad,2026-11-05T12:00:00\r\n"
        "file,folder,label,labelled_at\r\n"
        "ogd_3/d.tif,ogd_3,good,2026-11-05T12:00:00\r\n")
    assert sbf.load_review(tmp_path / "sorted.csv") == {"nt_control/a.tif": "bad", "ogd_3/d.tif": "good"}


def test_load_review_without_header_stops(tmp_path):
    (tmp_path / "no_header.csv").write_text("nt_control/a.tif,nt_control,bad,2026-11-05T12:00:00\n")
    with pytest.raises(SystemExit, match="no header row"):
        sbf.load_review(tmp_path / "no_header.csv")


def test_export_without_csv_writes_every_image(monkeypatch, data_dir):
    files = run_export(monkeypatch, data_dir, data_dir / "all")
    assert len(files) == 6 * 4                           # 6 images, each cut into 4 pieces of 32 px


def test_export_keeps_only_good_images(monkeypatch, data_dir, capsys):
    files = run_export(monkeypatch, data_dir, data_dir / "good", "--review_csv", str(data_dir / "review.csv"))
    assert images(files) == {"nt_control/a", "ogd_3/d"}
    assert len(files) == 2 * 4
    printed = capsys.readouterr().out
    assert "1 bad, 1 unsure" in printed                  # nt_control
    assert "1 bad, 1 not reviewed" in printed            # ogd_3


def test_export_keep_good_and_unsure(monkeypatch, data_dir):
    files = run_export(monkeypatch, data_dir, data_dir / "gu", "--review_csv", str(data_dir / "review.csv"),
                       "--keep", "good", "unsure")
    assert images(files) == {"nt_control/a", "nt_control/c", "ogd_3/d"}


def test_export_stops_when_csv_matches_no_image(monkeypatch, data_dir):
    write_csv(data_dir / "other.csv", [("some_folder/x.tif", "good")])
    with pytest.raises(SystemExit, match="none of the 1 files"):
        run_export(monkeypatch, data_dir, data_dir / "x", "--review_csv", str(data_dir / "other.csv"))


# --- class sets (--classes) -------------------------------------------------------------------

ALL_SIX = ["nt_control", "ogd_0.5", "ogd_1.5", "ogd_3", "ogd_azo_1.5", "ogd_sod_3"]


@pytest.fixture
def six_folders(tmp_path):
    """The six condition folders of the real data set, 5 small 3-channel images each."""
    rng = np.random.default_rng(1)
    for condition in ALL_SIX:
        (tmp_path / condition).mkdir()
        for i in range(5):
            tifffile.imwrite(tmp_path / condition / f"{condition}_{i}.tif",
                             rng.integers(0, 4000, (3, 32, 32), dtype=np.uint16), photometric="minisblack")
    return tmp_path


def run_shortcut(monkeypatch, data_dir, out_dir, *extra):
    monkeypatch.setattr(sys, "argv", ["shortcut_baseline_fluorescent.py", str(data_dir), "--prep", "raw",
                                      "--out_dir", str(out_dir), *extra])
    sbf.main()


@pytest.mark.parametrize("classes, folders_read, class_line", [
    ("fluorescent_ogd_vs_control", 4, "nt_control = nt_control; ogd = ogd_0.5 + ogd_1.5 + ogd_3"),
    ("fluorescent_ogd_3_vs_control", 2, "nt_control = nt_control; ogd_3 = ogd_3"),
])
def test_shortcut_class_sets_read_only_their_folders(monkeypatch, six_folders, tmp_path, capsys,
                                                    classes, folders_read, class_line):
    run_shortcut(monkeypatch, six_folders, tmp_path / "out", "--classes", classes)
    text = (tmp_path / "out" / f"shortcut_fluorescent_raw_{classes}.txt").read_text()
    assert class_line in text
    train, val = (int(n) for n in
                  text.split("train images: ")[1].split("\n")[0].split("   validation images: "))
    assert train + val == folders_read * 5                     # azo and sod are never read
    assert "pure guessing gives 0.500" in text                  # two classes


def test_shortcut_class_set_missing_folder_stops(monkeypatch, six_folders, tmp_path):
    import shutil
    shutil.rmtree(six_folders / "ogd_1.5")
    with pytest.raises(SystemExit, match="ogd_1.5"):
        run_shortcut(monkeypatch, six_folders, tmp_path / "out", "--classes", "fluorescent_ogd_vs_control")


def test_shortcut_with_class_set_and_review_csv(monkeypatch, six_folders, tmp_path):
    write_csv(tmp_path / "review.csv", [(f"{c}/{c}_{i}.tif", "bad" if i == 0 else "good")
                                        for c in ALL_SIX for i in range(5)])
    run_shortcut(monkeypatch, six_folders, tmp_path / "out", "--classes", "fluorescent_ogd_3_vs_control",
                 "--review_csv", str(tmp_path / "review.csv"))
    text = (tmp_path / "out" / "shortcut_fluorescent_raw_fluorescent_ogd_3_vs_control_good.txt").read_text()
    assert "left out: 2 bad" in text                            # one per folder read, azo/sod not counted


def test_training_dataset_uses_the_same_class_sets(six_folders):
    """MicrogliaDataset with a fluorescent class set reads only its folders and merges the OGD ones."""
    pytest.importorskip("torch")
    from saffron.data.datasets import MicrogliaDataset
    for name, (labels, merge_map) in sbf.FLUORESCENT_CLASS_SETS.items():
        dataset = MicrogliaDataset(six_folders, labels=labels, merge_map=merge_map)
        read = {Path(path).parent.name: labels[label] for path, label in dataset.samples}
        if name == "fluorescent_ogd_vs_control":
            assert read == {"nt_control": "nt_control", "ogd_0.5": "ogd", "ogd_1.5": "ogd", "ogd_3": "ogd"}
        elif name == "fluorescent_ogd_3_vs_control":
            assert read == {"nt_control": "nt_control", "ogd_3": "ogd_3"}
        else:
            assert read == {c: c for c in ALL_SIX}


def test_training_script_exposes_the_fluorescent_class_sets():
    pytest.importorskip("torch")
    pytest.importorskip("umap")
    import supconloss_test_old as training
    for name, value in sbf.FLUORESCENT_CLASS_SETS.items():
        assert training.CLASS_SETS[name] == value
    assert {"all", "rodent", "condition"} <= training.CLASS_SETS.keys()
