"""Tests for cross-validation by sample, the one-pass validation metrics, pool_folds.py,
plot_runs.py's balanced accuracy, and the saliency sanity-check helpers."""

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from saffron.data.datasets import sample_id, split_indices_by_sample, split_indices  # noqa: E402
import pool_folds  # noqa: E402
import plot_runs  # noqa: E402


# --- sample IDs and the split ---------------------------------------------------------------

@pytest.mark.parametrize("name, expected", [
    ("4-56-10_40x_cortex_1.tif", "4-56-10"),
    ("4-56-10_40x_cortex_1_top_left.npy", "4-56-10"),
    ("4-59_1_40x_thalamus_2.tif", "4-59_1"),                # underscore inside the sample ID
    ("nt_control/4-50-4_40x_thalamus_5_bottom_right_top_left.npy", "4-50-4"),
])
def test_sample_id(name, expected):
    assert sample_id(name) == expected


def test_sample_id_without_magnification_raises():
    with pytest.raises(ValueError, match="no sample ID"):
        sample_id("mice_p12_image_01.npy")


def fake_dataset(tmp_path, samples_per_folder):
    """A dataset-like object: folder -> number of samples, 3 fields x 4 pieces per sample."""
    rows = []
    for label, (folder, n_samples) in enumerate(samples_per_folder.items()):
        for s in range(n_samples):
            for field in range(3):
                for piece in ["top_left", "top_right", "bottom_left", "bottom_right"]:
                    rows.append((str(tmp_path / folder / f"4-{label}-{s}_40x_cortex_{field}_{piece}.npy"), label))
    return SimpleNamespace(samples=rows, data_dir=tmp_path)


def samples_of(dataset, indices):
    return {sample_id(dataset.samples[i][0]) for i in indices}


def test_every_sample_validated_exactly_once(tmp_path):
    dataset = fake_dataset(tmp_path, {"nt_control": 4, "ogd_3": 5, "ogd_0.5": 5})
    seen = []
    for fold in range(4):
        train, val = split_indices_by_sample(dataset, n_folds=4, fold=fold)
        assert not samples_of(dataset, train) & samples_of(dataset, val)      # no sample on both sides
        assert sorted(train + list(val)) == list(range(len(dataset.samples)))  # every file used once
        folders = {Path(dataset.samples[i][0]).parent.name for i in val}
        assert folders == {"nt_control", "ogd_3", "ogd_0.5"}                   # each fold holds out every folder
        seen += sorted(samples_of(dataset, val))
    assert sorted(seen) == sorted({sample_id(path) for path, _ in dataset.samples})


def test_split_is_repeatable_and_independent_of_other_folders(tmp_path):
    both = fake_dataset(tmp_path, {"nt_control": 4, "ogd_3": 5})
    one = fake_dataset(tmp_path, {"nt_control": 4})
    a = samples_of(both, split_indices_by_sample(both, n_folds=4, fold=1)[1])
    b = samples_of(both, split_indices_by_sample(both, n_folds=4, fold=1)[1])
    c = samples_of(one, split_indices_by_sample(one, n_folds=4, fold=1)[1])
    assert a == b
    assert {s for s in a if s.startswith("4-0-")} == c


def test_folder_with_fewer_samples_than_folds(tmp_path, capsys):
    """ogd_1.5 after cleaning: 3 samples, 4 folds. Every sample still validated once."""
    dataset = fake_dataset(tmp_path, {"nt_control": 4, "ogd_1.5": 3})
    seen = []
    for fold in range(4):
        _, val = split_indices_by_sample(dataset, n_folds=4, fold=fold)
        seen += sorted(samples_of(dataset, val))
    assert sorted(seen) == sorted({sample_id(path) for path, _ in dataset.samples})
    assert len(seen) == len(set(seen)) == 7
    assert "ogd_1.5 has 3 samples, fewer than 4 folds; folds 3..3 validate on none of it" in capsys.readouterr().out


def test_fewer_than_two_folds_raises(tmp_path):
    with pytest.raises(ValueError, match="at least 2"):
        split_indices_by_sample(fake_dataset(tmp_path, {"a": 4}), n_folds=1, fold=0)


def test_fold_out_of_range_raises(tmp_path):
    with pytest.raises(ValueError, match="fold must be"):
        split_indices_by_sample(fake_dataset(tmp_path, {"a": 4}), n_folds=4, fold=4)


def test_split_indices_image_is_unchanged(tmp_path):
    from saffron.data.datasets import split_indices_by_image
    dataset = fake_dataset(tmp_path, {"nt_control": 4, "ogd_3": 5})
    assert split_indices(dataset, split="image") == split_indices_by_image(dataset, val_split=0.2, seed=42)
    with pytest.raises(ValueError, match="split must be"):
        split_indices(dataset, split="animal")


# --- pool_folds.py --------------------------------------------------------------------------

def test_balanced_accuracy_matches_sklearn():
    from sklearn.metrics import balanced_accuracy_score
    rng = np.random.default_rng(0)
    true = rng.choice(["ctrl", "ogd"], size=200, p=[0.25, 0.75])
    predicted = rng.choice(["ctrl", "ogd"], size=200)
    assert pool_folds.balanced_accuracy(true, predicted) == pytest.approx(balanced_accuracy_score(true, predicted))
    # always guessing the big class: high accuracy, balanced accuracy 0.5
    assert pool_folds.balanced_accuracy(true, ["ogd"] * 200) == pytest.approx(0.5)


def rows_for(run, fold, samples, data_dir="/data/li_raw_256_good", stage="2"):
    rows = []
    for sample, true, correct in samples:
        for piece in range(4):
            predicted = true if piece < correct else ("ogd" if true == "ctrl" else "ctrl")
            rows.append({"file": f"{true}/{sample}_40x_{piece}.npy", "sample": sample, "folder": true,
                         "true": true, "predicted": predicted, "fold": str(fold), "stage": stage,
                         "classes": "fluorescent_ogd_3_vs_control", "data_dir": data_dir, "run": run})
    return rows


def test_summarize_reports_folds_and_samples():
    rows = rows_for("job1_task0", 0, [("4-50-4", "ctrl", 4), ("4-50-1", "ogd", 1)]) \
        + rows_for("job1_task1", 1, [("4-50-7", "ctrl", 2), ("4-50-5", "ogd", 4)])
    text = "\n".join(pool_folds.summarize(rows, ["ctrl", "ogd"]))
    assert "pieces: 16   samples: 4   folds: 2" in text
    assert "balanced accuracy 0.688" in text        # ctrl 6/8, ogd 5/8
    assert "fold 0: balanced accuracy 0.625" in text
    assert "4-50-1" in text and "<- mostly wrong" in text


def test_pool_folds_cli_groups_and_warns(tmp_path, capsys, monkeypatch):
    for name, rows in {
        "a.csv": rows_for("job1_task0", 0, [("4-50-4", "ctrl", 4)]),
        "b.csv": rows_for("job1_task1", 1, [("4-50-7", "ctrl", 4)]),
        "c.csv": rows_for("job1_task4", 0, [("4-50-4", "ctrl", 4)], data_dir="/data/li_saved_256_good"),
        "d.csv": rows_for("job2_task0", 0, [("4-50-4", "ctrl", 4)]),     # the same fold again
    }.items():
        pool_folds.write_predictions(tmp_path / name, rows)
    monkeypatch.setattr(sys, "argv", ["pool_folds.py", *(str(tmp_path / n) for n in "abcd"
                                                         for n in [f"{n}.csv"])])
    pool_folds.main()
    out = capsys.readouterr().out
    assert out.count("classes fluorescent_ogd_3_vs_control   stage 2") == 2      # li_raw and li_saved
    assert "WARNING: 4 pieces are in the validation set of more than one run" in out


# --- training script: one-pass validation ---------------------------------------------------

def test_evaluate_loss_does_not_depend_on_batch_size():
    torch = pytest.importorskip("torch")
    pytest.importorskip("umap")
    pytest.importorskip("pytorch_metric_learning")
    from torch.utils.data import DataLoader, TensorDataset
    import supconloss_test_old as training

    torch.manual_seed(0)
    images = torch.randn(26, 1, 8, 8)
    labels = torch.tensor([0] * 7 + [1] * 19)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 2))
    embed = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 5),
                                torch.nn.Unflatten(1, (5,)))
    loss_func = training.SupConLoss(temperature=0.07)
    results = [training.evaluate(model, embed, DataLoader(TensorDataset(images, labels), batch_size=b),
                                 loss_func, "cpu") for b in (24, 26, 5)]
    assert results[0]["loss"] == pytest.approx(results[1]["loss"]) == pytest.approx(results[2]["loss"])
    with torch.no_grad():
        predicted = model(images).argmax(1).numpy()
    assert results[0]["accuracy"] == pytest.approx(np.mean(predicted == labels.numpy()))
    assert results[0]["balanced_accuracy"] == pytest.approx(pool_folds.balanced_accuracy(labels.numpy(), predicted))


def test_report_validation_writes_rows_in_loader_order(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    pytest.importorskip("umap")
    import supconloss_test_old as training
    from saffron.data.datasets import TransformedSubset

    dataset = fake_dataset(tmp_path, {"nt_control": 1, "ogd_3": 1})
    indices = [12, 0, 13, 1]                                  # validation order (shuffled)
    data_val = SimpleNamespace(dataset=TransformedSubset(dataset, indices))
    result = {"true": np.array([1, 0, 1, 0]), "predicted": np.array([1, 0, 0, 0])}
    monkeypatch.chdir(tmp_path)
    training.report_validation(result, data_val, ["nt_control", "ogd_3"], 2, "job9_task3",
                               "fluorescent_ogd_3_vs_control", tmp_path, 1)
    with open(tmp_path / "predictions" / "job9_task3_stage2.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["file"] for r in rows] == [str(Path(dataset.samples[i][0]).relative_to(tmp_path)) for i in indices]
    assert [r["true"] for r in rows] == ["ogd_3", "nt_control", "ogd_3", "nt_control"]
    assert rows[2]["predicted"] == "nt_control" and rows[0]["sample"] == "4-1-0" and rows[0]["fold"] == "1"
    assert "balanced accuracy 0.750" in capsys.readouterr().out


# --- plot_runs.py ---------------------------------------------------------------------------

def test_parse_log_reads_balanced_accuracy_and_old_logs(tmp_path):
    new = tmp_path / "new.err"
    new.write_text(
        "Batch: 100%|##| 3/3 [00:01<00:00, Loss=2.5]\n"
        "Epoch:  50%|#| 1/2 [00:02<00:02, Loss=4.1, Acc=0.8, BalAcc=0.6]\n"
        "Batch: 100%|##| 3/3 [00:01<00:00, Loss=2.4]\n"
        "Epoch: 100%|##| 2/2 [00:04<00:00, Loss=4.0, Acc=0.85, BalAcc=0.7]\n")
    old = tmp_path / "old.err"
    old.write_text("Batch: 100%|##| 3/3 [00:01<00:00, Loss=2.5]\n"
                   "Epoch: 100%|##| 1/1 [00:02<00:00, Loss=4.1, Acc=0.8]\n")
    stage = plot_runs.parse_log(new)[0]
    assert stage["val_acc"] == [0.8, 0.85] and stage["val_bal_acc"] == [0.6, 0.7]
    assert stage["val_loss"] == [4.1, 4.0]
    old_stage = plot_runs.parse_log(old)[0]
    assert old_stage["val_acc"] == [0.8] and np.isnan(old_stage["val_bal_acc"][0])


def test_run_label_names_class_set_and_fold(tmp_path):
    (tmp_path / "logs_output").mkdir()
    (tmp_path / "logs_error").mkdir()
    (tmp_path / "logs_output" / "run.out").write_text(
        "Prep: li_raw   piece size: 256   data: x\n  classes              = fluorescent_ogd_3_vs_control\n"
        "  split_by             = sample\n  fold                 = 2\n  learning_rate        = 0.0003\n"
        "Loaded 368 images\nClass distribution: {'nt_control': 180, 'ogd_3': 188}\n")
    err = tmp_path / "logs_error" / "run.err"
    label = plot_runs.run_label(err)
    assert label.startswith("li_raw 256px, ogd_3 vs control, fold 2, lr 0.0003")
    assert plot_runs.class_count(err) == 2
    assert plot_runs.guess_line(err, label) == (0.51, "always guess the largest class")    # 188 / 368


# --- brightness test with sample folds ------------------------------------------------------

def test_shortcut_sample_split_validates_every_sample_once(tmp_path, monkeypatch):
    import shortcut_baseline_fluorescent as sbf
    rng = np.random.default_rng(0)
    for condition, prefix in [("nt_control", "4-50"), ("ogd_3", "4-51")]:
        (tmp_path / "data" / condition).mkdir(parents=True)
        for s in range(4):
            for field in range(3):
                tifffile.imwrite(tmp_path / "data" / condition / f"{prefix}-{s}_40x_cortex_{field}.tif",
                                 rng.integers(0, 4000, (3, 32, 32), dtype=np.uint16), photometric="minisblack")
    monkeypatch.setattr(sys, "argv", ["shortcut_baseline_fluorescent.py", str(tmp_path / "data"), "--prep", "raw",
                                      "--classes", "fluorescent_ogd_3_vs_control", "--split_by", "sample",
                                      "--n_folds", "4", "--out_dir", str(tmp_path / "out")])
    sbf.main()
    text = (tmp_path / "out" / "shortcut_fluorescent_raw_fluorescent_ogd_3_vs_control_sample4fold.txt").read_text()
    assert "4-fold cross-validation by sample" in text
    assert text.count("validation 6 images") == 4                       # 2 samples x 3 fields per fold
    per_sample = text.split("per sample (share")[1]
    assert all(f"{p}-{s} " in per_sample for p in ["4-50", "4-51"] for s in range(4))


# --- saliency helpers -----------------------------------------------------------------------

def test_cell_pixels_and_rank_correlation():
    pytest.importorskip("torch")
    pytest.importorskip("umap")
    import saliency_map
    mask = np.zeros((20, 20))
    mask[5:9, 5:9] = 1.0
    assert saliency_map.cell_pixels(mask).sum() == 16                   # the mask's white pixels, not 15%
    image = np.arange(100.0).reshape(10, 10)
    assert saliency_map.cell_pixels(image).sum() == 15                  # brightest 15% of a grey image
    a = np.arange(16.0).reshape(4, 4)
    assert saliency_map.rank_correlation(a, a ** 2) == pytest.approx(1.0)
    assert np.isnan(saliency_map.rank_correlation(a, np.zeros((4, 4))))
