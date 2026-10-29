"""End-to-end CLI tests (train -> evaluate -> report) on CPU, 2 folds."""

import glob
import json
import os

import pandas as pd
import polars as pl
import pytest
import yaml

from neuralnet_lightning.__main__ import main
from neuralnet_lightning.binary.report import LONG_COLUMNS


@pytest.fixture
def config_path(tmp_path, per_ring_schema):
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump({
        "name": "smoke_run",
        "model": "MLP",
        "results_root": str(tmp_path / "results"),
        "dataset": {"data_path": per_ring_schema.data_path, "rings_col": "cl_ring_%i",
                    "et_col": "cl_et", "eta_col": "cl_eta"},
        "model_args": {"hidden": 3},
        "training": {"max_epochs": 5, "batch_size": 64, "learning_rate": 0.05, "patience": 5,
                     "n_splits": 2, "n_inits": 1,
                     "accelerator": "cpu", "devices": 1},
    }))
    return str(path)


def test_train_evaluate_report(config_path, tmp_path):
    run = tmp_path / "results" / "smoke_run" / "all"

    assert main(["train", "--config", config_path]) == 0
    sidecars = sorted(glob.glob(str(run / "checkpoints" / "*.json")))
    assert [os.path.basename(p) for p in sidecars] == ["fold_1_init_1.json", "fold_2_init_1.json"]
    sidecar = json.loads(open(sidecars[0]).read())
    assert sidecar["model_kwargs"]["hidden"] == 3 and sidecar["model_kwargs"]["input_dim"] == 50
    assert sidecar["fold_kwargs"]["pos_weight"] == pytest.approx(1.0)
    assert sidecar["n_train"] + sidecar["n_val"] == 600
    assert os.path.exists(run / sidecar["checkpoint"])

    # Retraining one (fold, init) replaces only its own artefacts.
    assert main(["train", "--config", config_path, "--fold", "2", "--init", "1"]) == 0
    assert len(glob.glob(str(run / "checkpoints" / "*.ckpt"))) == 2

    assert main(["evaluate", "--config", config_path]) == 0
    long_df = pd.read_csv(run / "metrics" / "folds_long.csv")
    assert list(long_df.columns) == LONG_COLUMNS
    assert len(long_df) == 2 * 3  # folds x operating points
    assert long_df["auc_roc"].min() > 0.9  # the synthetic classes are well separated

    scores = pl.read_parquet(run / "scores" / "fold_1.parquet")
    assert scores.columns == ["y_true", "y_score", "in_sample", "et", "eta"]
    assert scores.height == 600 and (~scores["in_sample"]).sum() == 300
    assert os.path.exists(run / "plots" / "roc_folds.pdf")

    assert main(["evaluate", "--config", config_path, "--reuse-scores", "--no-plots"]) == 0
    assert main(["report", "--config", config_path]) == 0
    table_dir = tmp_path / "results" / "smoke_run" / "pd_table"
    for stem in ("pd_table_tight", "pd_table_integrated"):
        assert os.path.exists(table_dir / f"{stem}.tex") and os.path.exists(table_dir / f"{stem}.html")
        assert r"smoke\_run" in (table_dir / f"{stem}.tex").read_text()  # '_' escaped for LaTeX


def test_untrained_region_is_a_clean_error(config_path):
    assert main(["evaluate", "--config", config_path, "--et-bin", "0", "--eta-bin", "0"]) == 1


def test_init_needs_fold(config_path):
    assert main(["train", "--config", config_path, "--init", "1"]) == 1


def test_fused_train_evaluate(tmp_path, per_ring_schema):
    path = tmp_path / "fused.yaml"
    path.write_text(yaml.safe_dump({
        "name": "smoke_fused",
        "model": "Fused",
        "results_root": str(tmp_path / "results"),
        "dataset": {"data_path": per_ring_schema.data_path, "rings_col": "cl_ring_%i",
                    "et_col": "cl_et", "eta_col": "cl_eta"},
        "training": {"max_epochs": 5, "batch_size": 64, "learning_rate": 0.01, "patience": 5,
                     "n_splits": 2, "n_inits": 1, "accelerator": "cpu", "devices": 1},
    }))
    run = tmp_path / "results" / "smoke_fused" / "all"

    assert main(["train", "--config", str(path)]) == 0
    sidecar = json.loads((run / "checkpoints" / "fold_1_init_1.json").read_text())
    assert sidecar["model_kwargs"]["n_rings"] == 100 and sidecar["model_kwargs"]["cell_shape"] == [7, 7, 15]

    assert main(["evaluate", "--config", str(path), "--no-plots"]) == 0
    long_df = pd.read_csv(run / "metrics" / "folds_long.csv")
    assert len(long_df) == 2 * 3
    assert long_df["auc_roc"].min() > 0.9


def test_cells_on_a_dataset_without_them_is_a_clean_error(tmp_path, listed_schema):
    path = tmp_path / "cnn2d.yaml"
    path.write_text(yaml.safe_dump({
        "name": "cnn2d_on_mc21",
        "model": "CNN2D",
        "results_root": str(tmp_path / "results"),
        "dataset": {"data_path": listed_schema.data_path, "rings_col": "Cluster.ringsE",
                    "et_col": "Cluster.et", "eta_col": "Cluster.eta", "label_col": "target"},
        "training": {"accelerator": "cpu", "devices": 1},
    }))
    assert main(["train", "--config", str(path)]) == 1
