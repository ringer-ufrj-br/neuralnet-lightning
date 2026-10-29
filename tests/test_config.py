import glob
import os

import pytest
import yaml

from neuralnet_lightning.config import ConfigError, from_dict, load_config

DATASET = {"data_path": "data/", "rings_col": "cl_ring_%i", "et_col": "cl_et", "eta_col": "cl_eta"}


def test_defaults_live_in_the_dataclasses():
    config = from_dict({"name": "exp", "model": "MLP", "dataset": DATASET})
    assert config.results_root == "results"
    assert config.training.batch_size == 1024 and config.training.n_splits == 10
    assert config.evaluation.operating_points == {"tight": 0.90, "medium": 0.95, "loose": 0.99}
    assert config.dataset.label_col is None and config.model_args == {}


@pytest.mark.parametrize("raw, match", [
    ({"name": "exp", "model": "MLP", "dataset": DATASET, "max_epoch": 3}, r"Unknown key\(s\) in 'config'"),
    ({"name": "exp", "model": "MLP", "dataset": DATASET, "training": {"bacth_size": 8}}, r"Unknown key\(s\) in 'training'"),
    ({"name": "exp", "model": "MLP", "dataset": {**DATASET, "label": "target"}}, r"Unknown key\(s\) in 'dataset'"),
    ({"model": "MLP", "dataset": DATASET}, "Missing required key 'name'"),
    ({"name": "exp", "model": "MLP", "dataset": {"data_path": "x"}}, "Missing required key 'rings_col'"),
    ({"name": "exp", "model": "MLP", "dataset": DATASET, "training": {"batch_size": "big"}}, "must be int"),
    ({"name": "exp", "model": "MLP", "dataset": DATASET, "training": {"n_splits": 0}}, "n_splits must be >= 1"),
    ({"name": "exp", "model": "MLP", "dataset": DATASET,
      "evaluation": {"operating_points": {"tight": 90}}}, r"must be in \(0, 1\)"),
    ({"name": "a/b", "model": "MLP", "dataset": DATASET}, "plain directory name"),
])
def test_invalid_configs_are_rejected(raw, match):
    with pytest.raises(ConfigError, match=match):
        from_dict(raw)


def test_load_config_with_cli_overrides(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump({"name": "exp", "model": "MLP", "dataset": DATASET,
                                    "training": {"learning_rate": 1, "devices": 1}}))
    config = load_config(str(path), accelerator="cpu", devices=None)
    assert config.training.accelerator == "cpu" and config.training.devices == 1
    assert config.training.learning_rate == 1  # an int is accepted where a float is expected


def test_dataset_from_a_file(tmp_path):
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "mc25.yaml").write_text(yaml.safe_dump(DATASET))
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump({"name": "exp", "model": "MLP", "dataset": "datasets/mc25.yaml"}))

    # Resolved against the config's folder, not the working directory.
    config = load_config(str(path))
    assert config.dataset.rings_col == "cl_ring_%i" and config.dataset.max_files is None


def test_dataset_file_with_overrides(tmp_path):
    (tmp_path / "mc25.yaml").write_text(yaml.safe_dump(DATASET))
    config = from_dict({"name": "exp", "model": "MLP", "dataset": {"file": "mc25.yaml", "max_files": 3}},
                       base_dir=str(tmp_path))
    assert config.dataset.max_files == 3 and config.dataset.et_col == "cl_et"

    with pytest.raises(ConfigError, match=r"Unknown key\(s\) in 'dataset'"):
        from_dict({"name": "exp", "model": "MLP", "dataset": {"file": "mc25.yaml", "max_file": 3}},
                  base_dir=str(tmp_path))


def test_missing_files_are_config_errors(tmp_path):
    with pytest.raises(ConfigError, match="Dataset file .* not found"):
        from_dict({"name": "exp", "model": "MLP", "dataset": "nope.yaml"}, base_dir=str(tmp_path))
    with pytest.raises(ConfigError, match="Config file .* not found"):
        load_config(str(tmp_path / "nope.yaml"))


def test_shipped_configs_are_valid():
    configs = os.path.join(os.path.dirname(__file__), "..", "configs")
    paths = glob.glob(os.path.join(configs, "*.yaml"))
    assert len(paths) >= 4
    for path in paths:
        load_config(path)
    # Every dataset file is valid on its own, whether or not an experiment uses it yet.
    for path in glob.glob(os.path.join(configs, "datasets", "*.yaml")):
        from_dict({"name": "exp", "model": "MLP", "dataset": os.path.abspath(path)})
