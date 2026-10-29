"""
Where every artefact of one run lives, derived from (results_root, experiment name, region).

Kept in one dataclass so training, evaluation, the report and the shell launchers agree on the
layout by construction instead of each joining the same strings:

    <results_root>/<name>/<region>/
        artifacts/    preprocessor.joblib, val_indices_fold_N.npy
        checkpoints/  fold_N_init_M.ckpt + fold_N_init_M.json (the sidecar evaluate reads)
        history/      fold_N_init_M.csv
        logs/         Lightning's own logs
        scores/       fold_N.parquet
        metrics/      folds_long.csv
        plots/
"""

import os
from dataclasses import dataclass
from typing import Optional

from neuralnet_lightning.binning import FULL_PHASE_SPACE, Region


@dataclass(frozen=True)
class RunPaths:
    """Artefact paths of one (experiment, region). Nothing is created on construction."""

    root: str

    @classmethod
    def for_run(cls, results_root: str, name: str, region: Optional[Region]) -> "RunPaths":
        """The run directory of `name` in `region` (None: the whole phase space)."""
        return cls(os.path.join(results_root, name, region.label if region else FULL_PHASE_SPACE))

    @property
    def artifacts(self) -> str:
        return os.path.join(self.root, "artifacts")

    @property
    def checkpoints(self) -> str:
        return os.path.join(self.root, "checkpoints")

    @property
    def history(self) -> str:
        return os.path.join(self.root, "history")

    @property
    def logs(self) -> str:
        return os.path.join(self.root, "logs")

    @property
    def scores(self) -> str:
        return os.path.join(self.root, "scores")

    @property
    def metrics(self) -> str:
        return os.path.join(self.root, "metrics")

    @property
    def plots(self) -> str:
        return os.path.join(self.root, "plots")

    @property
    def preprocessor(self) -> str:
        return os.path.join(self.artifacts, "preprocessor.joblib")

    @property
    def folds_long(self) -> str:
        """The region's per-fold metrics table - also the 'region is evaluated' marker."""
        return os.path.join(self.metrics, "folds_long.csv")

    @staticmethod
    def stem(fold: int, init: int) -> str:
        """The name every per-model artefact shares, e.g. 'fold_3_init_2'."""
        return f"fold_{fold}_init_{init}"

    def checkpoint(self, fold: int, init: int) -> str:
        return os.path.join(self.checkpoints, f"{self.stem(fold, init)}.ckpt")

    def sidecar(self, fold: int, init: int) -> str:
        return os.path.join(self.checkpoints, f"{self.stem(fold, init)}.json")

    def history_file(self, fold: int, init: int) -> str:
        return os.path.join(self.history, f"{self.stem(fold, init)}.csv")

    def val_indices(self, fold: int) -> str:
        return os.path.join(self.artifacts, f"val_indices_fold_{fold}.npy")

    def scores_file(self, fold: int) -> str:
        return os.path.join(self.scores, f"fold_{fold}.parquet")

    def rel(self, path: str) -> str:
        """`path` relative to the run directory, as stored in sidecars so a tree can be moved."""
        return os.path.relpath(path, self.root)

    def resolve(self, rel_path: str) -> str:
        """Inverse of `rel`."""
        return os.path.join(self.root, rel_path)
