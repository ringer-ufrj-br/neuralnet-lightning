"""
Shared training/evaluation pipeline.

The pipeline is split into two independent phases so that they can be run as separate
commands, on different machines and at different times:

* `train`    - loads data, fits the preprocessor on the training rows only, runs the K-Fold
               cross-validation and persists everything an evaluation needs: one checkpoint
               per fold under a fixed name, the preprocessor, each fold's validation indices and
               a manifest describing the dataset the split was derived from.
* `evaluate` - reloads those artefacts, re-runs inference over the whole region for every fold,
               persists the raw scores and produces the per-fold metrics, the plots and the
               region's slice of the cross-validation table.

Keeping them apart means re-scoring, re-plotting or re-cutting the working points never
requires retraining, and the numbers that end up in the table always come from a checkpoint
that is on disk and addressable.
"""

import glob
import hashlib
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

logger = logging.getLogger(__name__)

# Ensure root directory is in path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from ai.loader.loader import DataLoader
from ai.label.label_generator import validate_files
from ai.trainer.trainer import ModelTrainer
from ai.evaluation.monitor import ModelMonitor
from ai.evaluation.summary import (
    DEFAULT_OPERATING_POINTS,
    ModelSummary,
    compute_metrics,
    compute_operating_points,
)
from ai.binning.kinematics import GRID
from ai.preprocess.base import ET, ETA, LABEL, ROW_ID, DatasetSchema


def _atomic_write_json(payload: Dict[str, Any], filepath: str) -> str:
    """
    Writes JSON via a temporary file plus os.replace.

    Under SLURM every fold of a region is its own process writing the same manifest; a plain
    open()/write() lets a reader observe a half-written file. os.replace is atomic on POSIX,
    so a reader always sees either the old or the new complete file.

    Args:
        payload (Dict[str, Any]): JSON-serialisable content.
        filepath (str): Destination path.

    Returns:
        str: The written path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    tmp_path = f"{filepath}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, filepath)
    return filepath


class BasePipeline:
    """
    End-to-end training and evaluation pipeline shared by every model architecture.

    Subclasses declare the model class and the preprocessor, plus whatever model kwargs are
    derived from the feature array; everything else - data loading, kinematic binning, the
    cross-validation, artefact persistence, scoring and reporting - is common.
    """

    #: LightningModule subclass this pipeline trains (a BaseBinaryClassifier subclass).
    model_class: Type[pl.LightningModule]

    #: Preprocessor class for this architecture (a BasePreprocessor subclass).
    preprocessor_class: Type[Any]

    #: Registry name, set by @register_pipeline. Also the results/<NAME>/ directory.
    model_name: str = "Model"

    #: Metric monitored by EarlyStopping/ModelCheckpoint, and its improvement direction.
    monitor_metric: str = "val_sp"
    monitor_mode: str = "max"

    def __init__(
        self,
        schema: Optional[DatasetSchema] = None,
        results_root: str = "results",
        max_epochs: int = 20,
        batch_size: int = 32,
        patience: int = 5,
        accelerator: str = "auto",
        devices: Union[int, str, List[int]] = "auto",
        et_bin: Optional[int] = None,
        eta_bin: Optional[int] = None
    ) -> None:
        """
        Initializes the pipeline and resolves the results directory for this kinematic region.

        Nothing here knows the dataset's column names: the schema translates whatever is on
        disk into the canonical `label` / `et` / `eta` / `ring_i` vocabulary every stage below
        works in, and the binning carries the region edges the dataset was defined against.

        Args:
            schema (Optional[DatasetSchema]): Dataset layout. Defaults to the mc25 layout.
            results_root (str): Root the artefacts are written under, as
                `<results_root>/<model>/<region>`. Give each dataset its own root: the region
                directories are named after bin *indices*, so two datasets sharing a root
                would silently overwrite each other's identically-named regions. Defaults to
                'results'.
            max_epochs (int): Maximum training epochs. Defaults to 20.
            batch_size (int): Training batch size. Defaults to 32.
            patience (int): Early stopping patience. Defaults to 5.
            accelerator (str): PyTorch Lightning accelerator ('auto', 'cpu', 'cuda'). Defaults to 'auto'.
            devices (Union[int, str, List[int]]): Devices specification. Defaults to 'auto'.
            et_bin (Optional[int]): Et bin index. Trains on the whole dataset when None
                (together with eta_bin) or on only that kinematic slice when both are set -
                the Ringer one-network-per-region scheme. Defaults to None.
            eta_bin (Optional[int]): |eta| bin index. Defaults to None.

        Raises:
            ValueError: If exactly one of et_bin/eta_bin is set, or if the region is outside
                the configured grid.
        """
        if (et_bin is None) != (eta_bin is None):
            raise ValueError("❌ et_bin and eta_bin must be set together (or both left as None).")

        self.schema = schema or DatasetSchema()
        self.label_col = LABEL
        self.data_path = self.schema.data_path
        self.max_files = self.schema.max_files
        self.et_bin = et_bin
        self.eta_bin = eta_bin

        self.results_root = results_root
        self.results_dir = os.path.join(results_root, self.model_name)
        if et_bin is not None:
            GRID.validate(et_bin, eta_bin)
            self.results_dir = os.path.join(self.results_dir, GRID.bin_label(et_bin, eta_bin))
            logger.info(f"🎯 Kinematic bin selected: {GRID.bin_description(et_bin, eta_bin)}")

        self.artifacts_dir = os.path.join(self.results_dir, "artifacts")
        self.checkpoints_dir = os.path.join(self.results_dir, "checkpoints")
        self.history_dir = os.path.join(self.results_dir, "history")
        self.scores_dir = os.path.join(self.results_dir, "scores")
        self.manifest_path = os.path.join(self.artifacts_dir, "manifest.json")
        self.preprocessor_path = os.path.join(self.artifacts_dir, "preprocessor.joblib")

        self.loader = DataLoader(data_path=self.data_path, max_files=self.max_files)
        self.preprocessor = self.build_preprocessor()

        self.trainer = ModelTrainer(
            max_epochs=max_epochs,
            batch_size=batch_size,
            patience=patience,
            log_dir=os.path.join(self.results_dir, "lightning_logs"),
            checkpoint_dir=self.checkpoints_dir,
            accelerator=accelerator,
            devices=devices,
            monitor_metric=self.monitor_metric,
            monitor_mode=self.monitor_mode
        )

        self.monitor = ModelMonitor(output_dir=os.path.join(self.results_dir, "plots"))
        self.summary = ModelSummary(output_dir=os.path.join(self.results_dir, "metrics"))

    # ------------------------------------------------------------------ hooks

    def build_preprocessor(self) -> Any:
        """
        Builds the preprocessor for this architecture. The default instantiates
        `preprocessor_class` with no arguments; override only if it needs constructor
        arguments.

        Returns:
            Any: A fresh, unfitted preprocessor instance.

        Raises:
            NotImplementedError: If the subclass declares neither preprocessor_class nor an
                override.
        """
        if getattr(self, "preprocessor_class", None) is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set preprocessor_class or override build_preprocessor()."
            )
        return self.preprocessor_class()

    def build_model_kwargs(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Builds the constructor kwargs for the model, given the (already preprocessed) training
        feature array - this is where an architecture picks up e.g. its input dimension.

        Args:
            X (np.ndarray): Preprocessed training features.

        Returns:
            Dict[str, Any]: Keyword arguments for model_class.
        """
        return {}

    def required_columns(self, available: List[str]) -> Optional[List[str]]:
        """
        Declares which dataset columns this architecture actually consumes, so the loader can
        prune everything else at the parquet scan (the raw files carry 300+ columns, most of
        them nested calorimeter images no MLP-style model ever touches - loading them all is
        what used to exhaust memory on full-dataset runs).

        Args:
            available (List[str]): Column names present in the dataset files.

        Returns:
            Optional[List[str]]: Columns to load, or None to load every column.
        """
        return self.preprocessor.required_columns(available)

    # ------------------------------------------------------------------ data

    def load_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Loads the dataset in the canonical column vocabulary and applies the kinematic cut.

        Runs as a single lazy polars query - the join with any side table, the column
        projection, the label derivation and the region filter all happen inside the streaming
        parquet scan - so peak memory is bound by the selected columns of the selected rows,
        never by the full dataset. A ring stored as element `i` of a nested list is projected
        exactly like one stored in its own column, so the layout costs nothing either way.

        Deterministic given the same files on disk, which is what lets `train` and `evaluate`
        run as separate processes over the same row ordering.

        Returns:
            Optional[pd.DataFrame]: The prepared DataFrame, or None when nothing could be loaded.
        """
        logger.info("📂 Loading dataset...")
        files = self.loader.get_files()
        if not files:
            logger.error("❌ No data was loaded.")
            return None

        if self.schema.needs_file_paths:
            validate_files(files)

        lazy_frame = self.schema.scan(files)
        available = self.schema.canonical_columns(lazy_frame.collect_schema().names())

        columns = self.required_columns(available)
        if columns is None:
            columns = [name for name in available if name != self.label_col]
        keep = list(dict.fromkeys(
            list(columns)
            + [name for name in (ET, ETA, ROW_ID) if name in available]
            + [self.label_col]
        ))
        logger.info(f"🔎 Projecting scan down to {len(keep)} canonical column(s).")
        lazy_frame = self.schema.project(lazy_frame, keep)

        if self.et_bin is not None:
            logger.info(f"✂️ Restricting to kinematic bin {GRID.bin_label(self.et_bin, self.eta_bin)} "
                        f"({GRID.bin_description(self.et_bin, self.eta_bin)})...")
            lazy_frame = lazy_frame.filter(
                GRID.filter_expr(self.et_bin, self.eta_bin, ET, ETA)
            )

        df = lazy_frame.collect(engine="streaming").to_pandas()

        if df.empty:
            if self.et_bin is not None:
                logger.error("❌ No data remaining after kinematic binning.")
            else:
                logger.error("❌ No data was loaded.")
            return None

        if df[self.label_col].isna().any():
            raise RuntimeError(
                f"❌ {int(df[self.label_col].isna().sum())} row(s) have no label. Check "
                f"dataset.label in the config against the dataset's actual contents."
            )

        logger.info(f"   {len(df)} rows loaded.")
        return df


    def dataset_fingerprint(self, df: pd.DataFrame, Y: np.ndarray) -> Dict[str, Any]:
        """
        Describes the exact dataset the fold indices were drawn from, so `evaluate` can
        refuse to run against a different one.

        Rows are referenced positionally, because not every dataset has a stable physical key
        (the mc25 tables' (run_number, event_number, cl_idx) repeats heavily across files).
        That makes this fingerprint the guard rail: it pins the file list, the row/class counts
        and an order-sensitive row digest; a mismatch means the positional indices no longer
        point at the same rows.

        The row digest hashes the label sequence plus a per-row quantity in row order - the
        dataset's own row id when it has one, otherwise Et. Counts alone are permutation
        invariant, and labels alone can be constant within a source file, so only a per-row
        quantity makes a reordering *inside* a file - which streaming collects are in principle
        free to do - detectable.

        Args:
            df (pd.DataFrame): The loaded DataFrame.
            Y (np.ndarray): The label array.

        Returns:
            Dict[str, Any]: Fingerprint fields.
        """
        files = sorted(self.loader.get_files())
        digest = hashlib.sha1("\n".join(files).encode()).hexdigest()

        row_hasher = hashlib.sha1(np.ascontiguousarray(Y, dtype=np.int8).tobytes())
        if ROW_ID in df.columns:
            row_key = ("row_id", np.ascontiguousarray(df[ROW_ID].to_numpy(), dtype=np.uint64))
        elif ET in df.columns:
            row_key = ("et", np.ascontiguousarray(df[ET].to_numpy(), dtype=np.float32))
        else:
            row_key = ("labels_only", None)
        if row_key[1] is not None:
            row_hasher.update(row_key[1].tobytes())

        return {
            "data_path": self.data_path,
            "max_files": self.max_files,
            "n_files": len(files),
            "files_sha1": digest,
            "n_rows": int(len(df)),
            "n_positives": int((Y == 1).sum()),
            "n_negatives": int((Y == 0).sum()),
            "row_key": row_key[0],
            "rows_sha1": row_hasher.hexdigest(),
        }

    @staticmethod
    def _check_fingerprint(stored: Dict[str, Any], current: Dict[str, Any]) -> None:
        """
        Compares two dataset fingerprints and raises on any difference that would invalidate
        the stored positional fold indices.

        Args:
            stored (Dict[str, Any]): Fingerprint recorded at training time.
            current (Dict[str, Any]): Fingerprint of the data just loaded.

        Raises:
            RuntimeError: If the fingerprints disagree.
        """
        # Only keys the stored fingerprint carries are compared: a manifest that omits one
        # says nothing about it, and the remaining keys - `rows_sha1` above all - still catch
        # a dataset that moved under the fold indices.
        blocking = ["n_rows", "n_files", "files_sha1", "n_positives", "n_negatives",
                    "row_key", "rows_sha1"]
        differences = [
            f"{key}: trained on {stored.get(key)!r}, now {current.get(key)!r}"
            for key in blocking
            if key in stored and stored.get(key) != current.get(key)
        ]
        if differences:
            raise RuntimeError(
                "❌ The dataset changed since training, so the stored fold indices no longer "
                "identify the same rows:\n  - " + "\n  - ".join(differences) +
                "\n   Re-run `train` for this region before evaluating."
            )

    # ------------------------------------------------------------------ train

    def train(
        self,
        n_splits: int = 5,
        learning_rate: float = 0.001,
        target_fold: Optional[int] = None,
        seed: int = 42,
        n_inits: int = 1,
        target_init: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Trains the cross-validation folds and persists every artefact `evaluate` will need.

        No metrics, plots or tables are produced here - training's only job is to leave behind
        reproducible models. Safe to run as N parallel single-fold jobs: the split is a pure
        function of (data, seed), the shared artefacts are written atomically with identical
        content, and each fold owns its own checkpoint and sidecar.

        Args:
            n_splits (int): Number of K-Fold splits. Defaults to 5.
            learning_rate (float): Model learning rate. Defaults to 0.001.
            target_fold (Optional[int]): Train only this fold (1-indexed), for SLURM parallelism.
            seed (int): Seed for the fold partition. Defaults to 42.
            n_inits (int): Independent initialisations per fold; the best is kept. Defaults to 1.
            target_init (Optional[int]): Train only this initialisation (1-indexed), leaving the
                checkpoint under its own name and the winner unpicked. One training per
                scheduler job; `select_best_inits` finishes the job afterwards.

        Returns:
            List[Dict[str, Any]]: The fold records returned by ModelTrainer.fit_kfold.
        """
        logger.info(f"🚀 Starting training: {self.model_name} ({self.region_label()})")

        df = self.load_dataframe()
        if df is None:
            return []

        Y = self.preprocessor.get_labels(df, label_col=self.label_col)
        if Y is None:
            logger.error("❌ Labels column not found.")
            return []

        # The k-fold partition is the whole scheme: k-1 partitions train and 1 validates, which
        # is what drives early stopping and the choice between initialisations. There is no
        # separate holdout, because `evaluate` scores every fold over the full region anyway.
        logger.info(f"✂️ Stratified {n_splits}-fold partition over {len(df)} rows.")
        X_all = self.preprocessor.fit_transform(df)

        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.preprocessor.save(self.preprocessor_path)

        _atomic_write_json({
            "model": self.model_name,
            "model_class": self.model_class.__name__,
            "et_bin": self.et_bin,
            "eta_bin": self.eta_bin,
            "region": self.region_label(),
            "label_col": self.label_col,
            "seed": seed,
            "n_splits": n_splits,
            "n_inits": n_inits,
            "learning_rate": learning_rate,
            "monitor_metric": self.monitor_metric,
            "n_rows": int(len(df)),
            "dataset": self.dataset_fingerprint(df, Y),
            "schema": self.schema.describe(),
            "preprocessor": os.path.relpath(self.preprocessor_path, self.results_dir),
        }, self.manifest_path)
        logger.info(f"📄 Wrote manifest: {self.manifest_path}")

        model_kwargs = {'learning_rate': learning_rate, **self.build_model_kwargs(X_all)}
        logger.info(f"🏋️ Training {n_splits} folds (kwargs={model_kwargs}, weighted loss enabled)...")

        fold_records = self.trainer.fit_kfold(
            self.model_class, model_kwargs, X_all, Y,
            n_splits=n_splits, target_fold=target_fold, seed=seed, n_inits=n_inits,
            target_init=target_init
        )

        os.makedirs(self.history_dir, exist_ok=True)
        for record in fold_records:
            fold = record["fold"]
            # One training per job: this job owns only its own initialisation's artefacts.
            # `select_best_inits` renames the winner's to the plain fold_N names afterwards.
            stem = f"fold_{fold}" if target_init is None else f"fold_{fold}_init_{target_init}"

            history_path = os.path.join(self.history_dir, f"{stem}.csv")
            loss_callback = record["loss_callback"]
            pd.DataFrame({
                "epoch": range(max(len(loss_callback.train_loss), len(loss_callback.val_loss))),
                "train_loss": pd.Series(loss_callback.train_loss),
                "val_loss": pd.Series(loss_callback.val_loss),
            }).to_csv(history_path, index=False)

            # The rows this fold validated on rather than trained on. Evaluation scores every
            # row regardless, so these only mark which predictions are out of sample.
            val_rel = None
            if record.get("val_ids") is not None:
                val_path = os.path.join(self.artifacts_dir, f"val_indices_fold_{fold}.npy")
                np.save(val_path, np.sort(record["val_ids"]))
                val_rel = os.path.relpath(val_path, self.results_dir)

            _atomic_write_json({
                "fold": fold,
                "init": target_init,
                "checkpoint": os.path.relpath(record["checkpoint"], self.results_dir),
                "val_indices": val_rel,
                "n_inits": record.get("n_inits", 1),
                "best_init": record.get("best_init", 1),
                "pos_weight": record["pos_weight"],
                "best_score": record["best_score"],
                "monitor_metric": self.monitor_metric,
                "epochs": record["epochs"],
                "n_train": record["n_train"],
                "n_val": record["n_val"],
                "model_kwargs": {k: v for k, v in model_kwargs.items()},
                "history": os.path.relpath(history_path, self.results_dir),
            }, os.path.join(self.checkpoints_dir, f"{stem}.json"))

        logger.info(f"✅ Training complete. Artefacts under: {self.results_dir}")
        if target_init is None:
            logger.info(f"   Next: python ai/run.py evaluate {self.cli_region_args()}")
        else:
            logger.info(f"   Next, once every initialisation of this region has finished: "
                        f"python ai/run.py select {self.cli_region_args()}")
        return fold_records

    def select_best_inits(self) -> Dict[int, Dict[str, Any]]:
        """
        Picks each fold's best initialisation and promotes it to the plain `fold_N` names the
        rest of the pipeline expects.

        This is the join point of the one-training-per-job layout: every (fold, init) job wrote
        `fold_N_init_M.ckpt` plus a sidecar with its monitored score, and nothing compared them
        because the siblings were still running elsewhere. Here they are all on disk, so the
        winner per fold is renamed to `fold_N.ckpt` / `fold_N.json` / `history/fold_N.csv` and
        the losing checkpoints are deleted - otherwise n_inits would multiply the checkpoints
        on disk.

        Idempotent: a region whose folds are already settled is left alone, so re-running a
        failed scheduler step is safe.

        Returns:
            Dict[int, Dict[str, Any]]: The winning sidecar per fold.

        Raises:
            FileNotFoundError: If no per-initialisation sidecar exists for this region.
        """
        pattern = os.path.join(self.checkpoints_dir, "fold_*_init_*.json")
        per_init: Dict[int, List[Dict[str, Any]]] = {}
        for path in sorted(glob.glob(pattern)):
            with open(path) as handle:
                info = json.load(handle)
            info["_sidecar"] = path
            per_init.setdefault(int(info["fold"]), []).append(info)

        if not per_init:
            settled = self.load_fold_infos()
            if settled:
                logger.info(f"✔️ {self.region_label()}: already settled ({len(settled)} fold(s)).")
                return settled
            raise FileNotFoundError(
                f"❌ No per-initialisation sidecars in '{self.checkpoints_dir}'. Run `train` first."
            )

        better = max if self.monitor_mode == "max" else min
        winners: Dict[int, Dict[str, Any]] = {}
        for fold, candidates in sorted(per_init.items()):
            scored = [c for c in candidates if c.get("best_score") is not None]
            winner = better(scored or candidates, key=lambda c: c.get("best_score") or 0.0)
            logger.info(f"🏆 Fold {fold}: kept initialisation {winner['init']} of "
                        f"{len(candidates)} ({self.monitor_metric}={winner.get('best_score')})")

            for name, key in (("ckpt", "checkpoint"), ("csv", "history")):
                source = os.path.join(self.results_dir, winner[key])
                target = os.path.join(os.path.dirname(source), f"fold_{fold}.{name}")
                if os.path.exists(source):
                    os.replace(source, target)
                winner[key] = os.path.relpath(target, self.results_dir)

            winner.pop("init", None)
            winner["n_inits"] = len(candidates)
            _atomic_write_json({k: v for k, v in winner.items() if k != "_sidecar"},
                               os.path.join(self.checkpoints_dir, f"fold_{fold}.json"))
            winners[fold] = winner

            for loser in candidates:
                if loser is winner:
                    continue
                for key in ("checkpoint", "history"):
                    path = os.path.join(self.results_dir, loser[key])
                    if os.path.exists(path):
                        os.remove(path)
            for entry in candidates:
                os.remove(entry["_sidecar"])

        logger.info(f"✅ {self.region_label()}: {len(winners)} fold(s) settled.")
        logger.info(f"   Next: python ai/run.py evaluate {self.cli_region_args()}")
        return winners

    # --------------------------------------------------------------- evaluate

    def evaluate(
        self,
        threshold: float = 0.5,
        operating_points: Optional[Dict[str, float]] = None,
        reuse_scores: bool = False,
        make_plots: bool = True
    ) -> pd.DataFrame:
        """
        Scores every trained fold over the whole region and writes its metrics, plots and
        its slice of the cross-validation table.

        Args:
            threshold (float): Fixed decision threshold for the global metric set. Defaults to 0.5.
            operating_points (Optional[Dict[str, float]]): Working point name -> target PD.
                Defaults to {"tight": 0.90, "medium": 0.95, "loose": 0.99}.
            reuse_scores (bool): Skip inference and read `scores/fold_N.parquet` written by an
                earlier evaluation. Lets working points and plots be recut in seconds without
                touching the data or the GPU. Defaults to False.
            make_plots (bool): Whether to render the ROC/PR/confusion/loss figures. Defaults to True.

        Returns:
            pd.DataFrame: This region's long-format table (see ai.evaluation.pd_table.LONG_COLUMNS).

        Raises:
            FileNotFoundError: If the region has not been trained yet.
            RuntimeError: If the dataset no longer matches the one training split.
        """
        operating_points = operating_points or DEFAULT_OPERATING_POINTS
        manifest = self.load_manifest()
        fold_infos = self.load_fold_infos()

        if not fold_infos:
            raise FileNotFoundError(
                f"❌ No trained folds found in '{self.checkpoints_dir}'. Run `train` for this region first."
            )

        logger.info(f"📊 Evaluating {self.model_name} ({self.region_label()}): "
                    f"{len(fold_infos)} fold(s), threshold={threshold}")

        fold_scores = self._collect_scores(manifest, fold_infos, reuse_scores)

        metric_rows, operating_rows, long_rows = [], [], []
        for fold in sorted(fold_scores):
            y_true, y_prob = fold_scores[fold]
            pos_weight = fold_infos[fold].get("pos_weight")

            metrics = compute_metrics(y_true, y_prob, threshold=threshold, pos_weight=pos_weight)
            metric_rows.append({"Fold": fold, **metrics})

            points = compute_operating_points(y_true, y_prob, operating_points)
            for point in points:
                operating_rows.append({"Fold": fold, **point})
                long_rows.append({
                    "model": self.model_name,
                    "et_bin": self.et_bin,
                    "eta_bin": self.eta_bin,
                    "fold": fold,
                    "operating_point": point["Operating_Point"],
                    "target_pd": point["Target_PD"],
                    "threshold": point["Threshold"],
                    "pd": point["PD"],
                    "fa": point["FA"],
                    "sp": point["SP_Index"],
                    "auc_roc": metrics["AUC_ROC"],
                    "auc_pr": metrics["AUC_PR"],
                    "n_signal": metrics["N_Positives"],
                    "n_background": metrics["N_Negatives"],
                })
                logger.info(
                    f"   fold {fold} {point['Operating_Point']:<7} PD={point['PD']:.4f} "
                    f"(target {point['Target_PD']:.2f}) -> FA={point['FA']:.4f}, "
                    f"SP={point['SP_Index']:.4f}, threshold={point['Threshold']:.4f}"
                )

        self.summary.save_metrics(metric_rows, filename="per_fold.csv")
        self.summary.save_operating_points(operating_rows, filename="operating_points.csv")
        long_df = self.summary.save_long_table(long_rows, filename="folds_long.csv")

        if make_plots:
            self._render_plots(fold_scores, threshold, operating_points)

        logger.info(f"✅ Evaluation complete. Results under: {self.results_dir}")
        return long_df

    def _collect_scores(
        self,
        manifest: Dict[str, Any],
        fold_infos: Dict[int, Dict[str, Any]],
        reuse_scores: bool
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Produces (y_true, y_prob) for every fold, either by reloading cached score files or by
        running each fold's checkpoint over the full region (in-sample rows included).

        Args:
            manifest (Dict[str, Any]): The training manifest.
            fold_infos (Dict[int, Dict[str, Any]]): Per-fold sidecars, keyed by fold number.
            reuse_scores (bool): Read cached scores instead of re-running inference.

        Returns:
            Dict[int, Tuple[np.ndarray, np.ndarray]]: Mapping fold -> (y_true, y_prob).
        """
        if reuse_scores:
            cached = {}
            for fold in sorted(fold_infos):
                path = os.path.join(self.scores_dir, f"fold_{fold}.parquet")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"❌ --reuse-scores was given but '{path}' does not exist.")
                frame = pd.read_parquet(path)
                cached[fold] = (frame["y_true"].to_numpy(), frame["y_prob"].to_numpy())
                logger.info(f"📂 Reusing cached scores for fold {fold}: {path}")
            return cached

        df = self.load_dataframe()
        if df is None:
            raise RuntimeError("❌ No data was loaded; cannot evaluate.")

        Y = self.preprocessor.get_labels(df, label_col=self.label_col)
        self._check_fingerprint(manifest.get("dataset", {}), self.dataset_fingerprint(df, Y))

        preprocessor = type(self.preprocessor).load(os.path.join(self.results_dir, manifest["preprocessor"]))
        self.preprocessor = preprocessor

        # Every fold is scored over the WHOLE region, in-sample rows included. Train and
        # validation are separated during training - that is what drives early stopping and
        # model selection - but the reported efficiencies deliberately cover the full phase
        # space rather than only each fold's held-out partition. The `in_sample` column below
        # records which rows the fold trained on, so an out-of-sample-only cut stays available
        # to anyone who wants it.
        y_true = Y.flatten()
        X_all = preprocessor.transform(df)
        logger.info(f"🧾 Scoring the full region: {len(y_true)} rows "
                    f"({int((y_true == 1).sum())} signal, {int((y_true == 0).sum())} background).")

        # Kept alongside the scores so the table can be re-cut per kinematic region later
        # without re-running inference.
        kinematics = {
            column: df[column].to_numpy()
            for column in (ET, ETA, ROW_ID) if column in df.columns
        }

        def out_of_sample_mask(fold: int) -> Optional[np.ndarray]:
            """
            Boolean mask of the rows this fold did NOT train on, or None when unknown.

            Args:
                fold (int): Fold number.

            Returns:
                Optional[np.ndarray]: True where the row was held out from this fold's training.
            """
            rel = fold_infos[fold].get("val_indices")
            if not rel:
                return None
            path = os.path.join(self.results_dir, rel)
            if not os.path.exists(path):
                return None
            mask = np.zeros(len(y_true), dtype=bool)
            mask[np.load(path)] = True
            return mask

        os.makedirs(self.scores_dir, exist_ok=True)
        scores = {}
        for fold in sorted(fold_infos):
            checkpoint = os.path.join(self.results_dir, fold_infos[fold]["checkpoint"])
            if not os.path.exists(checkpoint):
                logger.warning(f"⚠️ Fold {fold}: checkpoint '{checkpoint}' is missing; skipping.")
                continue

            logger.info(f"🧠 Fold {fold}: loading {checkpoint}")
            # pos_weight is excluded from save_hyperparameters (it is a training-time buffer,
            # not architecture), so Lightning cannot rebuild it from the checkpoint's hparams
            # and the state_dict keys would not match. Feed it back from the fold sidecar.
            model = self.model_class.load_from_checkpoint(
                checkpoint, map_location="cpu", pos_weight=fold_infos[fold]["pos_weight"]
            )
            y_prob = self._predict(model, X_all)
            scores[fold] = (y_true, y_prob)

            held_out = out_of_sample_mask(fold)
            columns = {"y_true": y_true, "y_prob": y_prob, **kinematics}
            if held_out is not None:
                columns["in_sample"] = ~held_out
                logger.info(f"   fold {fold}: {int(held_out.sum())} of {len(y_true)} rows were out of sample")

            path = os.path.join(self.scores_dir, f"fold_{fold}.parquet")
            pd.DataFrame(columns).to_parquet(path, index=False)
            logger.info(f"💾 Saved scores for fold {fold} to: {path}")

        return scores

    def _predict(self, model: pl.LightningModule, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        """
        Runs batched inference and returns post-sigmoid probabilities.

        Batched rather than in one shot because a region can be tens of millions of rows,
        which would not fit in memory as a single forward pass.

        Args:
            model (pl.LightningModule): Trained model, already restored from a checkpoint.
            X (np.ndarray): Preprocessed features.
            batch_size (int): Inference batch size. Defaults to 8192.

        Returns:
            np.ndarray: Probabilities, shape (N,).
        """
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                chunk = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32)
                outputs.append(torch.sigmoid(model(chunk)).cpu().numpy().flatten())
        return np.concatenate(outputs) if outputs else np.empty(0)

    def _render_plots(
        self,
        fold_scores: Dict[int, Tuple[np.ndarray, np.ndarray]],
        threshold: float,
        operating_points: Dict[str, float]
    ) -> None:
        """
        Renders the per-fold figures plus the fold-overlay ROC for this region.

        Args:
            fold_scores (Dict[int, Tuple[np.ndarray, np.ndarray]]): Mapping fold -> (y_true, y_prob).
            threshold (float): Decision threshold used for the confusion matrix.
            operating_points (Dict[str, float]): Working point name -> target PD.
        """
        logger.info(f"🖼️ Rendering plots into {self.monitor.output_dir}...")
        for fold in sorted(fold_scores):
            y_true, y_prob = fold_scores[fold]
            points = compute_operating_points(y_true, y_prob, operating_points)

            self.monitor.plot_roc_curve(y_true, y_prob, filename=f"roc_curve_fold_{fold}.pdf", operating_points=points)
            self.monitor.plot_pr_curve(y_true, y_prob, filename=f"pr_curve_fold_{fold}.pdf")
            self.monitor.plot_confusion_matrix(
                y_true, (y_prob >= threshold).astype(int),
                filename=f"confusion_matrix_fold_{fold}.pdf"
            )

            history_path = os.path.join(self.history_dir, f"fold_{fold}.csv")
            if os.path.exists(history_path):
                history = pd.read_csv(history_path)
                self.monitor.plot_loss(
                    history["train_loss"].dropna().tolist(),
                    history["val_loss"].dropna().tolist(),
                    filename=f"loss_curve_fold_{fold}.pdf"
                )

        if len(fold_scores) > 1:
            self.monitor.plot_roc_folds(
                fold_scores,
                filename="roc_folds.pdf",
                title=f"ROC Curve — {self.model_name} ({self.region_label()})"
            )

    # ----------------------------------------------------------------- shared

    def load_manifest(self) -> Dict[str, Any]:
        """
        Reads the training manifest for this region.

        Returns:
            Dict[str, Any]: The manifest contents.

        Raises:
            FileNotFoundError: If this region has not been trained.
        """
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(
                f"❌ No manifest at '{self.manifest_path}'. Run `train` for this region first."
            )
        with open(self.manifest_path) as handle:
            return json.load(handle)

    def load_fold_infos(self) -> Dict[int, Dict[str, Any]]:
        """
        Reads the per-fold sidecars written by train().

        Each fold owns its own file rather than sharing one manifest entry, so N parallel
        single-fold SLURM jobs never contend over the same file.

        Returns:
            Dict[int, Dict[str, Any]]: Mapping fold number -> sidecar contents.
        """
        infos = {}
        if not os.path.isdir(self.checkpoints_dir):
            return infos
        for name in sorted(os.listdir(self.checkpoints_dir)):
            if not name.startswith("fold_") or not name.endswith(".json"):
                continue
            with open(os.path.join(self.checkpoints_dir, name)) as handle:
                info = json.load(handle)
            infos[int(info["fold"])] = info
        return infos

    def region_label(self) -> str:
        """
        Human-readable label of the kinematic region this pipeline instance covers.

        Returns:
            str: e.g. 'et2_eta0' or 'full phase space'.
        """
        if self.et_bin is None:
            return "full phase space"
        return GRID.bin_label(self.et_bin, self.eta_bin)

    def cli_region_args(self) -> str:
        """
        The `--et-bin/--eta-bin` fragment that reproduces this region on the command line.

        Returns:
            str: e.g. '--et-bin 2 --eta-bin 0', or '' for the ungridded case.
        """
        if self.et_bin is None:
            return ""
        return f"--et-bin {self.et_bin} --eta-bin {self.eta_bin}"
