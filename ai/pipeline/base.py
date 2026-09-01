"""
Shared training/evaluation pipeline.

The pipeline is split into two independent phases so that they can be run as separate
commands, on different machines and at different times:

* `train`    - loads data, fits the preprocessor on the training rows only, runs the K-Fold
               cross-validation and persists everything an evaluation needs: one checkpoint
               per fold under a fixed name, the fitted preprocessor, the holdout indices and
               a manifest describing the dataset the split was derived from.
* `evaluate` - reloads those artefacts, re-runs inference over the holdout for every fold,
               persists the raw scores and produces the per-fold metrics, the plots and the
               region's slice of the cross-validation table.

Keeping them apart means re-scoring, re-plotting or re-cutting the working points never
requires retraining, and the numbers that end up in the table always come from a checkpoint
that is on disk and addressable.
"""

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
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Ensure root directory is in path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from ai.loader.loader import DataLoader
from ai.label.label_generator import LabelGenerator
from ai.trainer.trainer import ModelTrainer
from ai.evaluation.monitor import ModelMonitor
from ai.evaluation.summary import (
    DEFAULT_OPERATING_POINTS,
    ModelSummary,
    compute_metrics,
    compute_operating_points,
)
from ai.binning.kinematics import bin_filter_expr, bin_label, bin_description

MANIFEST_VERSION = 1


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
    holdout split, cross-validation, artefact persistence, scoring and reporting - is common.
    """

    #: LightningModule subclass this pipeline trains.
    model_class: Type[pl.LightningModule]

    #: Metric monitored by EarlyStopping/ModelCheckpoint, and its improvement direction.
    monitor_metric: str = "val_sp"
    monitor_mode: str = "max"

    def __init__(
        self,
        data_path: Optional[str] = None,
        max_files: Optional[int] = None,
        label_col: str = 'label',
        model_name: str = "Model",
        max_epochs: int = 20,
        batch_size: int = 32,
        patience: int = 5,
        num_workers: int = 0,
        accelerator: str = "auto",
        devices: Union[int, str, List[int]] = "auto",
        et_bin: Optional[int] = None,
        eta_bin: Optional[int] = None
    ) -> None:
        """
        Initializes the pipeline and resolves the results directory for this kinematic region.

        Args:
            data_path (Optional[str]): Data folder or pattern path.
            max_files (Optional[int]): Maximum number of files to process per folder.
            label_col (str): Column name containing labels. Defaults to 'label'.
            model_name (str): Model name for logging and results folder.
            max_epochs (int): Maximum training epochs. Defaults to 20.
            batch_size (int): Training batch size. Defaults to 32.
            patience (int): Early stopping patience. Defaults to 5.
            num_workers (int): Parallel worker subprocesses. Defaults to 0.
            accelerator (str): PyTorch Lightning accelerator ('auto', 'cpu', 'cuda'). Defaults to 'auto'.
            devices (Union[int, str, List[int]]): Devices specification. Defaults to 'auto'.
            et_bin (Optional[int]): Et bin index (0-4, see ai.binning.kinematics). Trains on the
                whole dataset when None (together with eta_bin) or on only that kinematic slice
                when both are set - the standard 5x5=25-network Ringer scheme. Defaults to None.
            eta_bin (Optional[int]): |eta| bin index (0-4). Defaults to None.

        Raises:
            ValueError: If exactly one of et_bin/eta_bin is set.
        """
        if (et_bin is None) != (eta_bin is None):
            raise ValueError("❌ et_bin and eta_bin must be set together (or both left as None).")

        self.model_name = model_name
        self.label_col = label_col
        self.data_path = data_path
        self.max_files = max_files
        self.et_bin = et_bin
        self.eta_bin = eta_bin

        self.results_dir = os.path.join("results", self.model_name)
        if et_bin is not None:
            self.results_dir = os.path.join(self.results_dir, bin_label(et_bin, eta_bin))
            logger.info(f"🎯 Kinematic bin selected: {bin_description(et_bin, eta_bin)}")

        self.artifacts_dir = os.path.join(self.results_dir, "artifacts")
        self.checkpoints_dir = os.path.join(self.results_dir, "checkpoints")
        self.history_dir = os.path.join(self.results_dir, "history")
        self.scores_dir = os.path.join(self.results_dir, "scores")
        self.manifest_path = os.path.join(self.artifacts_dir, "manifest.json")
        self.preprocessor_path = os.path.join(self.artifacts_dir, "preprocessor.joblib")
        self.test_indices_path = os.path.join(self.artifacts_dir, "test_indices.npy")

        self.loader = DataLoader(data_path=data_path, max_files=max_files)
        self.preprocessor = self.build_preprocessor()

        self.trainer = ModelTrainer(
            max_epochs=max_epochs,
            batch_size=batch_size,
            patience=patience,
            num_workers=num_workers,
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
        Builds the preprocessor for this architecture. Must honour the
        fit/transform/fit_transform/save/load/get_labels contract.

        Returns:
            Any: A preprocessor instance.
        """
        raise NotImplementedError

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
        return None

    # ------------------------------------------------------------------ data

    def load_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Loads the dataset, attaches labels and applies the kinematic bin cut.

        Runs as a single lazy polars query - column projection, per-file labelling and the
        bin filter all happen inside the streaming parquet scan - so peak memory is bound by
        the selected columns of the selected rows, never by the full dataset.

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

        LabelGenerator.validate_files(files)

        lazy_frame = self.loader.scan(files)
        available = [name for name in lazy_frame.collect_schema().names() if name != 'file_path']

        logger.info("🏷️ Generating labels...")
        lazy_frame = lazy_frame.with_columns(
            LabelGenerator.label_expr(file_path_col='file_path', label_col=self.label_col)
        )

        columns = self.required_columns(available)
        if columns is not None:
            keep = list(dict.fromkeys(
                list(columns)
                + [col for col in ('cl_et', 'cl_eta') if col in available]
                + [self.label_col]
            ))
            logger.info(f"🔎 Projecting scan down to {len(keep)} of {len(available)} columns.")
            lazy_frame = lazy_frame.select(keep)
        else:
            lazy_frame = lazy_frame.drop('file_path')

        if self.et_bin is not None:
            logger.info(f"✂️ Restricting to kinematic bin {bin_label(self.et_bin, self.eta_bin)} "
                        f"({bin_description(self.et_bin, self.eta_bin)})...")
            lazy_frame = lazy_frame.filter(bin_filter_expr(self.et_bin, self.eta_bin))

        df = lazy_frame.collect(engine="streaming").to_pandas()

        if df.empty:
            if self.et_bin is not None:
                logger.error("❌ No data remaining after kinematic binning.")
            else:
                logger.error("❌ No data was loaded.")
            return None

        logger.info(f"   {len(df)} rows loaded.")
        return df

    def dataset_fingerprint(self, df: pd.DataFrame, Y: np.ndarray) -> Dict[str, Any]:
        """
        Describes the exact dataset the holdout indices were drawn from, so `evaluate` can
        refuse to run against a different one.

        The parquet rows carry no stable physical key - (run_number, event_number, cl_idx)
        repeats heavily across files - so the holdout can only be referenced positionally.
        That makes this fingerprint the guard rail: it pins the file list, the row/class
        counts and an order-sensitive row digest; a mismatch means the positional indices no
        longer point at the same rows.

        The row digest hashes the label sequence and the cl_et sequence in row order. Counts
        alone are permutation-invariant, and labels alone are constant within each source file
        (they derive from the file path), so only a per-row quantity like cl_et makes a
        reordering *inside* a file - which streaming collects are in principle free to do -
        detectable.

        Args:
            df (pd.DataFrame): The loaded DataFrame.
            Y (np.ndarray): The label array.

        Returns:
            Dict[str, Any]: Fingerprint fields.
        """
        files = sorted(self.loader.get_files())
        digest = hashlib.sha1("\n".join(files).encode()).hexdigest()

        row_hasher = hashlib.sha1(np.ascontiguousarray(Y, dtype=np.int8).tobytes())
        if 'cl_et' in df.columns:
            row_hasher.update(np.ascontiguousarray(df['cl_et'].to_numpy(), dtype=np.float32).tobytes())

        return {
            "data_path": self.data_path,
            "max_files": self.max_files,
            "n_files": len(files),
            "files_sha1": digest,
            "n_rows": int(len(df)),
            "n_positives": int((Y == 1).sum()),
            "n_negatives": int((Y == 0).sum()),
            "rows_sha1": row_hasher.hexdigest(),
        }

    @staticmethod
    def _check_fingerprint(stored: Dict[str, Any], current: Dict[str, Any]) -> None:
        """
        Compares two dataset fingerprints and raises on any difference that would invalidate
        the stored positional holdout indices.

        Args:
            stored (Dict[str, Any]): Fingerprint recorded at training time.
            current (Dict[str, Any]): Fingerprint of the data just loaded.

        Raises:
            RuntimeError: If the fingerprints disagree.
        """
        blocking = ["n_rows", "n_files", "files_sha1", "n_positives", "n_negatives", "rows_sha1"]
        differences = [
            f"{key}: trained on {stored.get(key)!r}, now {current.get(key)!r}"
            for key in blocking
            if stored.get(key) != current.get(key)
        ]
        if differences:
            raise RuntimeError(
                "❌ The dataset changed since training, so the stored holdout indices no longer "
                "identify the same rows:\n  - " + "\n  - ".join(differences) +
                "\n   Re-run `train` for this region before evaluating."
            )

    # ------------------------------------------------------------------ train

    def train(
        self,
        n_splits: int = 5,
        test_size: float = 0.15,
        learning_rate: float = 0.001,
        target_fold: Optional[int] = None,
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Trains the cross-validation folds and persists every artefact `evaluate` will need.

        No metrics, plots or tables are produced here - training's only job is to leave behind
        reproducible models. Safe to run as N parallel single-fold jobs: the split is a pure
        function of (data, seed), the shared artefacts are written atomically with identical
        content, and each fold owns its own checkpoint and sidecar.

        Args:
            n_splits (int): Number of K-Fold splits. Defaults to 5.
            test_size (float): Holdout test dataset ratio. Defaults to 0.15.
            learning_rate (float): Model learning rate. Defaults to 0.001.
            target_fold (Optional[int]): Train only this fold (1-indexed), for SLURM parallelism.
            seed (int): Seed for the holdout split and the fold partition. Defaults to 42.

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

        logger.info(f"✂️ Splitting {test_size * 100:g}% of the data into a stratified holdout...")
        train_idx, test_idx = train_test_split(
            np.arange(len(df)), test_size=test_size, random_state=seed, shuffle=True, stratify=Y
        )
        train_idx = np.sort(train_idx)
        test_idx = np.sort(test_idx)

        logger.info("⚙️ Fitting preprocessor on the training rows only...")
        X_train = self.preprocessor.fit_transform(df.iloc[train_idx])
        Y_train = Y[train_idx]

        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.preprocessor.save(self.preprocessor_path)
        np.save(self.test_indices_path, test_idx)
        logger.info(f"💾 Saved {len(test_idx)} holdout indices to: {self.test_indices_path}")

        _atomic_write_json({
            "manifest_version": MANIFEST_VERSION,
            "model": self.model_name,
            "model_class": self.model_class.__name__,
            "et_bin": self.et_bin,
            "eta_bin": self.eta_bin,
            "region": self.region_label(),
            "label_col": self.label_col,
            "seed": seed,
            "test_size": test_size,
            "n_splits": n_splits,
            "learning_rate": learning_rate,
            "monitor_metric": self.monitor_metric,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "dataset": self.dataset_fingerprint(df, Y),
            "preprocessor": os.path.relpath(self.preprocessor_path, self.results_dir),
            "test_indices": os.path.relpath(self.test_indices_path, self.results_dir),
        }, self.manifest_path)
        logger.info(f"📄 Wrote manifest: {self.manifest_path}")

        model_kwargs = {'learning_rate': learning_rate, **self.build_model_kwargs(X_train)}
        logger.info(f"🏋️ Training {n_splits} folds (kwargs={model_kwargs}, weighted loss enabled)...")

        fold_records = self.trainer.fit_kfold(
            self.model_class, model_kwargs, X_train, Y_train,
            n_splits=n_splits, target_fold=target_fold, seed=seed
        )

        os.makedirs(self.history_dir, exist_ok=True)
        for record in fold_records:
            fold = record["fold"]

            history_path = os.path.join(self.history_dir, f"fold_{fold}.csv")
            loss_callback = record["loss_callback"]
            pd.DataFrame({
                "epoch": range(max(len(loss_callback.train_loss), len(loss_callback.val_loss))),
                "train_loss": pd.Series(loss_callback.train_loss),
                "val_loss": pd.Series(loss_callback.val_loss),
            }).to_csv(history_path, index=False)

            _atomic_write_json({
                "fold": fold,
                "checkpoint": os.path.relpath(record["checkpoint"], self.results_dir),
                "pos_weight": record["pos_weight"],
                "best_score": record["best_score"],
                "monitor_metric": self.monitor_metric,
                "epochs": record["epochs"],
                "n_train": record["n_train"],
                "n_val": record["n_val"],
                "model_kwargs": {k: v for k, v in model_kwargs.items()},
                "history": os.path.relpath(history_path, self.results_dir),
            }, os.path.join(self.checkpoints_dir, f"fold_{fold}.json"))

        logger.info(f"✅ Training complete. Artefacts under: {self.results_dir}")
        logger.info(f"   Next: python ai/run.py evaluate {self.cli_region_args()}")
        return fold_records

    # --------------------------------------------------------------- evaluate

    def evaluate(
        self,
        threshold: float = 0.5,
        operating_points: Optional[Dict[str, float]] = None,
        reuse_scores: bool = False,
        make_plots: bool = True
    ) -> pd.DataFrame:
        """
        Scores every trained fold on the holdout and writes this region's metrics, plots and
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
            pd.DataFrame: This region's long-format table (see ai.evaluation.tabelao.LONG_COLUMNS).

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
        running inference from each fold's checkpoint over the holdout.

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

        test_idx = np.load(os.path.join(self.results_dir, manifest["test_indices"]))
        df_test = df.iloc[test_idx]
        y_true = Y[test_idx].flatten()

        preprocessor = type(self.preprocessor).load(os.path.join(self.results_dir, manifest["preprocessor"]))
        self.preprocessor = preprocessor
        X_test = preprocessor.transform(df_test)
        logger.info(f"🧾 Holdout: {len(y_true)} rows "
                    f"({int((y_true == 1).sum())} signal, {int((y_true == 0).sum())} background).")

        # Kept alongside the scores so the table can be re-cut per kinematic region later
        # without re-running inference.
        kinematics = {
            column: df_test[column].to_numpy()
            for column in ("cl_et", "cl_eta") if column in df_test.columns
        }

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
            y_prob = self._predict(model, X_test)
            scores[fold] = (y_true, y_prob)

            path = os.path.join(self.scores_dir, f"fold_{fold}.parquet")
            pd.DataFrame({"y_true": y_true, "y_prob": y_prob, **kinematics}).to_parquet(path, index=False)
            logger.info(f"💾 Saved scores for fold {fold} to: {path}")

        return scores

    def _predict(self, model: pl.LightningModule, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        """
        Runs batched inference and returns post-sigmoid probabilities.

        Batched rather than in one shot because the holdout can be tens of millions of rows,
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
        return bin_label(self.et_bin, self.eta_bin)

    def cli_region_args(self) -> str:
        """
        The `--et-bin/--eta-bin` fragment that reproduces this region on the command line.

        Returns:
            str: e.g. '--et-bin 2 --eta-bin 0', or '' for the ungridded case.
        """
        if self.et_bin is None:
            return ""
        return f"--et-bin {self.et_bin} --eta-bin {self.eta_bin}"
