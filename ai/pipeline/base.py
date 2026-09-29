"""
Shared training/evaluation pipeline.

The pipeline is split into two independent phases so that they can be run as separate
commands, on different machines and at different times:

* `train`    - loads data, fits the preprocessor on the training rows only, runs the K-Fold
               cross-validation and persists everything an evaluation needs: one checkpoint
               per fold under a fixed name, the preprocessor, and each fold's validation
               indices.
* `evaluate` - reloads those artefacts, re-runs inference over the whole region for every fold,
               persists the raw scores and produces the per-fold metrics, the plots and the
               region's slice of the cross-validation table.

Keeping them apart means re-scoring, re-plotting or re-cutting the working points never
requires retraining, and the numbers that end up in the table always come from a checkpoint
that is on disk and addressable.
"""

import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from ai.trainer.trainer import MONITOR, ModelTrainer
from ai.evaluation.monitor import ModelMonitor
from ai.evaluation.summary import DEFAULT_OPERATING_POINTS, compute_metrics, compute_operating_points
from ai.binning import kinematics
from ai.label.label_generator import validate_files
from ai.preprocess.base import ET, ETA, LABEL, ROW_ID, DatasetSchema

logger = logging.getLogger(__name__)


def _atomic_write_json(payload: Dict[str, Any], filepath: str) -> str:
    """
    Writes JSON via a temporary file plus os.replace.

    Under SLURM these sidecars are written by many processes at once and read by the step that
    follows; a plain open()/write() lets a reader observe a half-written file. os.replace is
    atomic on POSIX, so a reader always sees either the old or the new complete file.
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
    cross-validation, artefact persistence, scoring and reporting - is common. A preprocessor
    that needs constructor arguments is declared as `functools.partial(PreprocessX, ...)`.
    """

    #: LightningModule subclass this pipeline trains (a BaseBinaryClassifier subclass).
    model_class: Type[pl.LightningModule]

    #: Preprocessor class for this architecture (a BasePreprocessor subclass).
    preprocessor_class: Type[Any]

    #: Registry name, set by @register_pipeline. Also the results/<NAME>/ directory.
    model_name: str = "Model"

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
        Resolves the results directory for this kinematic region.

        Nothing here knows the dataset's column names: the schema (the mc25 layout by default)
        translates whatever is on disk into the canonical `label` / `et` / `eta` / `ring_i`
        vocabulary every stage below works in.

        Artefacts go to `<results_root>/<model>/<region>`. Give each dataset its own root: the
        region directories are named after bin *indices*, so two datasets sharing a root would
        silently overwrite each other's identically-named regions. With et_bin and eta_bin both
        None the pipeline covers the whole dataset; with both set, only that kinematic slice -
        the Ringer one-network-per-region scheme. Raises ValueError if exactly one is set or
        the region is outside the grid.
        """
        if (et_bin is None) != (eta_bin is None):
            raise ValueError("❌ et_bin and eta_bin must be set together (or both left as None).")

        self.schema = schema or DatasetSchema()
        self.et_bin = et_bin
        self.eta_bin = eta_bin

        self.results_dir = os.path.join(results_root, self.model_name)
        if et_bin is not None:
            kinematics.validate(et_bin, eta_bin)
            self.results_dir = os.path.join(self.results_dir, kinematics.bin_label(et_bin, eta_bin))
            logger.info(f"🎯 Kinematic bin selected: {kinematics.bin_description(et_bin, eta_bin)}")

        self.artifacts_dir = os.path.join(self.results_dir, "artifacts")
        self.checkpoints_dir = os.path.join(self.results_dir, "checkpoints")
        self.history_dir = os.path.join(self.results_dir, "history")
        self.scores_dir = os.path.join(self.results_dir, "scores")
        self.preprocessor_path = os.path.join(self.artifacts_dir, "preprocessor.joblib")

        self.preprocessor = self.preprocessor_class()

        self.trainer = ModelTrainer(
            max_epochs=max_epochs,
            batch_size=batch_size,
            patience=patience,
            log_dir=os.path.join(self.results_dir, "lightning_logs"),
            checkpoint_dir=self.checkpoints_dir,
            accelerator=accelerator,
            devices=devices
        )

        self.monitor = ModelMonitor(output_dir=os.path.join(self.results_dir, "plots"))

    # ------------------------------------------------------------------ hooks

    def build_model_kwargs(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Constructor kwargs for the model, given the (already preprocessed) training feature
        array - this is where an architecture picks up e.g. its input dimension.
        """
        return {}

    # ------------------------------------------------------------------ data

    def load_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Loads the dataset in the canonical column vocabulary and applies the kinematic cut, or
        returns None when nothing could be loaded.

        Runs as a single lazy polars query - the join with any side table, the column
        projection, the label derivation and the region filter all happen inside the streaming
        parquet scan - so peak memory is bound by the selected columns of the selected rows,
        never by the full dataset. A ring stored as element `i` of a nested list is projected
        exactly like one stored in its own column, so the layout costs nothing either way.

        Deterministic given the same files on disk, which is what lets `train` and `evaluate`
        run as separate processes over the same row ordering.
        """
        logger.info("📂 Loading dataset...")
        files = self.schema.files()
        if not files:
            logger.error("❌ No data was loaded.")
            return None

        if self.schema.needs_file_paths:
            validate_files(files)

        lazy_frame = self.schema.scan(files)
        available = self.schema.canonical_columns(lazy_frame.collect_schema().names())

        columns = self.preprocessor.required_columns(available)
        if columns is None:
            columns = [name for name in available if name != LABEL]
        keep = list(dict.fromkeys(
            list(columns)
            + [name for name in (ET, ETA, ROW_ID) if name in available]
            + [LABEL]
        ))
        logger.info(f"🔎 Projecting scan down to {len(keep)} canonical column(s).")
        lazy_frame = self.schema.project(lazy_frame, keep)

        if self.et_bin is not None:
            logger.info(f"✂️ Restricting to kinematic bin {kinematics.bin_label(self.et_bin, self.eta_bin)} "
                        f"({kinematics.bin_description(self.et_bin, self.eta_bin)})...")
            lazy_frame = lazy_frame.filter(kinematics.filter_expr(self.et_bin, self.eta_bin, ET, ETA))

        df = lazy_frame.collect(engine="streaming").to_pandas()

        if df.empty:
            if self.et_bin is not None:
                logger.error("❌ No data remaining after kinematic binning.")
            else:
                logger.error("❌ No data was loaded.")
            return None

        if df[LABEL].isna().any():
            raise RuntimeError(
                f"❌ {int(df[LABEL].isna().sum())} row(s) have no label. Check "
                f"dataset.label in the config against the dataset's actual contents."
            )

        logger.info(f"   {len(df)} rows loaded.")
        return df

    # ------------------------------------------------------------------ train

    def train(
        self,
        n_splits: int = 5,
        learning_rate: float = 0.001,
        target_fold: Optional[int] = None,
        seed: int = 42,
        n_inits: int = 1,
        target_init: Optional[int] = None
    ) -> None:
        """
        Trains the cross-validation folds and persists every artefact `evaluate` will need.

        No metrics, plots or tables are produced here, and nothing is picked or discarded:
        training's only job is to leave behind reproducible models, every (fold, init) pair
        under its own `fold_N_init_M` name. Choosing each fold's best initialisation is up to
        `evaluate`. Safe to run as parallel single-(fold, init) jobs: the split is a pure
        function of (data, seed), the shared artefacts are written with identical content, and
        each (fold, init) owns its own checkpoint and sidecar.
        """
        logger.info(f"🚀 Starting training: {self.model_name} ({self.region_label()})")

        df = self.load_dataframe()
        if df is None:
            return

        Y = df[LABEL].to_numpy(np.float32)

        # The k-fold partition is the whole scheme: k-1 partitions train and 1 validates, which
        # is what drives early stopping and the choice between initialisations. There is no
        # separate holdout, because `evaluate` scores every fold over the full region anyway.
        logger.info(f"✂️ Stratified {n_splits}-fold partition over {len(df)} rows.")
        X_all = self.preprocessor.fit_transform(df)

        os.makedirs(self.artifacts_dir, exist_ok=True)
        joblib.dump(self.preprocessor, self.preprocessor_path)
        logger.info(f"💾 Saved preprocessor to: {self.preprocessor_path}")

        model_kwargs = {'learning_rate': learning_rate, **self.build_model_kwargs(X_all)}
        logger.info(f"🏋️ Training {n_splits} folds (kwargs={model_kwargs}, weighted loss enabled)...")

        records = self.trainer.fit_kfold(
            self.model_class, model_kwargs, X_all, Y,
            n_splits=n_splits, target_fold=target_fold, seed=seed, n_inits=n_inits,
            target_init=target_init
        )

        os.makedirs(self.history_dir, exist_ok=True)
        for record in records:
            fold = record["fold"]
            stem = f"fold_{fold}_init_{record['init']}"

            history_path = os.path.join(self.history_dir, f"{stem}.csv")
            loss_callback = record["loss_callback"]
            pd.DataFrame({
                "epoch": range(max(len(loss_callback.train_loss), len(loss_callback.val_loss))),
                "train_loss": pd.Series(loss_callback.train_loss),
                "val_loss": pd.Series(loss_callback.val_loss),
            }).to_csv(history_path, index=False)

            # The rows this fold validated on rather than trained on. Evaluation scores every
            # row regardless, so these only mark which predictions are out of sample.
            val_path = os.path.join(self.artifacts_dir, f"val_indices_fold_{fold}.npy")
            np.save(val_path, np.sort(record["val_ids"]))

            _atomic_write_json({
                "fold": fold,
                "init": record["init"],
                "checkpoint": os.path.relpath(record["checkpoint"], self.results_dir),
                "val_indices": os.path.relpath(val_path, self.results_dir),
                "pos_weight": record["pos_weight"],
                "best_score": record["best_score"],
                "monitor_metric": MONITOR,
                "epochs": record["epochs"],
                "n_train": record["n_train"],
                "n_val": record["n_val"],
                "model_kwargs": model_kwargs,
                "history": os.path.relpath(history_path, self.results_dir),
            }, os.path.join(self.checkpoints_dir, f"{stem}.json"))

        logger.info(f"✅ Training complete. Artefacts under: {self.results_dir}")
        logger.info(f"   Next, once every training of this region has finished: "
                    f"python ai/run.py evaluate {self.cli_region_args()}")

    # --------------------------------------------------------------- evaluate

    def evaluate(
        self,
        operating_points: Optional[Dict[str, float]] = None,
        reuse_scores: bool = False,
        make_plots: bool = True
    ) -> None:
        """
        Scores every trained fold over the whole region and writes its metrics, plots and its
        slice of the cross-validation table (see ai.evaluation.pd_table.LONG_COLUMNS).

        `operating_points` maps working point name -> target PD (tight/medium/loose at
        90/95/99% by default). `reuse_scores` skips inference and reads the
        `scores/fold_N.parquet` of an earlier evaluation, so working points and plots can be
        recut in seconds without touching the data or the GPU. Raises FileNotFoundError if the
        region has not been trained yet.
        """
        operating_points = operating_points or DEFAULT_OPERATING_POINTS
        fold_infos = self.load_fold_infos()

        if not fold_infos:
            raise FileNotFoundError(
                f"❌ No trained folds found in '{self.checkpoints_dir}'. Run `train` for this region first."
            )

        logger.info(f"📊 Evaluating {self.model_name} ({self.region_label()}): "
                    f"{len(fold_infos)} fold(s)")

        fold_scores = self._collect_scores(fold_infos, reuse_scores)

        long_rows = []
        for fold in sorted(fold_scores):
            y_true, y_prob = fold_scores[fold]
            metrics = compute_metrics(y_true, y_prob)

            for point in compute_operating_points(y_true, y_prob, operating_points):
                long_rows.append({"model": self.model_name, "et_bin": self.et_bin,
                                  "eta_bin": self.eta_bin, "fold": fold, **point, **metrics})
                logger.info(
                    f"   fold {fold} {point['operating_point']:<7} PD={point['pd']:.4f} "
                    f"(target {point['target_pd']:.2f}) -> FA={point['fa']:.4f}, "
                    f"SP={point['sp']:.4f}, threshold={point['threshold']:.4f}"
                )

        metrics_dir = os.path.join(self.results_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        long_path = os.path.join(metrics_dir, "folds_long.csv")
        pd.DataFrame(long_rows).to_csv(long_path, index=False)
        logger.info(f"📝 Saved long-format fold table ({len(long_rows)} rows) to: {long_path}")

        if make_plots:
            self._render_plots(fold_scores, operating_points, fold_infos)

        logger.info(f"✅ Evaluation complete. Results under: {self.results_dir}")

    def _collect_scores(
        self,
        fold_infos: Dict[int, Dict[str, Any]],
        reuse_scores: bool
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        (y_true, y_prob) for every fold, either by reloading cached score files or by running
        each fold's checkpoint over the full region (in-sample rows included).
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

        self.preprocessor = joblib.load(self.preprocessor_path)
        logger.info(f"📂 Loaded preprocessor from: {self.preprocessor_path}")

        # Every fold is scored over the WHOLE region, in-sample rows included. Train and
        # validation are separated during training - that is what drives early stopping and
        # model selection - but the reported efficiencies deliberately cover the full phase
        # space rather than only each fold's held-out partition. The `in_sample` column below
        # records which rows the fold trained on, so an out-of-sample-only cut stays available
        # to anyone who wants it.
        y_true = df[LABEL].to_numpy(np.float32)
        X_all = self.preprocessor.transform(df)
        logger.info(f"🧾 Scoring the full region: {len(y_true)} rows "
                    f"({int((y_true == 1).sum())} signal, {int((y_true == 0).sum())} background).")

        # Kept alongside the scores so the table can be re-cut per kinematic region later
        # without re-running inference.
        kinematic_columns = {
            column: df[column].to_numpy()
            for column in (ET, ETA, ROW_ID) if column in df.columns
        }

        def out_of_sample_mask(fold: int) -> Optional[np.ndarray]:
            """True where the row was held out from this fold's training; None when unknown."""
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

            logger.info(f"🧠 Fold {fold}: best initialisation is {checkpoint} "
                        f"({MONITOR}={fold_infos[fold].get('best_score')})")
            # pos_weight is excluded from save_hyperparameters (it is a training-time buffer,
            # not architecture), so Lightning cannot rebuild it from the checkpoint's hparams
            # and the state_dict keys would not match. Feed it back from the fold sidecar.
            model = self.model_class.load_from_checkpoint(
                checkpoint, map_location="cpu", pos_weight=fold_infos[fold]["pos_weight"]
            )
            y_prob = self._predict(model, X_all)
            scores[fold] = (y_true, y_prob)

            held_out = out_of_sample_mask(fold)
            columns = {"y_true": y_true, "y_prob": y_prob, **kinematic_columns}
            if held_out is not None:
                columns["in_sample"] = ~held_out
                logger.info(f"   fold {fold}: {int(held_out.sum())} of {len(y_true)} rows were out of sample")

            path = os.path.join(self.scores_dir, f"fold_{fold}.parquet")
            pd.DataFrame(columns).to_parquet(path, index=False)
            logger.info(f"💾 Saved scores for fold {fold} to: {path}")

        return scores

    def _predict(self, model: pl.LightningModule, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        """
        Post-sigmoid probabilities, shape (N,). Batched rather than in one shot because a
        region can be tens of millions of rows, which would not fit in memory as a single
        forward pass.
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
        operating_points: Dict[str, float],
        fold_infos: Dict[int, Dict[str, Any]]
    ) -> None:
        """Renders the per-fold figures plus the fold-overlay ROC for this region."""
        logger.info(f"🖼️ Rendering plots into {self.monitor.output_dir}...")
        for fold in sorted(fold_scores):
            y_true, y_prob = fold_scores[fold]
            points = compute_operating_points(y_true, y_prob, operating_points)
            # The confusion matrix needs one cut; the tightest working point is the one the
            # trigger would actually run at, so it is the cut worth picturing.
            cut = min(points, key=lambda point: point["target_pd"])["threshold"]

            self.monitor.plot_roc_curve(y_true, y_prob, filename=f"roc_curve_fold_{fold}.pdf", operating_points=points)
            self.monitor.plot_pr_curve(y_true, y_prob, filename=f"pr_curve_fold_{fold}.pdf")
            self.monitor.plot_confusion_matrix(
                y_true, (y_prob >= cut).astype(int),
                filename=f"confusion_matrix_fold_{fold}.pdf"
            )

            history_path = os.path.join(self.results_dir, fold_infos[fold].get("history", ""))
            if os.path.isfile(history_path):
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

    def load_fold_infos(self) -> Dict[int, Dict[str, Any]]:
        """
        The sidecar of each fold's best initialisation - the highest monitored score - keyed by
        fold number. Training keeps every (fold, init) model; this is where one is picked per
        fold. A single `fold_N.json` left by older versions counts as that fold's only init.
        """
        candidates: Dict[int, List[Dict[str, Any]]] = {}
        for path in sorted(glob.glob(os.path.join(self.checkpoints_dir, "fold_*.json"))):
            with open(path) as handle:
                info = json.load(handle)
            candidates.setdefault(int(info["fold"]), []).append(info)
        return {fold: max(infos, key=lambda info: info.get("best_score") or 0.0)
                for fold, infos in sorted(candidates.items())}

    def region_label(self) -> str:
        """Human-readable label of this region, e.g. 'et2_eta0' or 'full phase space'."""
        if self.et_bin is None:
            return "full phase space"
        return kinematics.bin_label(self.et_bin, self.eta_bin)

    def cli_region_args(self) -> str:
        """The `--et-bin/--eta-bin` fragment that reproduces this region ('' when ungridded)."""
        if self.et_bin is None:
            return ""
        return f"--et-bin {self.et_bin} --eta-bin {self.eta_bin}"
