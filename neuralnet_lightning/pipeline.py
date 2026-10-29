"""
The orchestrator: config -> loader -> preprocessor -> training / evaluation, for one region.

The two phases run as separate commands, on different machines and at different times:

* `train`    - loads the region, fits the preprocessor, runs the K-Fold and persists every
               (fold, init) model plus what evaluation needs to reproduce its inputs.
* `evaluate` - picks each fold's best initialisation, scores the whole region with it, and
               hands the scores to the model's task layer for metrics and plots.

Keeping them apart means re-scoring or re-cutting operating points never requires retraining,
and every reported number comes from a checkpoint that is on disk and addressable.
"""

import logging
import os
from typing import Optional

import joblib
import polars as pl

from neuralnet_lightning.binning import Region
from neuralnet_lightning.config import ExperimentConfig
from neuralnet_lightning.data.loader import load_dataframe
from neuralnet_lightning.data.schema import ET, ETA, LABEL, ROW_ID
from neuralnet_lightning.evaluation import best_inits, load_scores, score_folds
from neuralnet_lightning.models.registry import get_model
from neuralnet_lightning.paths import RunPaths
from neuralnet_lightning.preprocess.base import BasePreprocessor
from neuralnet_lightning.training import fit_kfold

logger = logging.getLogger(__name__)


class Pipeline:
    """One experiment in one region (None: the whole phase space)."""

    def __init__(self, config: ExperimentConfig, region: Optional[Region] = None) -> None:
        self.config = config
        self.region = region
        self.model_cls = get_model(config.model)
        self.paths = RunPaths.for_run(config.results_root, config.name, region)

    @property
    def where(self) -> str:
        """Human-readable region, for logs."""
        return f"{self.region.label}: {self.region.description}" if self.region else "full phase space"

    def _load(self, preprocessor: BasePreprocessor) -> pl.DataFrame:
        """The region's rows, with exactly the columns the preprocessor consumes."""
        return load_dataframe(self.config.dataset, preprocessor.required_columns(), self.region)

    def train(self, fold: Optional[int] = None, init: Optional[int] = None) -> None:
        """
        Trains the requested (fold, init) models - all of them by default. Safe to run as
        parallel single-model jobs: the split is a pure function of (data, seed), the shared
        artefacts are written atomically with identical content, and each (fold, init) owns its
        checkpoint and sidecar.
        """
        logger.info(f"🚀 Training '{self.config.name}' ({self.config.model}) on {self.where}")
        preprocessor = self.model_cls.preprocessor()
        df = self._load(preprocessor)

        y = df[LABEL].cast(pl.Float32).to_numpy()
        X = preprocessor.fit_transform(df)
        # Everything training needs is in X and y now; the frame would otherwise stay resident,
        # as large as X itself, for the whole of training.
        del df

        os.makedirs(self.paths.artifacts, exist_ok=True)
        tmp_path = f"{self.paths.preprocessor}.tmp.{os.getpid()}"
        joblib.dump(preprocessor, tmp_path)
        os.replace(tmp_path, self.paths.preprocessor)
        logger.info(f"💾 Saved preprocessor to: {self.paths.preprocessor}")

        # The config's model_args win over data-derived kwargs: they are the explicit choice.
        model_kwargs = {"learning_rate": self.config.training.learning_rate,
                        **preprocessor.model_kwargs(X), **self.config.model_args}
        logger.info(f"🏋️ Model kwargs: {model_kwargs}")

        fit_kfold(self.model_cls, model_kwargs, X, y, self.config.training, self.paths,
                  stratify=self.model_cls.stratify_labels(y), fold=fold, init=init)

        logger.info(f"✅ Artefacts under: {self.paths.root}")
        region_args = f" {self.region.cli_args}" if self.region else ""
        logger.info(f"   Once every training of this region has finished: "
                    f"python -m neuralnet_lightning evaluate --config <config>{region_args}")

    def evaluate(self, reuse_scores: bool = False, make_plots: bool = True) -> None:
        """
        Scores each fold's best initialisation over the whole region and writes the region's
        metrics and plots. `reuse_scores` skips data and inference and reads the scores of an
        earlier evaluation. Raises FileNotFoundError if the region has not been trained.
        """
        infos = best_inits(self.paths, self.model_cls.monitor_mode)
        if not infos:
            raise FileNotFoundError(f"❌ No trained model found in '{self.paths.checkpoints}'. "
                                    f"Run `train` for this region first.")
        logger.info(f"📊 Evaluating '{self.config.name}' on {self.where}: {len(infos)} fold(s)")

        if reuse_scores:
            y, scores = load_scores(infos, self.paths)
        else:
            preprocessor = joblib.load(self.paths.preprocessor)
            df = self._load(preprocessor)
            y = df[LABEL].cast(pl.Float32).to_numpy()

            # The fold indices only mean something over the same rows in the same order.
            expected = next(iter(infos.values()))
            if len(y) != expected["n_train"] + expected["n_val"]:
                raise ValueError(f"❌ The region has {len(y)} rows but was trained on "
                                 f"{expected['n_train'] + expected['n_val']}: the data or the dataset "
                                 f"config changed since training. Retrain, or restore them.")

            X = preprocessor.transform(df)
            # Kept next to the scores so the table can be re-cut later without inference.
            extra = {name: df[name].to_numpy() for name in (ET, ETA, ROW_ID) if name in df.columns}
            del df
            scores = score_folds(self.model_cls, X, y, infos, self.paths, extra)

        self.model_cls.summarize(self.config.name, self.region, y, scores, infos,
                                 self.config.evaluation.operating_points, self.paths, make_plots)
        logger.info(f"✅ Evaluation complete. Results under: {self.paths.root}")
