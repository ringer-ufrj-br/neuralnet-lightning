"""The base class of every electron x jet classifier."""

import inspect
import logging
from typing import Any, ClassVar, Dict, Optional, Tuple, Type, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score, MetricCollection, Precision, Recall

from neuralnet_lightning.binary.metrics import max_sp_index
from neuralnet_lightning.binary.summary import summarize
from neuralnet_lightning.preprocess.base import BasePreprocessor

logger = logging.getLogger(__name__)


class BaseBinaryClassifier(pl.LightningModule):
    """
    Base of the binary classifiers: pos_weight-weighted BCE loss, epoch-level metrics, the
    `val_sp` validation metric and the Adam optimizer.

    A subclass sets `preprocessor` and implements `build_network`, without an `__init__`:

        @register_model("MyNet")
        class MyNet(BaseBinaryClassifier):
            preprocessor = PreprocessMLP
            def build_network(self, input_dim: int = 50, hidden: int = 5) -> nn.Module:
                return nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    The preprocessor's `model_kwargs` and the config's `model_args` are passed to
    `build_network` and saved as hyperparameters. `forward`, `compute_loss` and
    `configure_optimizers` may also be overridden.
    """

    #: The input representation this model consumes.
    preprocessor: ClassVar[Type[BasePreprocessor]]

    #: Metric for early stopping, checkpointing and init selection.
    monitor: ClassVar[str] = "val_sp"
    monitor_mode: ClassVar[str] = "max"

    def __init__(
        self,
        learning_rate: float = 0.001,
        pos_weight: Optional[Union[float, torch.Tensor]] = None,
        **arch_kwargs: Any
    ) -> None:
        """`pos_weight` is not saved as a hyperparameter; evaluation restores it from the sidecar."""
        super().__init__()
        self.learning_rate = learning_rate

        # build_network's defaults are saved too; bind() rejects undeclared arguments.
        arch = inspect.signature(self.build_network).bind(**arch_kwargs)
        arch.apply_defaults()
        self.save_hyperparameters({"learning_rate": learning_rate, **arch.arguments})

        self.network = self.build_network(**arch.arguments)

        # A buffer: follows the module's device but is not optimised.
        if pos_weight is not None:
            pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32).reshape(1)
        self.register_buffer("pos_weight", pos_weight)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        metrics = self.build_metrics()
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

        # Accumulated across validation batches for the epoch-level SP index.
        self._val_preds: list = []
        self._val_targets: list = []

    # ---------------------------------------------------------- core contract

    @classmethod
    def fold_kwargs(cls, y_train: np.ndarray) -> Dict[str, Any]:
        """pos_weight = n_negatives / n_positives over the fold's training labels."""
        y_train = np.asarray(y_train).ravel()
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        if n_pos == 0:
            logger.warning("⚠️ No positive samples in the training split; pos_weight defaults to 1.0.")
            return {"pos_weight": 1.0}
        pos_weight = n_neg / n_pos
        logger.info(f"⚖️ Class weight (training split only): negatives={n_neg}, positives={n_pos} "
                    f"-> pos_weight={pos_weight:.4f}")
        return {"pos_weight": pos_weight}

    @classmethod
    def stratify_labels(cls, y: np.ndarray) -> Optional[np.ndarray]:
        """Labels the K-Fold stratifies on."""
        return y

    def predict_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Post-sigmoid signal probabilities, shape (Batch,)."""
        return torch.sigmoid(self(x)).flatten()

    #: Scores -> figures of merit, plots and folds_long.csv for one region.
    summarize = staticmethod(summarize)

    # ------------------------------------------------------------------ hooks

    def build_network(self, **kwargs: Any) -> nn.Module:
        """Builds the network: (Batch, ...) inputs -> (Batch, 1) logits. Implemented by subclasses."""
        raise NotImplementedError(f"{type(self).__name__} must implement build_network() and return an nn.Module.")

    def build_metrics(self) -> MetricCollection:
        """Per-epoch metrics, without the train_/val_ prefix. Updated per step, computed per epoch."""
        return MetricCollection({
            "acc": Accuracy(task="binary"),
            "precision": Precision(task="binary"),
            "recall": Recall(task="binary"),
            "f1": F1Score(task="binary"),
            "auc_roc": AUROC(task="binary"),
            "auc_pr": AveragePrecision(task="binary"),
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logits of shape (Batch, 1)."""
        return self.network(x)

    def compute_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(loss, predicted probabilities, integer targets) for one batch."""
        x, y = batch
        y = y.unsqueeze(1).float()
        logits = self(x)
        return self.criterion(logits, y), torch.sigmoid(logits), y.long()

    def configure_optimizers(self) -> Any:
        """Adam over every trainable parameter."""
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    # ------------------------------------------------------------------ steps

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds, y_int = self.compute_loss(batch)
        self.train_metrics.update(preds, y_int)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        """Logs and resets the training metrics."""
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds, y_int = self.compute_loss(batch)
        self.val_metrics.update(preds, y_int)
        self._val_preds.append(preds.detach())
        self._val_targets.append(y_int.detach())
        self.log("val_loss", loss, prog_bar=False)
        return loss

    def on_validation_epoch_end(self) -> None:
        """Logs the validation metrics and `val_sp`, computed over the whole validation set."""
        if not self._val_preds:
            return
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

        preds = torch.cat(self._val_preds)
        targets = torch.cat(self._val_targets)
        self.log("val_sp", max_sp_index(preds, targets), prog_bar=True)

        self._val_preds.clear()
        self._val_targets.clear()
