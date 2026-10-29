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
    Everything every binary architecture shares: the pos_weight-weighted BCE criterion, the
    epoch-level metric collection, the SP-index validation metric that EarlyStopping and
    ModelCheckpoint monitor, and the optimizer.

    A new architecture subclasses this, names its preprocessor and implements ONE method:

        @register_model("MyNet")
        class MyNet(BaseBinaryClassifier):
            preprocessor = PreprocessMLP
            def build_network(self, input_dim: int = 50, hidden: int = 5) -> nn.Module:
                return nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    Do not write an `__init__` in the subclass. The preprocessor's `model_kwargs` and the
    config's `model_args` arrive here as `**arch_kwargs` and are forwarded to `build_network`,
    so declaring them as its parameters is all it takes - they are saved as hyperparameters and
    restored by `load_from_checkpoint`.

    Further hooks, none required: `forward` (a network that is not one callable module),
    `compute_loss` (auxiliary losses), `configure_optimizers` (another optimizer/scheduler).
    """

    #: The input representation this model consumes.
    preprocessor: ClassVar[Type[BasePreprocessor]]

    #: SP index maximised over every threshold (see on_validation_epoch_end). Higher is better.
    monitor: ClassVar[str] = "val_sp"
    monitor_mode: ClassVar[str] = "max"

    def __init__(
        self,
        learning_rate: float = 0.001,
        pos_weight: Optional[Union[float, torch.Tensor]] = None,
        **arch_kwargs: Any
    ) -> None:
        """
        `pos_weight` is left out of the saved hyperparameters: it is derived from the fold's
        training split, not chosen by the user, so evaluation feeds it back from the sidecar.
        """
        super().__init__()
        self.learning_rate = learning_rate

        # build_network's own defaults are folded in before saving, so self.hparams carries
        # every architecture parameter whether or not the caller passed it, and a checkpoint
        # keeps rebuilding the same network even if a default changes later. bind() also
        # rejects an argument build_network does not declare (a typo in model_args).
        arch = inspect.signature(self.build_network).bind(**arch_kwargs)
        arch.apply_defaults()
        self.save_hyperparameters({"learning_rate": learning_rate, **arch.arguments})

        #: The architecture itself, whatever build_network returned.
        self.network = self.build_network(**arch.arguments)

        # A buffer, not a parameter: it moves with .to(device) but the optimizer never sees it.
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
        """
        pos_weight = n_negatives / n_positives over one fold's training labels, so the weighted
        loss balances the classes instead of discarding background events.
        """
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
        """What the K-Fold stratifies on: the class itself, so every fold keeps the imbalance."""
        return y

    def predict_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Post-sigmoid signal probabilities, shape (Batch,)."""
        return torch.sigmoid(self(x)).flatten()

    #: Scores -> figures of merit, plots and folds_long.csv for one region.
    summarize = staticmethod(summarize)

    # ------------------------------------------------------------------ hooks

    def build_network(self, **kwargs: Any) -> nn.Module:
        """
        Builds the architecture. THE ONE METHOD A NEW ARCHITECTURE MUST IMPLEMENT.

        The module maps (Batch, ...) inputs to (Batch, 1) raw logits, NOT probabilities - the
        loss applies the sigmoid.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement build_network() and return an nn.Module.")

    def build_metrics(self) -> MetricCollection:
        """
        The per-epoch metric collection, without a train_/val_ prefix. Override only to add or
        drop metrics.

        Grouping them lets the threshold-based ones share one confusion-matrix state and the
        curve-based ones one prediction buffer. They are only `.update()`d in the steps and
        computed once per epoch: computing per batch used to dominate training time, because
        AUROC re-sorts its whole buffer on every compute.
        """
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
        """
        (loss, predicted probabilities, integer targets) for one batch - the last two feed the
        metrics. Override to add auxiliary losses; the metric and SP machinery keep working.
        """
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
        """Computes the accumulated training metrics once per epoch, outside the hot loop."""
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
        """
        Logs the epoch's validation metrics and `val_sp`, the SP index maximised over every
        threshold on the full validation set - not averaged per batch.
        """
        if not self._val_preds:
            return
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

        preds = torch.cat(self._val_preds)
        targets = torch.cat(self._val_targets)
        self.log("val_sp", max_sp_index(preds, targets), prog_bar=True)

        self._val_preds.clear()
        self._val_targets.clear()
