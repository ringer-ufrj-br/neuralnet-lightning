import inspect

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, MetricCollection, Precision, Recall, F1Score, AveragePrecision
from typing import Tuple, Any, Optional, Union

from ai.evaluation.metrics import max_sp_index


class BaseBinaryClassifier(pl.LightningModule):
    """
    Everything every architecture in this project shares: the weighted BCE criterion, the
    epoch-level metric collection, the SP-index validation hook that EarlyStopping and
    ModelCheckpoint monitor, and the optimizer.

    A new architecture subclasses this and implements ONE method, `build_network`:

        class ModelMyNet(BaseBinaryClassifier):
            def build_network(self, input_dim: int = 100) -> nn.Module:
                return nn.Sequential(nn.Linear(input_dim, 5), nn.ReLU(), nn.Linear(5, 1))

    Do not write an `__init__` in the subclass. Whatever keyword arguments the pipeline puts
    in `build_model_kwargs` arrive here as `**arch_kwargs` and are forwarded straight to
    `build_network`, so declaring them as `build_network` parameters is all that is needed -
    they are saved as hyperparameters and restored by `load_from_checkpoint` automatically.

    Three further hooks exist for architectures that need them; none is required:

        forward             - override when the network is not a single callable module.
        compute_loss        - override to add auxiliary losses (see ModelFused).
        configure_optimizers- override for a different optimizer or a scheduler.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        pos_weight: Optional[Union[float, torch.Tensor]] = None,
        **arch_kwargs: Any
    ) -> None:
        """
        `pos_weight` is left out of the saved hyperparameters: it is derived from the training
        split, not chosen by the user, so evaluation feeds it back from the fold sidecar.
        """
        super().__init__()
        self.learning_rate = learning_rate

        # build_network's own defaults are folded in before saving, so self.hparams carries
        # every architecture parameter whether or not the caller passed it, and a checkpoint
        # keeps rebuilding the same network even if a default changes later. bind() also
        # rejects an argument build_network does not declare.
        arch = inspect.signature(self.build_network).bind(**arch_kwargs)
        arch.apply_defaults()
        self.save_hyperparameters({"learning_rate": learning_rate, **arch.arguments})

        #: The architecture itself, whatever build_network returned.
        self.network = self.build_network(**arch.arguments)

        # pos_weight is a buffer, not a parameter: it moves with .to(device) but the optimizer
        # never touches it.
        if pos_weight is not None:
            pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32).reshape(1)
        self.register_buffer("pos_weight", pos_weight)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        metrics = self.build_metrics()
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')

        # Buffers accumulated across validation batches to compute the epoch-level SP Index
        self._val_preds: list = []
        self._val_targets: list = []

    # ------------------------------------------------------------------ hooks

    def build_network(self, **kwargs: Any) -> nn.Module:
        """
        Builds the architecture. THIS IS THE ONE METHOD A NEW ARCHITECTURE MUST IMPLEMENT.

        Declare the architecture's hyperparameters as this method's keyword arguments; the
        pipeline supplies them through build_model_kwargs. The module maps (Batch, ...) inputs
        to (Batch, 1) raw logits, NOT probabilities - the loss applies the sigmoid.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement build_network() and return an nn.Module."
        )

    def build_metrics(self) -> MetricCollection:
        """
        Builds the per-epoch metric collection, without a train_/val_ prefix. Override only to
        add or drop metrics.

        Grouping them in a MetricCollection lets the threshold-based ones (acc/precision/
        recall/f1) share one confusion-matrix state and the curve-based ones (auc_roc/auc_pr)
        share one prediction buffer. They are only ever `.update()`d in the steps and computed
        once per epoch: computing them per batch used to dominate training time, because
        AUROC re-sorts its whole buffer on every compute.
        """
        return MetricCollection({
            'acc': Accuracy(task="binary"),
            'precision': Precision(task="binary"),
            'recall': Recall(task="binary"),
            'f1': F1Score(task="binary"),
            'auc_roc': AUROC(task="binary"),
            'auc_pr': AveragePrecision(task="binary"),
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Raw logits of shape (Batch, 1). The default delegates to the module returned by
        build_network; override when the architecture needs more than a single callable.
        """
        return self.network(x)

    def compute_loss(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        (loss, predicted probabilities, integer targets) for one batch - the last two feed the
        metrics. Override to add auxiliary losses; the metric accumulation, logging and SP
        machinery then keep working unchanged.
        """
        x, y = batch
        y = y.unsqueeze(1).float()
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss, torch.sigmoid(logits), y.long()

    def configure_optimizers(self) -> Any:
        """Adam over every trainable parameter. Override for another optimizer or a scheduler."""
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    # ------------------------------------------------------------------ steps

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step. Accumulates metrics without computing them; see build_metrics."""
        loss, preds, y_int = self.compute_loss(batch)
        self.train_metrics.update(preds, y_int)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        """
        Computes and logs the accumulated training metrics once per epoch, keeping Lightning's
        logging machinery out of the hot loop.
        """
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step. Also buffers predictions so the epoch-level SP Index can be computed
        over the full validation set rather than per batch.
        """
        loss, preds, y_int = self.compute_loss(batch)
        self.val_metrics.update(preds, y_int)

        self._val_preds.append(preds.detach())
        self._val_targets.append(y_int.detach())

        self.log('val_loss', loss, prog_bar=False)
        return loss

    def on_validation_epoch_end(self) -> None:
        """
        Logs the epoch's validation metrics and the SP Index maximised over every decision
        threshold on the full validation set. `val_sp` is what EarlyStopping and
        ModelCheckpoint monitor.
        """
        if not self._val_preds:
            return

        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

        preds = torch.cat(self._val_preds)
        targets = torch.cat(self._val_targets)
        self.log('val_sp', max_sp_index(preds, targets), prog_bar=True)

        self._val_preds.clear()
        self._val_targets.clear()
