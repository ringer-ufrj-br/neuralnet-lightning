import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, MetricCollection, Precision, Recall, F1Score, AveragePrecision
from typing import Tuple, Any, Optional, Union

from ai.evaluation.metrics import compute_pd_fa, sp_index

class ModelCNN2D(pl.LightningModule):
    """
    2D Convolutional Neural Network (CNN2D) PyTorch Lightning module for binary classification
    with weighted loss support and comprehensive evaluation metrics.
    """

    def __init__(
        self, 
        learning_rate: float = 0.001,
        pos_weight: Optional[Union[float, torch.Tensor]] = None
    ) -> None:
        """
        Initializes ModelCNN2D architecture and evaluation metrics.

        Args:
            learning_rate (float): Optimizer learning rate. Defaults to 0.001.
            pos_weight (Optional[Union[float, torch.Tensor]]): Positive class weight for loss balancing. Defaults to None.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['pos_weight'])
        self.learning_rate = learning_rate

        # Network Architecture
        self.features = nn.Sequential(
            # Convolutional Block 1
            # Input shape: (Batch, 7, 7, 15)
            nn.Conv2d(in_channels=7, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # Output shape: (Batch, 32, 3, 7)
            
            # Convolutional Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # Classifier Head (Flattens (Batch, 64, 3, 7) into (Batch, 1344))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Register pos_weight as buffer (moves with device, not trained by optimizer)
        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor([pos_weight], dtype=torch.float32)
            elif pos_weight.ndim == 0:
                pos_weight = pos_weight.unsqueeze(0).float()
            else:
                pos_weight = pos_weight.float()
        self.register_buffer("pos_weight", pos_weight)

        # Loss Criterion
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        # Epoch-level metrics, grouped in a MetricCollection so the threshold-based ones
        # (acc/precision/recall/f1) share one confusion-matrix state and the curve-based ones
        # (auc_roc/auc_pr) share one prediction buffer. Updated with .update() in the steps -
        # never called directly - so nothing is computed per batch; all logging is on_epoch.
        # Computing these per step (the previous metric(preds, y) pattern) dominated training
        # time: AUROC alone re-sorted its whole buffer on every step of an epoch.
        metrics = MetricCollection({
            'acc': Accuracy(task="binary"),
            'precision': Precision(task="binary"),
            'recall': Recall(task="binary"),
            'f1': F1Score(task="binary"),
            'auc_roc': AUROC(task="binary"),
            'auc_pr': AveragePrecision(task="binary"),
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')

        # Buffers accumulated across validation batches to compute the epoch-level SP Index
        self._val_preds: list = []
        self._val_targets: list = []

    def set_pos_weight(self, pos_weight: Union[float, torch.Tensor]) -> None:
        """
        Dynamically sets or updates the positive class weight buffer and recreation of criterion.

        Args:
            pos_weight (Union[float, torch.Tensor]): Positive class weight value or tensor.
        """
        if not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor([pos_weight], dtype=torch.float32)
        elif pos_weight.ndim == 0:
            pos_weight = pos_weight.unsqueeze(0).float()
        else:
            pos_weight = pos_weight.float()
            
        self.register_buffer("pos_weight", pos_weight)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CNN2D module.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 7, 7, 15).

        Returns:
            torch.Tensor: Unnormalized output logits of shape (Batch, 1).
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step executed on batch.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Batch containing features (x) and targets (y).
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Calculated training loss tensor.
        """
        x, y = batch
        y = y.unsqueeze(1).float()
        
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.sigmoid(logits)
        y_int = y.long()
        
        self.train_metrics.update(preds, y_int)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        """
        Computes and logs the accumulated training metrics once per epoch. Logging them here
        instead of per step keeps Lightning's logging machinery out of the hot loop.
        """
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step executed on batch.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Batch containing features (x) and targets (y).
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Calculated validation loss tensor.
        """
        x, y = batch
        y = y.unsqueeze(1).float()
        
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.sigmoid(logits)
        y_int = y.long()
        
        self.val_metrics.update(preds, y_int)

        self._val_preds.append(preds.detach())
        self._val_targets.append(y_int.detach())

        self.log('val_loss', loss, prog_bar=False)

        return loss

    def on_validation_epoch_end(self) -> None:
        """
        Computes the SP Index over the full validation set at the default 0.5 decision
        boundary. Available for EarlyStopping/ModelCheckpoint to monitor, mirroring ModelMLP,
        and logs the accumulated validation metrics for the epoch.
        """
        if not self._val_preds:
            return

        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

        preds = torch.cat(self._val_preds)
        targets = torch.cat(self._val_targets)
        pd_rate, fa_rate = compute_pd_fa(preds, targets)
        sp = sp_index(pd_rate, fa_rate)

        self.log('val_sp', sp, prog_bar=True)

        self._val_preds.clear()
        self._val_targets.clear()

    def configure_optimizers(self) -> Any:
        """
        Configures model optimizer.

        Args:
            None

        Returns:
            Any: Adam optimizer instance.
        """
        return optim.Adam(self.parameters(), lr=self.learning_rate)
