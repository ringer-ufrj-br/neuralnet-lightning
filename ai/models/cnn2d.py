import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, Precision, Recall, F1Score, AveragePrecision
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

        # Training Metrics
        self.train_acc = Accuracy(task="binary")
        self.train_precision = Precision(task="binary")
        self.train_recall = Recall(task="binary")
        self.train_f1 = F1Score(task="binary")
        self.train_auc = AUROC(task="binary")
        self.train_pr_auc = AveragePrecision(task="binary")

        # Validation Metrics
        self.val_acc = Accuracy(task="binary")
        self.val_precision = Precision(task="binary")
        self.val_recall = Recall(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.val_auc = AUROC(task="binary")
        self.val_pr_auc = AveragePrecision(task="binary")

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
        
        self.train_acc(preds, y_int)
        self.train_precision(preds, y_int)
        self.train_recall(preds, y_int)
        self.train_f1(preds, y_int)
        self.train_auc(preds, y_int)
        self.train_pr_auc(preds, y_int)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_auc_roc', self.train_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_auc_pr', self.train_pr_auc, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss

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
        
        self.val_acc(preds, y_int)
        self.val_precision(preds, y_int)
        self.val_recall(preds, y_int)
        self.val_f1(preds, y_int)
        self.val_auc(preds, y_int)
        self.val_pr_auc(preds, y_int)

        self._val_preds.append(preds.detach())
        self._val_targets.append(y_int.detach())

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=False)
        self.log('val_f1', self.val_f1, prog_bar=True)
        self.log('val_precision', self.val_precision, prog_bar=False)
        self.log('val_recall', self.val_recall, prog_bar=False)
        self.log('val_auc_roc', self.val_auc, prog_bar=True)
        self.log('val_auc_pr', self.val_pr_auc, prog_bar=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        """
        Computes the SP Index over the full validation set at the default 0.5 decision
        boundary. Available for EarlyStopping/ModelCheckpoint to monitor, mirroring ModelMLP.
        """
        if not self._val_preds:
            return

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
