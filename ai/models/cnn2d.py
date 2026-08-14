import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC
from typing import Tuple, Any

class ModelCNN2D(pl.LightningModule):
    """
    2D Convolutional Neural Network (CNN2D) PyTorch Lightning module for binary classification.
    """

    def __init__(self, learning_rate: float = 0.001) -> None:
        """
        Initializes ModelCNN2D architecture and evaluation metrics.

        Args:
            learning_rate (float): Optimizer learning rate. Defaults to 0.001.
        """
        super().__init__()
        self.save_hyperparameters()
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

        # Binary Classification Metrics
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.train_auc = AUROC(task="binary")
        self.val_auc = AUROC(task="binary")
        
        self.criterion = nn.BCEWithLogitsLoss()

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
        
        self.train_acc(preds, y)
        self.train_auc(preds, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_auc', self.train_auc, on_step=False, on_epoch=True, prog_bar=False)
        
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
        
        self.val_acc(preds, y)
        self.val_auc(preds, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_auc', self.val_auc, prog_bar=True)
        
        return loss

    def configure_optimizers(self) -> Any:
        """
        Configures model optimizer.

        Args:
            None

        Returns:
            Any: Adam optimizer instance.
        """
        return optim.Adam(self.parameters(), lr=self.learning_rate)
