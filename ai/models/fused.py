import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC
from typing import Tuple, Any, Union, Optional

class ModelFused(pl.LightningModule):
    """
    Fused PyTorch Lightning module for binary classification.
    Combines a MLP branch (rings) and a CNN2D branch (cells) into a final decision MLP.
    """

    def __init__(self, learning_rate: float = 0.001, n_rings: int = 100, cell_shape: Tuple[int, int, int] = (7, 7, 15), rings_embed_dim: int = 32, cells_embed_dim: int = 64, fusion_source: str = "embedding", aux_loss_weight: float = 0.3, dropout: float = 0.5, pos_weight: Optional[float] = None) -> None:
        """
        Initializes ModelFused architecture and evaluation metrics.
        Expected input shape: (Batch, n_rings + C*H*W).
        """
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        c, h, w = cell_shape
        self.n_rings = n_rings
        self.cell_shape = cell_shape
        self.n_cells = c * h * w

        # Rings Branch (same architecture as ModelMLP)
        self.rings_branch = nn.Sequential(
            nn.Linear(n_rings, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, rings_embed_dim),
            nn.ReLU()
        )
        self.rings_head = nn.Linear(rings_embed_dim, 1)

        # Cells Branch (same architecture as ModelCNN2D)
        self.cells_features = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Flatten size computed dynamically
        with torch.no_grad():
            flat_dim = self.cells_features(torch.zeros(1, c, h, w)).flatten(1).shape[1]

        self.cells_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, cells_embed_dim),
            nn.ReLU()
        )
        self.cells_head = nn.Linear(cells_embed_dim, 1)

        # Fusion Head
        fusion_in = (rings_embed_dim + cells_embed_dim) if fusion_source == "embedding" else 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 32),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Binary Classification Metrics
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.train_auc = AUROC(task="binary")
        self.val_auc = AUROC(task="binary")

        pw = None if pos_weight is None else torch.tensor(float(pos_weight))
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    def _split_inputs(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Splits the input vector into the rings and cells branches."""
        if isinstance(x, (tuple, list)):
            rings, cells = x
        else:
            expected = self.n_rings + self.n_cells
            if x.shape[1] != expected:
                raise ValueError(f"Expected x with {expected} columns, got {x.shape[1]}.")
            rings = x[:, :self.n_rings]
            cells = x[:, self.n_rings:]

        if cells.dim() == 2:
            cells = cells.view(-1, *self.cell_shape)
        return rings, cells

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Fused module. Returns the fusion logits only."""
        logits, _, _ = self._forward_all(x)
        return logits

    def _forward_all(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning the fusion logits and the auxiliary branch logits."""
        rings, cells = self._split_inputs(x)

        z_rings = self.rings_branch(rings)
        z_cells = self.cells_branch(self.cells_features(cells))

        logit_rings = self.rings_head(z_rings)
        logit_cells = self.cells_head(z_cells)

        if self.hparams.fusion_source == "embedding":
            fused = torch.cat([z_rings, z_cells], dim=1)
        else:
            fused = torch.cat([logit_rings, logit_cells], dim=1)

        return self.fusion(fused), logit_rings, logit_cells

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the fusion loss plus the auxiliary supervision of each branch."""
        x, y = batch
        y = y.unsqueeze(1).float()

        logits, logit_rings, logit_cells = self._forward_all(x)
        loss = self.criterion(logits, y)

        # Without the auxiliary loss the weaker branch gets almost no gradient
        wgt = self.hparams.aux_loss_weight
        if wgt > 0:
            loss = loss + wgt * (self.criterion(logit_rings, y) + self.criterion(logit_cells, y))

        return loss, torch.sigmoid(logits), y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step executed on batch."""
        loss, preds, y = self._shared_step(batch)

        self.train_acc(preds, y)
        self.train_auc(preds, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_auc', self.train_auc, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step executed on batch."""
        loss, preds, y = self._shared_step(batch)

        self.val_acc(preds, y)
        self.val_auc(preds, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_auc', self.val_auc, prog_bar=True)

        return loss

    def configure_optimizers(self) -> Any:
        """Configures model optimizer."""
        return optim.Adam(self.parameters(), lr=self.learning_rate)
