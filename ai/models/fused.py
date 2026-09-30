import torch
import torch.nn as nn
from typing import Tuple, Union

from ai.models.base import BaseBinaryClassifier


class ModelFused(BaseBinaryClassifier):
    """
    Two-branch model: an MLP over the rings and a CNN over the calorimeter cells, joined by a
    fusion head. Each branch also carries an auxiliary classification head, because without
    auxiliary supervision the weaker branch receives almost no gradient.

    This is the example of an architecture that needs more than build_network: it overrides
    `forward` (two branches, not one callable) and `compute_loss` (the auxiliary terms).
    Everything else - metrics, SP index, logging, optimizer - still comes from the base.
    """

    def build_network(
        self,
        n_rings: int = 100,
        cell_shape: Tuple[int, int, int] = (7, 7, 15),
        rings_embed_dim: int = 32,
        cells_embed_dim: int = 64,
        fusion_source: str = "embedding",
        aux_loss_weight: float = 0.3,
        dropout: float = 0.5
    ) -> nn.Module:
        """
        Both branches, their auxiliary heads and the fusion head, as a ModuleDict that
        forward() wires together. fusion_source 'embedding' concatenates the two embeddings,
        anything else the two auxiliary logits; aux_loss_weight 0 disables the auxiliary losses.
        """
        c, h, w = cell_shape
        self.n_rings = n_rings
        self.cell_shape = cell_shape
        self.n_cells = c * h * w

        rings_branch = nn.Sequential(
            nn.Linear(n_rings, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, rings_embed_dim),
            nn.ReLU()
        )

        cells_features = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        with torch.no_grad():
            flat_dim = cells_features(torch.zeros(1, c, h, w)).flatten(1).shape[1]

        cells_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, cells_embed_dim),
            nn.ReLU()
        )

        fusion_in = (rings_embed_dim + cells_embed_dim) if fusion_source == "embedding" else 2

        return nn.ModuleDict({
            "rings_branch": rings_branch,
            "rings_head": nn.Linear(rings_embed_dim, 1),
            "cells_features": cells_features,
            "cells_branch": cells_branch,
            "cells_head": nn.Linear(cells_embed_dim, 1),
            "fusion": nn.Sequential(
                nn.Linear(fusion_in, 32),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            ),
        })

    def _split_inputs(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        (rings, cells) with cells shaped (B, C, H, W), from either the concatenated
        (rings | flattened cells) tensor or an explicit (rings, cells) pair.
        """
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

    def _forward_all(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs both branches and the fusion head: (fusion, rings, cells) logits."""
        rings, cells = self._split_inputs(x)
        net = self.network

        z_rings = net["rings_branch"](rings)
        z_cells = net["cells_branch"](net["cells_features"](cells))

        logit_rings = net["rings_head"](z_rings)
        logit_cells = net["cells_head"](z_cells)

        if self.hparams.fusion_source == "embedding":
            fused = torch.cat([z_rings, z_cells], dim=1)
        else:
            fused = torch.cat([logit_rings, logit_cells], dim=1)

        return net["fusion"](fused), logit_rings, logit_cells

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The fusion logits only - the auxiliary heads exist for training."""
        return self._forward_all(x)[0]

    def compute_loss(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fusion loss plus the weighted auxiliary supervision of each branch."""
        x, y = batch
        y = y.unsqueeze(1).float()

        logits, logit_rings, logit_cells = self._forward_all(x)
        loss = self.criterion(logits, y)

        weight = self.hparams.aux_loss_weight
        if weight > 0:
            loss = loss + weight * (self.criterion(logit_rings, y) + self.criterion(logit_cells, y))

        return loss, torch.sigmoid(logits), y.long()
