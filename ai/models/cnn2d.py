import torch.nn as nn

from ai.models.base import BaseBinaryClassifier


class ModelCNN2D(BaseBinaryClassifier):
    """
    2D CNN over the seven calorimeter layers treated as image channels.

    Everything except the architecture lives in BaseBinaryClassifier.
    """

    def build_network(self, in_channels: int = 7, cell_height: int = 7, cell_width: int = 15) -> nn.Module:
        """
        Builds the convolutional feature extractor plus its classifier head.

        Args:
            in_channels (int): Calorimeter layers, used as image channels. Defaults to 7.
            cell_height (int): Cell grid height. Defaults to 7.
            cell_width (int): Cell grid width. Defaults to 15.

        Returns:
            nn.Module: (Batch, in_channels, cell_height, cell_width) -> (Batch, 1) logits.
        """
        features = nn.Sequential(
            # Convolutional Block 1
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Convolutional Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Derived rather than hardcoded, so changing the kernel/pool sizes above or the cell
        # grid does not silently produce a shape mismatch in the first Linear.
        flat_dim = 64 * (cell_height // 2) * (cell_width // 2)

        return nn.Sequential(
            features,
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
