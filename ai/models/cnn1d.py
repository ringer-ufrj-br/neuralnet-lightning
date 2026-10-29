import torch.nn as nn
from ai.models.base import BaseBinaryClassifier 

class ModelCNN1D(BaseBinaryClassifier):
    def build_network(self, in_channels: int = 1, seq_len = 50) -> nn.Module:
        """The convolutional feature extractor plus its classifier head:
        (Batch, in_channels, seq_len) -> (Batch, 1) logits."""
        features = nn.Sequential(
            # Convolutional Block 1:
            nn.Conv1d(in_channels = in_channels, out_channels = 32, kernel_size = 3, padding = 1), 
            nn.BatchNorm1d(32), 
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size = 2), 

            # Convolutional Block 2:
            nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1),
            nn.BatchNorm1d(64)
            nn.ReLU(),
            )

        # flat_dim is important to derive rather than hardcode, so changing the kernet/pool sizes above or the cell grid does not silently produce a shape mismatch in the first linear.
        flat_dim = 64 * (seq_len // 2)

        return nn.sequencial( 
            features,
            nn.Flatten(), 
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, 1)
        )