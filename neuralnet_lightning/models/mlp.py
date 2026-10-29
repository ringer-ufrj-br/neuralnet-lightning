import torch.nn as nn

from neuralnet_lightning.binary.model import BaseBinaryClassifier
from neuralnet_lightning.models.registry import register_model
from neuralnet_lightning.preprocess.mlp import PreprocessMLP


@register_model("MLP")
class ModelMLP(BaseBinaryClassifier):
    """Ringer MLP: one hidden layer over the 50 selected rings (norm1)."""

    preprocessor = PreprocessMLP

    def build_network(self, input_dim: int = 50, hidden: int = 5) -> nn.Module:
        """(Batch, input_dim) -> (Batch, 1) logits; input_dim comes from PreprocessMLP."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
