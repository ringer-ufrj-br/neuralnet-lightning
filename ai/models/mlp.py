import torch.nn as nn

from ai.models.base import BaseBinaryClassifier


class ModelMLP(BaseBinaryClassifier):
    """
    The Ringer MLP: one hidden layer of 5 neurons over the 50 selected rings.

    Everything except the architecture lives in BaseBinaryClassifier.
    """

    def build_network(self, input_dim: int = 100) -> nn.Module:
        """(Batch, input_dim) -> (Batch, 1) logits; input_dim comes from PipelineMLP.build_model_kwargs."""
        return nn.Sequential(
            nn.Linear(input_dim, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
