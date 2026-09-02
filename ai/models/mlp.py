import torch.nn as nn

from ai.models.base import BaseBinaryClassifier


class ModelMLP(BaseBinaryClassifier):
    """
    The Ringer MLP: one hidden layer of 5 neurons over the 50 selected rings.

    Everything except the architecture lives in BaseBinaryClassifier.
    """

    def build_network(self, input_dim: int = 100) -> nn.Module:
        """
        Builds the ring-based MLP.

        Args:
            input_dim (int): Number of input features (rings). Supplied by
                PipelineMLP.build_model_kwargs from the preprocessed feature matrix.

        Returns:
            nn.Module: (Batch, input_dim) -> (Batch, 1) logits.
        """
        return nn.Sequential(
            nn.Linear(input_dim, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
