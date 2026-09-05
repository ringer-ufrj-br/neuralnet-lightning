import logging
from typing import Any, Dict

import numpy as np

from ai.pipeline.base import BasePipeline
from ai.pipeline.registry import register_pipeline
from ai.preprocess.mlp import PreprocessMLP
from ai.models.mlp import ModelMLP

logger = logging.getLogger(__name__)


@register_pipeline("MLP")
class PipelineMLP(BasePipeline):
    """
    Training and evaluation pipeline for the ring-based MLP.
    """

    model_class = ModelMLP
    preprocessor_class = PreprocessMLP

    def build_model_kwargs(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Derives the MLP input dimension from the preprocessed feature matrix.

        Args:
            X (np.ndarray): Preprocessed training features, shape (N, n_features).

        Returns:
            Dict[str, Any]: {'input_dim': n_features}, forwarded to ModelMLP.build_network.
        """
        input_dim = int(X.shape[1])
        logger.info(f"📐 Model input dimension: {input_dim}")
        return {"input_dim": input_dim}
