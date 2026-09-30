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
        """The MLP input dimension, from the preprocessed (N, n_features) matrix."""
        input_dim = int(X.shape[1])
        logger.info(f"📐 Model input dimension: {input_dim}")
        return {"input_dim": input_dim}
