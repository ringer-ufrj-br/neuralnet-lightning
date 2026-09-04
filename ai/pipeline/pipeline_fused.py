import logging
from typing import Any, Dict

import numpy as np

from ai.pipeline.base import BasePipeline
from ai.pipeline.registry import register_pipeline
from ai.preprocess.fused import PreprocessFused
from ai.models.fused import ModelFused

logger = logging.getLogger(__name__)


@register_pipeline("Fused")
class PipelineFused(BasePipeline):
    """
    Training and evaluation pipeline for the two-branch rings + cells model.

    The preprocessor emits one flat vector per event (rings first, then the flattened cell
    image); ModelFused splits it back apart, so the model needs to be told where the boundary
    is and what shape the cell half unflattens to.
    """

    model_class = ModelFused
    preprocessor_class = PreprocessFused

    def build_model_kwargs(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Tells the model where the rings end and the cell image begins.

        Args:
            X (np.ndarray): Preprocessed training features, shape (N, n_rings + C*H*W).

        Returns:
            Dict[str, Any]: {'n_rings': ..., 'cell_shape': ...}.
        """
        n_rings = len(self.preprocessor.ring_columns)
        cell_shape = self.preprocessor.cells_pp.target_shape
        logger.info(f"📐 Fused input: {n_rings} rings + cells{tuple(cell_shape)} = {X.shape[1]} features")
        return {"n_rings": n_rings, "cell_shape": tuple(cell_shape)}
