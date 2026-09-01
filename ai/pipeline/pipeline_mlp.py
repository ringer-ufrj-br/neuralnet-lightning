import logging
from typing import Any, Dict, List

import numpy as np

from ai.pipeline.base import BasePipeline
from ai.preprocess.mlp import PreprocessMLP
from ai.models.mlp import ModelMLP

logger = logging.getLogger(__name__)


class PipelineMLP(BasePipeline):
    """
    Training and evaluation pipeline for the ring-based MLP.

    Everything structural lives in BasePipeline; this class only declares which model and
    preprocessor to use and how the architecture's input dimension is derived from the data.
    """

    model_class = ModelMLP

    def build_preprocessor(self) -> PreprocessMLP:
        """
        Builds the ring-selection + log1p + StandardScaler preprocessor.

        Returns:
            PreprocessMLP: A fresh, unfitted preprocessor.
        """
        return PreprocessMLP()

    def required_columns(self, available: List[str]) -> List[str]:
        """
        Restricts the parquet scan to the 50 selected ring columns (resolving the
        cl_ring_*/cl_truth_ring_* family up front from the schema), which is what keeps a
        full-dataset MLP run within memory.

        Args:
            available (List[str]): Column names present in the dataset files.

        Returns:
            List[str]: The ring columns this model trains on.
        """
        return self.preprocessor.resolve_from_available(available)

    def build_model_kwargs(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Derives the MLP input dimension from the preprocessed feature matrix.

        Args:
            X (np.ndarray): Preprocessed training features, shape (N, n_features).

        Returns:
            Dict[str, Any]: {'input_dim': n_features}.
        """
        input_dim = int(X.shape[1])
        logger.info(f"📐 Model input dimension: {input_dim}")
        return {"input_dim": input_dim}
