"""The two-branch input: the rings followed by the flattened cell image, in one flat vector."""

import logging
import math
from typing import Any, Dict, List

import numpy as np
import polars as pl

from neuralnet_lightning.data.schema import N_RINGS, ring_name
from neuralnet_lightning.preprocess.base import BasePreprocessor
from neuralnet_lightning.preprocess.cnn2d import CELL_COLUMNS, PreprocessCNN2D

logger = logging.getLogger(__name__)


class PreprocessFused(BasePreprocessor):
    """One vector per event: the norm1 rings followed by the flattened cell image."""

    def __init__(self) -> None:
        self.cells_pp = PreprocessCNN2D()
        self.ring_columns = [ring_name(i) for i in range(N_RINGS)]

    def required_columns(self) -> List[str]:
        """Both branches' columns: the rings plus the cell images."""
        return self.ring_columns + list(CELL_COLUMNS)

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        """Float32 array of shape (N, n_rings + C*H*W)."""
        n_rings = len(self.ring_columns)
        cell_shape = self.cells_pp.target_shape
        X = np.zeros((df.height, n_rings + math.prod(cell_shape)), dtype=np.float32)

        rings = self.extract(df, self.ring_columns)
        total = np.abs(rings).sum(axis=1, keepdims=True)
        np.divide(rings, total, out=X[:, :n_rings], where=total > 0)

        # copy=False: the images must be written into X.
        self.cells_pp.build_images(df, out=X[:, n_rings:].reshape((df.height, *cell_shape), copy=False))

        logger.info(f"🔗 Fused features: {X.shape} ({n_rings} rings + {X.shape[1] - n_rings} cells)")
        return self.normalize(X)

    def model_kwargs(self, X: np.ndarray) -> Dict[str, Any]:
        """Ring count and cell-image shape, used by the model to split the input."""
        return {"n_rings": len(self.ring_columns), "cell_shape": tuple(self.cells_pp.target_shape)}
