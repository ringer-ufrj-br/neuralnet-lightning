import logging
from typing import List

import numpy as np
import polars as pl

from .base import BasePreprocessor, N_RINGS, ring_name
from .cnn2d import CELL_COLUMNS, PreprocessCNN2D

logger = logging.getLogger(__name__)


class PreprocessFused(BasePreprocessor):
    """
    Preprocessor for the Fused pipeline: one flat vector per event, the rings (each event
    divided by its total ring energy, the Ringer norm1) followed by the flattened cell image.
    """

    def __init__(self) -> None:
        self.cells_pp = PreprocessCNN2D()
        self.ring_columns = [ring_name(i) for i in range(N_RINGS)]

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        """
        Float32 array of shape (N, n_rings + C*H*W). The cell images are written straight into
        their slice of the one output array, so the large half is never built separately and
        concatenated.
        """
        n_rings = len(self.ring_columns)
        cell_shape = tuple(self.cells_pp.target_shape)
        X = np.zeros((df.height, n_rings + int(np.prod(cell_shape))), dtype=np.float32)

        rings = self.extract(df, self.ring_columns)
        total = np.abs(rings).sum(axis=1, keepdims=True)
        np.divide(rings, total, out=X[:, :n_rings], where=total > 0)

        # copy=False: a view is required, or the images would land in a discarded copy.
        self.cells_pp.build_images(df, out=X[:, n_rings:].reshape((df.height, *cell_shape), copy=False))

        logger.info(f"🔗 Fused features: {X.shape} ({n_rings} rings + {X.shape[1] - n_rings} cells)")
        return self.normalize(X)

    def required_columns(self, available: List[str]) -> List[str]:
        """Both branches' columns: the rings plus the cell images."""
        return self.ring_columns + list(CELL_COLUMNS)
