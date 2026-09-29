import logging
from typing import List

import numpy as np
import pandas as pd

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

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Float32 array of shape (N, n_rings + C*H*W)."""
        rings = self.extract(df, self.ring_columns)
        total = np.abs(rings).sum(axis=1, keepdims=True)
        rings = np.divide(rings, total, out=np.zeros_like(rings), where=total > 0)
        cells = self.cells_pp.build_images(df).reshape(len(df), -1)

        X = np.concatenate([rings, cells], axis=1).astype(np.float32)
        logger.info(f"🔗 Fused features: {X.shape} ({rings.shape[1]} rings + {cells.shape[1]} cells)")
        return self.normalize(X)

    def required_columns(self, available: List[str]) -> List[str]:
        """Both branches' columns: the rings plus the cell images."""
        return self.ring_columns + list(CELL_COLUMNS)
