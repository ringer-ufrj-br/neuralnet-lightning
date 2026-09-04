import numpy as np
import pandas as pd
import logging
from typing import Tuple, List

from .cnn2d import PreprocessCNN2D
from .base import BasePreprocessor

logger = logging.getLogger(__name__)

class PreprocessFused(BasePreprocessor):
    """
    Preprocessor for the Fused pipeline, combining ring features and calorimeter
    cell images into a single feature array.
    """

    def __init__(self, num_rings: int = 100, ring_norm: str = 'norm1') -> None:
        """
        Initializes PreprocessFused instance.
        Args:
            num_rings (int): Number of ring features to extract. Defaults to 100.
            ring_norm (str): Ring normalization: 'norm1', 'log' or None. Defaults to 'norm1'.
        """
        self.cells_pp = PreprocessCNN2D()
        self.num_rings = num_rings
        self.ring_norm = ring_norm
        self.ring_columns = [f"cl_ring_{i}" for i in range(self.num_rings)]

    def process_rings(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extracts and normalizes the ring features.
        Args:
            df (pd.DataFrame): Input DataFrame containing ring feature columns.
        Returns:
            np.ndarray: Processed float32 array of shape (N, num_rings).
        """
        cols = self.ring_columns
        logger.info(f"🧪 Extracting {len(cols)} ring features ({cols[0]} ... {cols[-1]})...")
        X = self.extract(df, cols)

        if self.ring_norm == 'norm1':
            # Ringer standard: divide by the total ring energy, removing the
            # absolute energy dependence and keeping the shower profile
            total = np.abs(X).sum(axis=1, keepdims=True)
            X = np.divide(X, total, out=np.zeros_like(X), where=total > 0)
        elif self.ring_norm == 'log':
            X = np.log1p(np.clip(X, 0, None))

        return X.astype(np.float32)

    def transform_split(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms the DataFrame into separate ring and cell arrays.
        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            Tuple[np.ndarray, np.ndarray]: Rings of shape (N, num_rings) and cells of shape (N, 7, 7, 15).
        """
        return self.process_rings(df), self.cells_pp.build_images(df)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms the DataFrame into a single array with rings and flattened cells.
        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            np.ndarray: Float32 array of shape (N, num_rings + C*H*W).
        """
        X_rings, X_cells = self.transform_split(df)
        X_cells_flat = X_cells.reshape(X_cells.shape[0], -1)

        X = np.concatenate([X_rings, X_cells_flat], axis=1).astype(np.float32)
        logger.info(f"🔗 Fused features: {X.shape} ({X_rings.shape[1]} rings + {X_cells_flat.shape[1]} cells)")
        return self.normalize(X)

    def required_columns(self, available: List[str]) -> List[str]:
        """
        Both branches' columns: the ring columns this model reads plus the 7 cell-image
        columns the CNN branch needs.

        Args:
            available (List[str]): Column names present in the dataset files.

        Returns:
            List[str]: Ring columns plus the cell columns.
        """
        return list(self.ring_columns) + self.cells_pp.required_columns(available)
