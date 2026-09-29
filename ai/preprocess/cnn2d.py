import logging
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from ai.preprocess.base import BasePreprocessor

logger = logging.getLogger(__name__)
tqdm.pandas(desc="Processing Samples")

#: Cell-image columns, one per calorimeter layer.
CELL_COLUMNS: List[str] = [
    'cl_cells_presampler',
    'cl_cells_em1',
    'cl_cells_em2',
    'cl_cells_em3',
    'cl_cells_had1',
    'cl_cells_had2',
    'cl_cells_had3',
]


class PreprocessCNN2D(BasePreprocessor):
    """
    Formats calorimeter cell energies into multi-channel 2D images, one channel per layer.
    Deterministic (padding + log1p), so the inherited no-op `fit` is all it needs.
    """

    #: (channels, height, width) every event's image is padded to.
    target_shape = (len(CELL_COLUMNS), 7, 15)

    def pad_array(self, arr: np.ndarray) -> np.ndarray:
        """One layer: -999 sensor anomalies zeroed, log1p of the clipped energies, zero-padded
        around the centre to the target height x width."""
        arr = np.stack(arr).astype(np.float32)
        arr = np.log1p(np.clip(np.where(arr == -999, 0, arr), 0, None))
        dh, dw = self.target_shape[1] - arr.shape[0], self.target_shape[2] - arr.shape[1]
        return np.pad(arr, ((dh // 2, dh - dh // 2), (dw // 2, dw - dw // 2)))

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Builds the cell images and normalises each one by its own total."""
        return self.normalize(self.build_images(df))

    def build_images(self, df: pd.DataFrame) -> np.ndarray:
        """
        The (N, C, H, W) cell images, unnormalised. Separate from `transform` so PreprocessFused
        can take the raw images and normalise once over the concatenated rings+cells vector.
        """
        logger.info("🖼️ Converting calorimeter layers to 2D image tensors...")
        layers = []
        for i, col in enumerate(CELL_COLUMNS):
            logger.info(f"⚡ [{i+1}/{len(CELL_COLUMNS)}] Processing channel: {col}")
            layers.append(np.stack(df[col].progress_apply(self.pad_array).values))
        return np.stack(layers, axis=1)

    def required_columns(self, available: List[str]) -> List[str]:
        """The cell-image columns, leaving the ring/shower-shape columns unread."""
        return list(CELL_COLUMNS)
