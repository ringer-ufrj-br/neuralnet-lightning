import logging
from typing import List, Optional

import numpy as np
import polars as pl

from ai.preprocess.base import BasePreprocessor

logger = logging.getLogger(__name__)

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

    def layer_cells(self, df: pl.DataFrame, col: str) -> np.ndarray:
        """
        One layer as an (N, h, w) array. Every event carries the same grid for a given layer
        (3x3 presampler, 3x15 EM1, 7x7 EM2, ...), so polars converts the nested lists into one
        fixed-size buffer and no per-event Python object is ever created.
        """
        first = df.get_column(col).head(1).to_list()[0]
        h, w = len(first), len(first[0])
        if h > self.target_shape[1] or w > self.target_shape[2]:
            raise ValueError(f"❌ Column '{col}' is {h}x{w}, larger than the "
                             f"{self.target_shape[1]}x{self.target_shape[2]} target image.")
        try:
            grid = df.select(pl.col(col).list.eval(pl.element().list.to_array(w)).list.to_array(h))
        except pl.exceptions.PolarsError as exc:
            raise ValueError(f"❌ Column '{col}' is not the same {h}x{w} grid on every event.") from exc
        return grid.to_series().to_numpy()

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        """Builds the cell images and normalises each one by its own total."""
        return self.normalize(self.build_images(df))

    def build_images(self, df: pl.DataFrame, out: Optional[np.ndarray] = None) -> np.ndarray:
        """
        The (N, C, H, W) cell images, unnormalised. Separate from `transform` so PreprocessFused
        can take the raw images and normalise once over the concatenated rings+cells vector.

        Each layer is written centred into its channel of one preallocated array, and cleaned
        there in place: -999 sensor anomalies zeroed, then log1p of the energies clipped at zero.
        The padding stays zero. `out`, a zeroed (N, C, H, W) float32 array or view, is filled
        instead of a new array when given.
        """
        _, height, width = self.target_shape
        X = np.zeros((df.height, *self.target_shape), dtype=np.float32) if out is None else out

        logger.info("🖼️ Converting calorimeter layers to 2D image tensors...")
        for i, col in enumerate(CELL_COLUMNS):
            logger.info(f"⚡ [{i+1}/{len(CELL_COLUMNS)}] Processing channel: {col}")
            cells = self.layer_cells(df, col)
            h, w = cells.shape[1:]
            top, left = (height - h) // 2, (width - w) // 2
            layer = X[:, i, top:top + h, left:left + w]
            layer[...] = cells
            del cells
            layer[layer == -999] = 0.0
            np.clip(layer, 0, None, out=layer)
            np.log1p(layer, out=layer)
        return X

    def required_columns(self, available: List[str]) -> List[str]:
        """The cell-image columns, leaving the ring/shower-shape columns unread."""
        return list(CELL_COLUMNS)
