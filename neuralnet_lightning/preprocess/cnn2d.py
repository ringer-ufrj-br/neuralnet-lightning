"""Calorimeter cell energies as multi-channel 2D images, one channel per layer."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

from neuralnet_lightning.preprocess.base import BasePreprocessor

logger = logging.getLogger(__name__)

#: Cell-image source columns (mc25 layout), one per calorimeter layer.
CELL_COLUMNS: List[str] = [
    "cl_cells_presampler",
    "cl_cells_em1",
    "cl_cells_em2",
    "cl_cells_em3",
    "cl_cells_had1",
    "cl_cells_had2",
    "cl_cells_had3",
]


class PreprocessCNN2D(BasePreprocessor):
    """Calorimeter cell energies as (C, H, W) images, one channel per layer."""

    #: (channels, height, width) every event's image is padded to.
    target_shape = (len(CELL_COLUMNS), 7, 15)

    def required_columns(self) -> List[str]:
        """The cell-image columns."""
        return list(CELL_COLUMNS)

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        """Cell images, each normalised by its own total."""
        return self.normalize(self.build_images(df))

    def model_kwargs(self, X: np.ndarray) -> Dict[str, Any]:
        """Image geometry for the network."""
        channels, height, width = self.target_shape
        return {"in_channels": channels, "cell_height": height, "cell_width": width}

    def layer_cells(self, df: pl.DataFrame, col: str) -> np.ndarray:
        """One layer as an (N, h, w) array; every event must have the same grid."""
        first = df[col][0].to_list()
        h, w = len(first), len(first[0])
        if h > self.target_shape[1] or w > self.target_shape[2]:
            raise ValueError(f"❌ Column '{col}' is {h}x{w}, larger than the "
                             f"{self.target_shape[1]}x{self.target_shape[2]} target image.")
        try:
            grid = df.select(pl.col(col).list.eval(pl.element().list.to_array(w)).list.to_array(h))
        except pl.exceptions.PolarsError as exc:
            raise ValueError(f"❌ Column '{col}' is not the same {h}x{w} grid on every event.") from exc
        return grid.to_series().to_numpy()

    def build_images(self, df: pl.DataFrame, out: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Unnormalised (N, C, H, W) images: each layer centred in its channel, -999 set to zero,
        then log1p of the energies clipped at zero. Fills `out` (zeroed, float32) when given.
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
