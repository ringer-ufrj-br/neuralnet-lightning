"""Base preprocessor: canonical DataFrame in, model input array out."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class BasePreprocessor:
    """
    Base class of the preprocessors.

    Setting `feature_columns` is enough for a column selection: the default `required_columns`
    and `transform` extract those columns, zero sensor anomalies and normalise each event. Other
    inputs override `required_columns` and `transform`. `fit` is overridden only when there is
    state to learn; the fitted instance is saved with joblib.
    """

    #: Canonical columns consumed, in feature order.
    feature_columns: Optional[List[str]] = None

    def required_columns(self) -> List[str]:
        """Canonical columns to load."""
        if self.feature_columns is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set feature_columns or override required_columns()."
            )
        return list(self.feature_columns)

    def fit(self, df: pl.DataFrame) -> "BasePreprocessor":
        """Learns whatever state this preprocessor needs. No-op here."""
        return self

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        """Model input array. Default: `feature_columns`, cleaned and normalised."""
        cols = self.required_columns()
        logger.info(f"🧪 Extracting {len(cols)} features ({cols[0]} ... {cols[-1]})...")
        return self.normalize(self.extract(df, cols))

    def fit_transform(self, df: pl.DataFrame) -> np.ndarray:
        """Fits on the given rows and transforms them in one call."""
        return self.fit(df).transform(df)

    def model_kwargs(self, X: np.ndarray) -> Dict[str, Any]:
        """Model constructor kwargs derived from the preprocessed data, e.g. the input dimension."""
        return {}

    # ---------------------------------------------------------------- helpers

    def extract(self, df: pl.DataFrame, cols: List[str]) -> np.ndarray:
        """`cols` as a float32 matrix, with NaN and the -999 sensor-anomaly marker set to zero."""
        X = np.empty((df.height, len(cols)), dtype=np.float32, order="F")
        for j, name in enumerate(cols):
            column = X[:, j]
            column[...] = df.get_column(name).to_numpy()
            np.nan_to_num(column, copy=False, nan=0.0)
            column[column == -999] = 0.0
        return X

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """
        norm1: divides each sample, in place, by the absolute sum of its features,
        r'_i = r_i / |sum_j r_j|. Samples summing to zero are left unchanged.
        """
        X = np.asarray(X, dtype=np.float32)
        axes = tuple(range(1, X.ndim))
        if not axes:
            return X
        total = np.abs(X.sum(axis=axes, keepdims=True))
        total[total == 0.0] = 1.0
        X /= total
        return X
