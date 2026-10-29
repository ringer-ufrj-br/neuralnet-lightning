"""
Preprocessing: canonical DataFrame in, model input array out.

A preprocessor never sees a dataset column name - only the canonical vocabulary of
neuralnet_lightning.data.schema - and never filters rows: what holds for every model belongs to
the data layer (neuralnet_lightning.data), what is specific to one input representation belongs
here. The one exception is a source column no dataset-wide name exists for yet, such as the
mc25 cell images PreprocessCNN2D reads.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class BasePreprocessor:
    """
    The contract every preprocessor honours, plus the parts that are the same for all of them.

    The baseline preprocessor is a column selection: set `feature_columns` and the inherited
    `required_columns` / `transform` do the rest - extract those columns, zero the sensor
    anomalies, normalise each event. PreprocessMLP is exactly this.

    A preprocessor whose input is not a flat slice of columns overrides `transform` and
    `required_columns` instead, and leaves `feature_columns` as None.

    Either kind overrides `fit` only if it has state to learn (a scaler, a mean, ...). `fit`
    sees the whole region, before the fold split, so that state includes every fold's
    validation rows. The pipeline persists the whole fitted instance with joblib, so anything
    stored on `self` in `fit` is restored for evaluation with no extra code.

    The frames are polars end to end: converting to pandas would copy the whole frame, briefly
    holding it twice. The feature matrix is built once and every later step works on it in
    place, so the peak is the frame plus one matrix.
    """

    #: Canonical columns this preprocessor consumes, in feature order. When set, it drives the
    #: default `required_columns` and `transform`.
    feature_columns: Optional[List[str]] = None

    def required_columns(self) -> List[str]:
        """
        Which canonical columns to load, so the scan can prune the rest: the raw files carry
        well over a hundred columns, and reading them all is what exhausts memory.
        """
        if self.feature_columns is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set feature_columns or override required_columns()."
            )
        return list(self.feature_columns)

    def fit(self, df: pl.DataFrame) -> "BasePreprocessor":
        """Learns whatever state this preprocessor needs. No-op here."""
        return self

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        """
        The model's input array. The default is the baseline path: extract `feature_columns`,
        clean sensor anomalies and normalise each event.
        """
        cols = self.required_columns()
        logger.info(f"🧪 Extracting {len(cols)} features ({cols[0]} ... {cols[-1]})...")
        return self.normalize(self.extract(df, cols))

    def fit_transform(self, df: pl.DataFrame) -> np.ndarray:
        """Fits on the given rows and transforms them in one call."""
        return self.fit(df).transform(df)

    def model_kwargs(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Model constructor kwargs only known once the data is preprocessed - typically the input
        dimension. Merged under the config's model_args, which win on a clash.
        """
        return {}

    # ---------------------------------------------------------------- helpers

    def extract(self, df: pl.DataFrame, cols: List[str]) -> np.ndarray:
        """
        `cols` as a float32 matrix with NaNs and the -999 sensor-anomaly marker zeroed (±inf
        clipped to the float32 range, as np.nan_to_num does). Column-major, filled one column
        at a time, so no intermediate copy of the frame is ever built.
        """
        X = np.empty((df.height, len(cols)), dtype=np.float32, order="F")
        for j, name in enumerate(cols):
            column = X[:, j]
            column[...] = df.get_column(name).to_numpy()
            np.nan_to_num(column, copy=False, nan=0.0)
            column[column == -999] = 0.0
        return X

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Normalises each sample by the absolute sum of its own features, so the network sees the
        shape of the energy deposition and not its absolute scale (the Et binning already
        accounts for scale). This is the NeuralRinger reference normalisation (norm1),
        r'_i = r_i / |sum_j r_j|.

        Done here, once per dataset, rather than inside the model, where it would be recomputed
        for every batch of every epoch on data that never changes. Works in place, so `X` must
        be an array this preprocessor owns. Samples summing to zero are left as they are rather
        than turned into NaNs.
        """
        X = np.asarray(X, dtype=np.float32)
        axes = tuple(range(1, X.ndim))
        if not axes:
            return X
        total = np.abs(X.sum(axis=axes, keepdims=True))
        total[total == 0.0] = 1.0
        X /= total
        return X
