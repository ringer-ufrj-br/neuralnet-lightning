import os
import logging
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BasePreprocessor:
    """
    The contract every preprocessor honours, plus the parts that are the same for all of them
    (persistence, label extraction, fit_transform).

    The baseline preprocessor is a column selection: set `feature_columns` and the inherited
    `required_columns` / `transform` do the rest - extract those columns, zero the sensor
    anomalies, normalise each event by its own total. PreprocessMLP is exactly this.

    A preprocessor whose input is not a flat slice of dataset columns (the CNN2D image
    builder, the Fused rings+cells concatenation) instead overrides `transform`, and usually
    `required_columns`, and leaves `feature_columns` as None.

    Either kind overrides `fit` only if it has state to learn from the training split (a
    scaler, a mean, ...). The default `fit` is a no-op.

    Persistence is joblib pickling of the whole instance, so anything stored on `self` in
    `fit` is restored by `load` - no per-preprocessor save/load code is needed.
    """

    #: Dataset columns this preprocessor consumes, in feature order. When set, it drives the
    #: default `required_columns` and `transform`. Left None by preprocessors that build their
    #: input some other way and override `transform`.
    feature_columns: Optional[List[str]] = None

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Normalises each sample by the absolute sum of its own features, so the network sees the
        shape of the energy deposition and not its absolute scale (the Et binning already
        accounts for scale). This is the NeuralRinger reference normalisation,
        r'_i = r_i / |sum_j r_j|.

        Call this as the last step of `transform`. It is done here, once per dataset, rather
        than inside the model, where it would be recomputed for every batch of every epoch on
        data that never changes.

        Args:
            X (np.ndarray): Feature array, first dimension being the batch.

        Returns:
            np.ndarray: Float32 array of the same shape, each sample scaled by its own total.
                Samples summing to zero are left as they are rather than turned into NaNs.
        """
        axes = tuple(range(1, X.ndim))
        if not axes:
            return X.astype(np.float32)
        total = np.abs(X.sum(axis=axes, keepdims=True))
        return (X / np.where(total == 0.0, 1.0, total)).astype(np.float32)

    def required_columns(self, available: List[str]) -> Optional[List[str]]:
        """
        Declares which dataset columns this preprocessor consumes, so the loader can prune the
        rest during the parquet scan. The raw files carry 300+ columns, most of them nested
        calorimeter images; reading them all is what used to exhaust memory.

        The default returns `feature_columns` (None when unset, i.e. load everything).

        Args:
            available (List[str]): Column names present in the dataset files.

        Returns:
            Optional[List[str]]: Columns to load, or None to load everything.
        """
        return self.feature_columns

    def extract(self, df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        """
        Pulls `cols` out of `df` as a clean float32 matrix: NaNs and the -999 sensor-anomaly
        marker are zeroed. A missing column raises KeyError here rather than being silently
        worked around - a wrong column set is a bug, not something to recover from.

        Args:
            df (pd.DataFrame): Input rows.
            cols (List[str]): Column names to extract, in order.

        Returns:
            np.ndarray: Cleaned float32 array, first dimension being the batch. Not normalised.
        """
        X = df[cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0)
        return np.where(X == -999, 0.0, X)

    def fit(self, df: pd.DataFrame) -> "BasePreprocessor":
        """
        Learns whatever state this preprocessor needs from the training split. The default is
        a no-op, for preprocessors that are pure functions of their input.

        Args:
            df (pd.DataFrame): Training rows.

        Returns:
            BasePreprocessor: self, for chaining.
        """
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Turns a DataFrame into the model's input array.

        The default is the baseline path: extract `feature_columns`, clean sensor anomalies
        and normalise each event by its own total. Preprocessors that build their input some
        other way override this; those must leave `feature_columns` as None.

        Args:
            df (pd.DataFrame): Rows to transform.

        Returns:
            np.ndarray: Float32 features, first dimension being the batch.
        """
        if self.feature_columns is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set feature_columns or override transform()."
            )
        cols = self.feature_columns
        logger.info(f"🧪 Extracting {len(cols)} features ({cols[0]} ... {cols[-1]})...")
        return self.normalize(self.extract(df, cols))

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fits on the given rows and transforms them in one call.

        Args:
            df (pd.DataFrame): Training rows.

        Returns:
            np.ndarray: The transformed features.
        """
        return self.fit(df).transform(df)

    def save(self, filepath: str) -> str:
        """
        Persists the fitted preprocessor alongside the trained checkpoints.

        Args:
            filepath (str): Destination path (.joblib).

        Returns:
            str: The written path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"💾 Saved preprocessor to: {filepath}")
        return filepath

    @staticmethod
    def load(filepath: str) -> "BasePreprocessor":
        """
        Loads a preprocessor previously written by save().

        Args:
            filepath (str): Path to the .joblib file.

        Returns:
            BasePreprocessor: The restored instance.
        """
        preprocessor = joblib.load(filepath)
        logger.info(f"📂 Loaded preprocessor from: {filepath}")
        return preprocessor

    def get_labels(self, df: pd.DataFrame, label_col: str = 'label') -> Optional[np.ndarray]:
        """
        Extracts target labels from the DataFrame.

        Args:
            df (pd.DataFrame): Input rows.
            label_col (str): Label column name. Defaults to 'label'.

        Returns:
            Optional[np.ndarray]: Float32 labels, or None when the column is absent.
        """
        if label_col in df.columns:
            return df[label_col].values.astype(np.float32)
        logger.warning(f"⚠️ Label column '{label_col}' not found in DataFrame.")
        return None
