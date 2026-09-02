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

    A new preprocessor implements TWO methods:

        required_columns(available) - which dataset columns to read from the parquet files
        transform(df)               - DataFrame -> float32 feature array

    and, only if it has state to learn from the training split (a scaler, a mean, ...),
    overrides `fit`. The default `fit` is a no-op, which is correct for a stateless
    preprocessor such as the CNN2D image builder.

    Persistence is joblib pickling of the whole instance, so anything stored on `self` in
    `fit` is restored by `load` - no per-preprocessor save/load code is needed.
    """

    def required_columns(self, available: List[str]) -> Optional[List[str]]:
        """
        Declares which dataset columns this preprocessor consumes, so the loader can prune the
        rest during the parquet scan. The raw files carry 300+ columns, most of them nested
        calorimeter images; reading them all is what used to exhaust memory.

        Args:
            available (List[str]): Column names present in the dataset files.

        Returns:
            Optional[List[str]]: Columns to load, or None to load everything.
        """
        return None

    def fit(self, df: pd.DataFrame) -> "BasePreprocessor":
        """
        Learns whatever state this preprocessor needs from the training split. The default is
        a no-op, for preprocessors that are pure functions of their input.

        IMPORTANT: only ever call this on training data. Fitting on the holdout leaks it.

        Args:
            df (pd.DataFrame): Training rows.

        Returns:
            BasePreprocessor: self, for chaining.
        """
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Turns a DataFrame into the model's input array. MUST be implemented.

        Args:
            df (pd.DataFrame): Rows to transform.

        Returns:
            np.ndarray: Float32 features, first dimension being the batch.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement transform() and return a float32 array."
        )

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
