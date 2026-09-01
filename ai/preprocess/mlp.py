import os
import numpy as np
import pandas as pd
import logging
import joblib
from typing import Optional, List
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _selected_ring_columns(prefix: str) -> List[str]:
    """
    Selected ring columns for MLP training - we selected 1/2 of rings in each layer (fixed,
    not parameterized). Mirrors the reference selection from prior Ringer trainings:

    pre-sample - 8 rings
    EM1 - 64 rings
    EM2 - 8 rings
    EM3 - 8 rings
    Had1 - 4 rings
    Had2 - 4 rings
    Had3 - 4 rings

    Args:
        prefix (str): printf-style column name template with one '%i' placeholder, e.g. 'cl_ring_%i'.

    Returns:
        List[str]: The 50 selected column names, in ring order.
    """
    # rings presample
    presample = [prefix % iring for iring in range(8 // 2)]

    # EM1 list
    sum_rings = 8
    em1 = [prefix % iring for iring in range(sum_rings, sum_rings + (64 // 2))]

    # EM2 list
    sum_rings = 8 + 64
    em2 = [prefix % iring for iring in range(sum_rings, sum_rings + (8 // 2))]

    # EM3 list
    sum_rings = 8 + 64 + 8
    em3 = [prefix % iring for iring in range(sum_rings, sum_rings + (8 // 2))]

    # HAD1 list
    sum_rings = 8 + 64 + 8 + 8
    had1 = [prefix % iring for iring in range(sum_rings, sum_rings + (4 // 2))]

    # HAD2 list
    sum_rings = 8 + 64 + 8 + 8 + 4
    had2 = [prefix % iring for iring in range(sum_rings, sum_rings + (4 // 2))]

    # HAD3 list
    sum_rings = 8 + 64 + 8 + 8 + 4 + 4
    had3 = [prefix % iring for iring in range(sum_rings, sum_rings + (4 // 2))]

    return presample + em1 + em2 + em3 + had1 + had2 + had3


class PreprocessMLP:
    """
    Preprocessor for MLP pipeline extracting the leading half of each calorimeter layer's
    ring features (50 of 100 rings, see _selected_ring_columns) from a DataFrame, applying
    cleaning, optional log-scale transformations, and StandardScaler normalization.

    Follows the scikit-learn fit/transform contract: `fit` must see the **training rows only**
    and `transform` is then applied to train and test alike. The previous single-method version
    called `fit_transform` on the full dataset, which both leaked holdout statistics into the
    normalisation and made the fitted state impossible to reuse in a separate evaluation run.
    """

    def __init__(
        self,
        use_scaler: bool = True,
        apply_log1p: bool = True
    ) -> None:
        """
        Initializes PreprocessMLP instance.

        Args:
            use_scaler (bool): Whether to apply StandardScaler normalization. Defaults to True.
            apply_log1p (bool): Whether to apply log1p transformation to energy values. Defaults to True.
        """
        self.use_scaler = use_scaler
        self.apply_log1p = apply_log1p
        self.ring_columns = _selected_ring_columns("cl_ring_%i")
        self.scaler = StandardScaler() if use_scaler else None
        self.fitted_columns: Optional[List[str]] = None
        self.is_fitted = False

    def resolve_from_available(self, available: List[str]) -> List[str]:
        """
        Determines which ring column family to use given the available column names:
        'cl_ring_*' with a fallback to 'cl_truth_ring_*'. Once fitted, the resolved family
        is pinned so that evaluation cannot silently switch to the other one. Works from
        names alone so the pipeline can project the parquet scan down to these columns
        before any data is read.

        Args:
            available (List[str]): Column names present in the dataset.

        Returns:
            List[str]: The 50 ring column names to use.

        Raises:
            ValueError: If neither column family is fully present.
        """
        present = set(available)

        if self.fitted_columns is not None:
            missing = [col for col in self.fitted_columns if col not in present]
            if missing:
                raise ValueError(f"❌ DataFrame is missing the columns this preprocessor was fitted on: {missing}")
            return self.fitted_columns

        if all(col in present for col in self.ring_columns):
            return self.ring_columns

        fallback_cols = _selected_ring_columns("cl_truth_ring_%i")
        if all(col in present for col in fallback_cols):
            logger.info("ℹ️ Ring columns 'cl_ring_*' not found; using fallback 'cl_truth_ring_*'.")
            return fallback_cols

        missing = [col for col in self.ring_columns if col not in present]
        logger.error(f"❌ Missing ring columns in DataFrame: {missing}")
        raise ValueError(f"❌ Missing ring columns in DataFrame: {missing}")

    def resolve_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Determines which ring column family this DataFrame carries (see resolve_from_available).

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            List[str]: The 50 ring column names to use.

        Raises:
            ValueError: If neither column family is fully present.
        """
        return self.resolve_from_available(list(df.columns))

    def _extract(self, df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        """
        Extracts and cleans the ring matrix: sensor anomalies removed, negative noise clipped
        and (optionally) log1p-compressed. Deterministic and stateless - everything that has to
        be learned from the training set lives in the scaler.

        Args:
            df (pd.DataFrame): Input DataFrame.
            cols (List[str]): Ring column names to extract.

        Returns:
            np.ndarray: Cleaned float32 feature array, before normalization.
        """
        X = df[cols].values.astype(np.float32)

        # Handle sensor anomalies represented as -999 or NaNs
        X = np.nan_to_num(X, nan=0.0)
        X = np.where(X == -999, 0.0, X)

        # Clip negative energy noise values and apply log1p transformation to compress tails
        if self.apply_log1p:
            X = np.log1p(np.clip(X, 0, None))

        return X

    def fit(self, df: pd.DataFrame) -> "PreprocessMLP":
        """
        Fits the normalization statistics on the given (training) rows.

        Args:
            df (pd.DataFrame): Training rows only.

        Returns:
            PreprocessMLP: self, for chaining.
        """
        cols = self.resolve_columns(df)
        X = self._extract(df, cols)

        if self.scaler is not None:
            logger.info(f"📐 Fitting StandardScaler on {len(X)} training rows...")
            self.scaler.fit(X)

        self.fitted_columns = cols
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms input DataFrame into processed numpy feature matrix.

        Args:
            df (pd.DataFrame): Input DataFrame containing ring feature columns.

        Returns:
            np.ndarray: Processed float32 feature array.

        Raises:
            RuntimeError: If a scaler is configured but has not been fitted yet.
        """
        if self.use_scaler and not self.is_fitted:
            raise RuntimeError("❌ PreprocessMLP.transform called before fit(). Call fit() or fit_transform() first.")

        cols = self.resolve_columns(df)
        logger.info(f"🧪 Extracting {len(cols)} ring features ({cols[0]} ... {cols[-1]})...")
        X = self._extract(df, cols)

        if self.scaler is not None:
            X = self.scaler.transform(X).astype(np.float32)

        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fits on the given rows and transforms them in one call.

        Args:
            df (pd.DataFrame): Training rows only.

        Returns:
            np.ndarray: Processed float32 feature array.
        """
        return self.fit(df).transform(df)

    def save(self, filepath: str) -> str:
        """
        Persists the fitted preprocessor so a later evaluation run reproduces the exact
        same input transformation without touching the training data again.

        Args:
            filepath (str): Destination path (.joblib).

        Returns:
            str: The written path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"💾 Saved fitted preprocessor to: {filepath}")
        return filepath

    @staticmethod
    def load(filepath: str) -> "PreprocessMLP":
        """
        Loads a preprocessor previously written by save().

        Args:
            filepath (str): Path to the .joblib file.

        Returns:
            PreprocessMLP: The restored instance.
        """
        preprocessor = joblib.load(filepath)
        logger.info(f"📂 Loaded fitted preprocessor from: {filepath}")
        return preprocessor

    def get_labels(self, df: pd.DataFrame, label_col: str = 'label') -> Optional[np.ndarray]:
        """
        Extracts target labels from DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.
            label_col (str): Label column name. Defaults to 'label'.

        Returns:
            Optional[np.ndarray]: Float32 numpy array of labels or None if column not found.
        """
        if label_col in df.columns:
            return df[label_col].values.astype(np.float32)

        # Fallback check for common label column names
        for fallback in ['label', 'has_truth_clus', 'target']:
            if fallback in df.columns:
                logger.info(f"ℹ️ Label column '{label_col}' not found, using fallback column '{fallback}'.")
                return df[fallback].values.astype(np.float32)

        logger.warning(f"⚠️ Label column '{label_col}' not found in DataFrame.")
        return None
