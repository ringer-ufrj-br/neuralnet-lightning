import numpy as np
import pandas as pd
import logging
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

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms input DataFrame into processed numpy feature matrix.

        Args:
            df (pd.DataFrame): Input DataFrame containing ring feature columns.

        Returns:
            np.ndarray: Processed float32 feature array.
        """
        # Determine ring columns: check cl_ring_i first, with fallback to cl_truth_ring_i
        cols = self.ring_columns
        if not all(col in df.columns for col in cols):
            fallback_cols = _selected_ring_columns("cl_truth_ring_%i")
            if all(col in df.columns for col in fallback_cols):
                logger.info("ℹ️ Ring columns 'cl_ring_*' not found; using fallback 'cl_truth_ring_*'.")
                cols = fallback_cols
            else:
                missing = [col for col in cols if col not in df.columns]
                logger.error(f"❌ Missing ring columns in DataFrame: {missing}")
                raise ValueError(f"❌ Missing ring columns in DataFrame: {missing}")

        logger.info(f"🧪 Extracting {len(cols)} ring features ({cols[0]} ... {cols[-1]})...")
        X = df[cols].values.astype(np.float32)

        # Handle sensor anomalies represented as -999 or NaNs
        X = np.nan_to_num(X, nan=0.0)
        X = np.where(X == -999, 0.0, X)

        # Clip negative energy noise values and apply log1p transformation to compress tails
        if self.apply_log1p:
            X = np.log1p(np.clip(X, 0, None))

        # Apply StandardScaler normalization across features
        if self.scaler is not None:
            logger.info("📐 Applying StandardScaler normalization...")
            X = self.scaler.fit_transform(X).astype(np.float32)

        return X

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

