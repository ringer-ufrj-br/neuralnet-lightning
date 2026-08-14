import numpy as np
import pandas as pd
import logging
from typing import Optional, List
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class PreprocessMLP:
    """
    Preprocessor for MLP pipeline extracting ring features from DataFrame,
    applying cleaning, optional log-scale transformations, and StandardScaler normalization.
    """

    def __init__(
        self, 
        num_rings: int = 100, 
        use_scaler: bool = True,
        apply_log1p: bool = True
    ) -> None:
        """
        Initializes PreprocessMLP instance.

        Args:
            num_rings (int): Number of ring features to extract. Defaults to 100.
            use_scaler (bool): Whether to apply StandardScaler normalization. Defaults to True.
            apply_log1p (bool): Whether to apply log1p transformation to energy values. Defaults to True.
        """
        self.num_rings = num_rings
        self.use_scaler = use_scaler
        self.apply_log1p = apply_log1p
        self.ring_columns = [f"cl_ring_{i}" for i in range(self.num_rings)]
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
            fallback_cols = [f"cl_truth_ring_{i}" for i in range(self.num_rings)]
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

