import numpy as np
import pandas as pd
import logging
from typing import Optional, List

from ai.preprocess.base import BasePreprocessor

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


class PreprocessMLP(BasePreprocessor):
    """
    Preprocessor for the MLP pipeline: extracts the leading half of each calorimeter layer's
    ring features (50 of 100 rings, see _selected_ring_columns), cleans sensor anomalies and
    normalises each event by its own total ring energy.

    Stateless: it selects, cleans and normalises each event by its own total ring energy, all
    pure functions of the input, so there is nothing to fit.
    """

    def __init__(self) -> None:
        """
        Initializes PreprocessMLP instance.
        """
        self.ring_columns = _selected_ring_columns("cl_ring_%i")

    def required_columns(self, available: List[str]) -> List[str]:
        """
        The 50 selected ring columns, resolved against the dataset schema. Restricting the
        parquet scan to these is what keeps a full-dataset MLP run within memory.

        Args:
            available (List[str]): Column names present in the dataset files.

        Returns:
            List[str]: The ring columns this model trains on.
        """
        return self.resolve_from_available(available)

    def resolve_from_available(self, available: List[str]) -> List[str]:
        """
        Determines which ring column family to use given the available column names:
        'cl_ring_*' with a fallback to 'cl_truth_ring_*'. Works from names alone, so the
        pipeline can project the parquet scan down to these columns before any data is read.
        The choice is a deterministic function of the schema, so training and evaluation on
        the same dataset always resolve to the same family.

        Args:
            available (List[str]): Column names present in the dataset.

        Returns:
            List[str]: The 50 ring column names to use.

        Raises:
            ValueError: If neither column family is fully present.
        """
        present = set(available)

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
        Resolves the ring columns against a DataFrame's schema.

        Args:
            df (pd.DataFrame): DataFrame to resolve against.

        Returns:
            List[str]: The resolved ring column names, in ring order.

        Raises:
            ValueError: If neither column family is fully present.
        """
        return self.resolve_from_available(list(df.columns))

    def _extract(self, df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        """
        Extracts and cleans the ring matrix: NaNs and the -999 sensor-anomaly marker are
        zeroed. Deterministic and stateless.

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

        return X

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms input DataFrame into the normalised feature matrix.

        Args:
            df (pd.DataFrame): Input DataFrame containing ring feature columns.

        Returns:
            np.ndarray: Float32 ring matrix, each event normalised by its own total.
        """
        cols = self.resolve_columns(df)
        logger.info(f"🧪 Extracting {len(cols)} ring features ({cols[0]} ... {cols[-1]})...")
        return self.normalize(self._extract(df, cols))

    def get_labels(self, df: pd.DataFrame, label_col: str = 'label') -> Optional[np.ndarray]:
        """
        Extracts target labels from DataFrame. Overrides the base only to keep the historical
        fallback column names; everything else about persistence and labelling is inherited.

        Args:
            df (pd.DataFrame): Input DataFrame.
            label_col (str): Label column name. Defaults to 'label'.

        Returns:
            Optional[np.ndarray]: Float32 numpy array of labels or None if column not found.
        """
        if label_col in df.columns:
            return df[label_col].values.astype(np.float32)

        # Fallback check for common label column names
        for fallback in ['has_truth_clus', 'target']:
            if fallback in df.columns:
                logger.info(f"ℹ️ Label column '{label_col}' not found, using fallback column '{fallback}'.")
                return df[fallback].values.astype(np.float32)

        logger.warning(f"⚠️ Label column '{label_col}' not found in DataFrame.")
        return None
