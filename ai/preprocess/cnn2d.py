import os
import numpy as np
import pandas as pd
import logging
import joblib
from typing import Tuple, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)
tqdm.pandas(desc="Processing Samples")

class PreprocessCNN2D:
    """
    Preprocessor for 2D Convolutional Neural Networks (CNN2D) that formats calorimeter cell energies into multi-channel 2D image tensors.
    """

    def __init__(self, target_shape: Tuple[int, int, int] = (7, 7, 15)) -> None:
        """
        Initializes PreprocessCNN2D instance.

        Args:
            target_shape (Tuple[int, int, int]): Tensor dimensions (channels, max_height, max_width). Defaults to (7, 7, 15).
        """
        self.target_shape = target_shape
        self.cell_columns = [
            'cl_cells_presampler',
            'cl_cells_em1',
            'cl_cells_em2',
            'cl_cells_em3',
            'cl_cells_had1',
            'cl_cells_had2',
            'cl_cells_had3'
        ]
        self.max_h = 7
        self.max_w = 15

    def pad_array(self, arr: np.ndarray) -> np.ndarray:
        """
        Preprocesses and pads a single 2D calorimeter cell energy layer.

        Args:
            arr (np.ndarray): Input 2D cell energy array.

        Returns:
            np.ndarray: Zero-padded float32 2D array of shape (7, 15).
        """
        arr = np.stack(arr).astype(np.float32)
        
        # Handle sensor anomalies (-999)
        arr = np.where(arr == -999, 0, arr)
        
        # Clip values and apply log1p transformation
        arr = np.log1p(np.clip(arr, 0, None))
        
        h, w = arr.shape
        pad_h = self.max_h - h
        pad_w = self.max_w - w
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        return np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=0)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms DataFrame cell energy columns into multi-channel 2D image tensors.

        Args:
            df (pd.DataFrame): Input DataFrame containing calorimeter cell columns.

        Returns:
            np.ndarray: Multi-channel tensor array of shape (N, 7, 7, 15).
        """
        missing = [col for col in self.cell_columns if col not in df.columns]
        if missing:
            logger.error(f"❌ Missing required cell columns: {missing}")
            raise ValueError(f"❌ Missing required cell columns: {missing}")

        num_samples = len(df)
        num_layers = len(self.cell_columns)
        
        X = np.zeros((num_samples, num_layers, self.max_h, self.max_w), dtype=np.float32)

        logger.info("🖼️ Converting calorimeter layers to 2D image tensors...")
        for i, col in enumerate(self.cell_columns):
            logger.info(f"⚡ [{i+1}/{num_layers}] Processing channel: {col}")
            layer_arrays = np.stack(df[col].progress_apply(self.pad_array).values)
            X[:, i, :, :] = layer_arrays
            
        return X

    def fit(self, df: pd.DataFrame) -> "PreprocessCNN2D":
        """
        No-op fit, present so this preprocessor honours the same fit/transform/save/load
        contract as PreprocessMLP and can be driven by the shared pipeline. The cell-to-image
        conversion is fully deterministic (padding + log1p), with nothing learned from data.

        Args:
            df (pd.DataFrame): Training rows only (unused).

        Returns:
            PreprocessCNN2D: self, for chaining.
        """
        return self

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fits on the given rows and transforms them in one call.

        Args:
            df (pd.DataFrame): Training rows only.

        Returns:
            np.ndarray: Multi-channel tensor array of shape (N, 7, 7, 15).
        """
        return self.fit(df).transform(df)

    def save(self, filepath: str) -> str:
        """
        Persists the preprocessor configuration alongside the trained checkpoints.

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
    def load(filepath: str) -> "PreprocessCNN2D":
        """
        Loads a preprocessor previously written by save().

        Args:
            filepath (str): Path to the .joblib file.

        Returns:
            PreprocessCNN2D: The restored instance.
        """
        preprocessor = joblib.load(filepath)
        logger.info(f"📂 Loaded preprocessor from: {filepath}")
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
        logger.warning(f"⚠️ Label column '{label_col}' not found in DataFrame.")
        return None
