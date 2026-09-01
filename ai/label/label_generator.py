import pandas as pd
import polars as pl
import os
import logging
from typing import List

logger = logging.getLogger(__name__)

class LabelGenerator:
    """
    Module responsible for generating labels for project dataset.
    Defined rules:
    - 'Zee'  -> Label 1 (Signal)
    - 'JF17' -> Label 0 (Background)
    """
    
    @staticmethod
    def get_label_from_path(file_path: str) -> int:
        """
        Returns label based on filename or file path.

        Args:
            file_path (str): File path or name.

        Returns:
            int: 1 for Zee, 0 for JF17.
        """
        file_path_lower = file_path.lower()
        if 'zee' in file_path_lower:
            return 1
        elif 'jf17' in file_path_lower:
            return 0
        else:
            logger.error(f"❌ Could not determine label for file path: {file_path}")
            raise ValueError(f"❌ Could not determine label for file path: {file_path}")

    @classmethod
    def validate_files(cls, files: List[str]) -> None:
        """
        Checks up front that every file resolves to a label. Labels are a pure function of
        the file path, so validating the (short) file list here lets label_expr run lazily
        over millions of rows without needing a per-row unknown-path check.

        Args:
            files (List[str]): Dataset file paths.

        Raises:
            ValueError: If any file path matches neither Zee nor JF17.
        """
        for file_path in files:
            cls.get_label_from_path(file_path)

    @classmethod
    def label_expr(cls, file_path_col: str = 'file_path', label_col: str = 'label') -> pl.Expr:
        """
        Polars expression computing the label from the file path column, for use inside a
        lazy query (the path strings are never materialized in memory). Mirrors
        get_label_from_path; call validate_files first so unknown paths fail early instead
        of producing nulls here.

        Args:
            file_path_col (str): Column containing file paths. Defaults to 'file_path'.
            label_col (str): Output label column name. Defaults to 'label'.

        Returns:
            pl.Expr: Expression yielding 1 for Zee rows and 0 for JF17 rows.
        """
        lower = pl.col(file_path_col).str.to_lowercase()
        return (
            pl.when(lower.str.contains('zee', literal=True)).then(1)
            .when(lower.str.contains('jf17', literal=True)).then(0)
            .otherwise(None)
            .alias(label_col)
        )

    @classmethod
    def apply_label(cls, df: pd.DataFrame, file_path_col: str = 'file_path', label_col: str = 'label') -> pd.DataFrame:
        """
        Applies label column to DataFrame based on the file path column.

        Args:
            df (pd.DataFrame): Input DataFrame containing file paths.
            file_path_col (str): Column name containing file paths. Defaults to 'file_path'.
            label_col (str): Output label column name. Defaults to 'label'.

        Returns:
            pd.DataFrame: DataFrame updated with target label column.
        """
        if file_path_col not in df.columns:
            logger.error(f"❌ Column '{file_path_col}' not found in DataFrame.")
            raise ValueError(f"❌ Column '{file_path_col}' not found in DataFrame.")
        
        logger.info(f"🏷️ Generating labels from '{file_path_col}' column...")
        df[label_col] = df[file_path_col].apply(cls.get_label_from_path)
        return df
