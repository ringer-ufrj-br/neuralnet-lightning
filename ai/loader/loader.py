import pandas as pd
import polars as pl
import glob
import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Data loader class for finding and reading parquet dataset files.
    """

    def __init__(self, data_path: Optional[str] = None, max_files: Optional[int] = None) -> None:
        """
        Initializes DataLoader instance.

        Args:
            data_path (Optional[str]): Path or glob pattern for dataset parquet files.
            max_files (Optional[int]): Maximum number of files per folder to load.
        """
        self.max_files = max_files
        if data_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_path = os.path.join(script_dir, "..", "..", "data", "parquet", "**", "*.parquet")
        else:
            self.data_path = data_path

    def get_files(self) -> List[str]:
        """
        Retrieves list of valid parquet files matching data_path.

        Args:
            None

        Returns:
            List[str]: List of resolved file paths.
        """
        if os.path.isdir(self.data_path):
            search_path = os.path.join(self.data_path, "**", "*.parquet")
        else:
            search_path = self.data_path
            
        files_per_folder = {}
        for f in glob.glob(search_path, recursive=True):
            if os.path.isfile(f):
                folder = os.path.dirname(f)
                if folder not in files_per_folder:
                    files_per_folder[folder] = []
                files_per_folder[folder].append(f)
        
        files = []
        for folder, file_list in files_per_folder.items():
            file_list.sort()
            if self.max_files is not None:
                files.extend(file_list[:self.max_files])
            else:
                files.extend(file_list)
                
        logger.info(f"📂 Found {len(files)} valid parquet files.")
        return files

    def load_dataset(self, files: List[str]) -> Optional[pd.DataFrame]:
        """
        Reads and concatenates parquet files into a single DataFrame using polars as the
        reading/concatenation engine (single lazy scan across all files, one Arrow-native
        concat), which avoids the per-file Python object overhead and the N-way in-memory
        duplication that pandas' read-then-concat loop incurs. Converted to pandas at the
        end for compatibility with the rest of the pipeline (label generation, preprocessors).

        Args:
            files (List[str]): List of file paths to load.

        Returns:
            Optional[pd.DataFrame]: Combined pandas DataFrame or None if empty.
        """
        if not files:
            logger.warning("⚠️ No files found to load.")
            return None

        logger.info(f"📥 Reading {len(files)} parquet files via polars...")
        df = pl.scan_parquet(files, include_file_paths="file_path", low_memory=True).collect().to_pandas()
        return df

    def execute(self) -> Optional[pd.DataFrame]:
        """
        Executes complete data loading pipeline (find files and read dataset).

        Args:
            None

        Returns:
            Optional[pd.DataFrame]: Concatenated dataset DataFrame.
        """
        files = self.get_files()
        df = self.load_dataset(files)
        return df

if __name__ == "__main__":
    loader = DataLoader()
    loader.execute()