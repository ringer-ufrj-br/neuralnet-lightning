import pandas as pd
import glob
import os
import logging
from typing import List, Optional
from tqdm import tqdm

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
        Reads parquet files and concatenates them into a single DataFrame.

        Args:
            files (List[str]): List of file paths to load.

        Returns:
            Optional[pd.DataFrame]: Combined pandas DataFrame or None if empty.
        """
        if not files:
            logger.warning("⚠️ No files found to load.")
            return None
        
        dfs = []
        for f in tqdm(files, desc="📥 Loading Parquets", unit="file"):
            df_temp = pd.read_parquet(f)
            df_temp['file_path'] = f
            dfs.append(df_temp)
            
        df = pd.concat(dfs, ignore_index=True)
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