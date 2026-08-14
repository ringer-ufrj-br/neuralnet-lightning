import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import logging
from typing import Dict, Union, List
import numpy as np

logger = logging.getLogger(__name__)

class ModelSummary:
    """
    Summary class for calculating and storing evaluation metrics in tabular CSV format.
    """

    def __init__(self, output_dir: str = "results/metrics") -> None:
        """
        Initializes ModelSummary instance.

        Args:
            output_dir (str): Directory where CSV metric files are saved. Defaults to 'results/metrics'.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def save_metrics(
        self, 
        y_true: Union[List[int], np.ndarray], 
        y_prob: Union[List[float], np.ndarray], 
        threshold: float = 0.5, 
        filename: str = "metrics_summary.csv"
    ) -> Dict[str, float]:
        """
        Calculates Accuracy, AUC, Precision, Recall, and F1-Score metrics, saving results to a CSV file.

        Args:
            y_true (Union[List[int], np.ndarray]): True target labels.
            y_prob (Union[List[float], np.ndarray]): Predicted probabilities.
            threshold (float): Classification decision threshold. Defaults to 0.5.
            filename (str): CSV output filename. Defaults to 'metrics_summary.csv'.

        Returns:
            Dict[str, float]: Dictionary containing computed metrics.
        """
        y_pred = (y_prob >= threshold).astype(int)
        
        metrics = {
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "AUC": float(roc_auc_score(y_true, y_prob)),
            "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "F1_Score": float(f1_score(y_true, y_pred, zero_division=0))
        }
        
        df = pd.DataFrame([metrics])
        filepath = os.path.join(self.output_dir, filename)
        
        if os.path.exists(filepath):
            df.to_csv(filepath, mode='a', header=False, index=False)
            logger.info(f"📊 Appended metrics to file: {filepath}")
        else:
            df.to_csv(filepath, index=False)
            logger.info(f"📝 Saved metrics to new file: {filepath}")
            
        return metrics

