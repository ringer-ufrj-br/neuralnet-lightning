import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    average_precision_score
)
import os
import logging
from typing import Dict, Union, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class ModelSummary:
    """
    Summary class for calculating and storing evaluation metrics in tabular CSV format
    with special focus on imbalanced classification metrics (PR-AUC, ROC-AUC, F1, Precision, Recall).
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
        pos_weight: Optional[float] = None,
        filename: str = "metrics_summary.csv"
    ) -> Dict[str, float]:
        """
        Calculates Accuracy, AUC-ROC, AUC-PR, Precision, Recall, and F1-Score metrics, saving results to a CSV file.

        Args:
            y_true (Union[List[int], np.ndarray]): True target labels.
            y_prob (Union[List[float], np.ndarray]): Predicted probabilities.
            threshold (float): Classification decision threshold. Defaults to 0.5.
            pos_weight (Optional[float]): Positive class weight used during training. Defaults to None.
            filename (str): CSV output filename. Defaults to 'metrics_summary.csv'.

        Returns:
            Dict[str, float]: Dictionary containing computed metrics.
        """
        y_true_arr = np.asarray(y_true).flatten()
        y_prob_arr = np.asarray(y_prob).flatten()
        y_pred = (y_prob_arr >= threshold).astype(int)
        
        n_pos = int((y_true_arr == 1).sum())
        n_neg = int((y_true_arr == 0).sum())
        
        # Calculate ROC-AUC and PR-AUC safely
        try:
            auc_roc = float(roc_auc_score(y_true_arr, y_prob_arr))
        except Exception:
            auc_roc = 0.0
            
        try:
            auc_pr = float(average_precision_score(y_true_arr, y_prob_arr))
        except Exception:
            auc_pr = 0.0
        
        metrics = {
            "Accuracy": float(accuracy_score(y_true_arr, y_pred)),
            "AUC_ROC": auc_roc,
            "AUC_PR": auc_pr,
            "Precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
            "F1_Score": float(f1_score(y_true_arr, y_pred, zero_division=0)),
            "Threshold": float(threshold),
            "N_Positives": n_pos,
            "N_Negatives": n_neg
        }
        
        if pos_weight is not None:
            metrics["Pos_Weight"] = float(pos_weight)
        
        df = pd.DataFrame([metrics])
        filepath = os.path.join(self.output_dir, filename)
        
        if os.path.exists(filepath):
            df.to_csv(filepath, mode='a', header=False, index=False)
            logger.info(f"📊 Appended metrics to file: {filepath}")
        else:
            df.to_csv(filepath, index=False)
            logger.info(f"📝 Saved metrics to new file: {filepath}")
            
        return metrics

