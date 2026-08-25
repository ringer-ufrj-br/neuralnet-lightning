import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
import os
import logging
from typing import Dict, Union, List, Optional
import numpy as np

from ai.evaluation.metrics import sp_index

logger = logging.getLogger(__name__)

DEFAULT_OPERATING_POINTS: Dict[str, float] = {"tight": 0.90, "medium": 0.95, "loose": 0.99}


def compute_operating_points(
    y_true: Union[List[int], np.ndarray],
    y_prob: Union[List[float], np.ndarray],
    targets: Optional[Dict[str, float]] = None
) -> List[Dict[str, float]]:
    """
    Computes FA (background false alarm rate) at fixed PD (signal detection probability)
    working points. For each target PD, the threshold is set to the (1 - PD) quantile of the
    signal-class score distribution, guaranteeing that exactly that fraction of signal is kept.

    Args:
        y_true (Union[List[int], np.ndarray]): True target labels.
        y_prob (Union[List[float], np.ndarray]): Predicted probabilities.
        targets (Optional[Dict[str, float]]): Mapping of working point name to target PD.
            Defaults to {"tight": 0.90, "medium": 0.95, "loose": 0.99}.

    Returns:
        List[Dict[str, float]]: One entry per working point with Threshold, achieved PD, FA and SP_Index.
    """
    targets = targets or DEFAULT_OPERATING_POINTS
    y_true_arr = np.asarray(y_true).flatten()
    y_prob_arr = np.asarray(y_prob).flatten()
    signal_scores = y_prob_arr[y_true_arr == 1]

    points = []
    for name, target_pd in targets.items():
        threshold = float(np.quantile(signal_scores, 1 - target_pd)) if len(signal_scores) > 0 else 0.5
        y_pred = (y_prob_arr >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
        pd_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fa_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        points.append({
            "Operating_Point": name,
            "Target_PD": float(target_pd),
            "Threshold": threshold,
            "PD": float(pd_rate),
            "FA": float(fa_rate),
            "SP_Index": float(sp_index(pd_rate, fa_rate))
        })
    return points


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

        tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
        pd_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fa_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        metrics = {
            "Accuracy": float(accuracy_score(y_true_arr, y_pred)),
            "AUC_ROC": auc_roc,
            "AUC_PR": auc_pr,
            "Precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
            "F1_Score": float(f1_score(y_true_arr, y_pred, zero_division=0)),
            "SP_Index": float(sp_index(pd_rate, fa_rate)),
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

    def save_operating_points(
        self,
        y_true: Union[List[int], np.ndarray],
        y_prob: Union[List[float], np.ndarray],
        targets: Optional[Dict[str, float]] = None,
        filename: str = "operating_points.csv"
    ) -> List[Dict[str, float]]:
        """
        Computes and saves the FA-at-fixed-PD working points (tight/medium/loose) to a CSV file.

        Args:
            y_true (Union[List[int], np.ndarray]): True target labels.
            y_prob (Union[List[float], np.ndarray]): Predicted probabilities.
            targets (Optional[Dict[str, float]]): Mapping of working point name to target PD.
                Defaults to {"tight": 0.90, "medium": 0.95, "loose": 0.99}.
            filename (str): CSV output filename. Defaults to 'operating_points.csv'.

        Returns:
            List[Dict[str, float]]: Computed operating points (see compute_operating_points).
        """
        points = compute_operating_points(y_true, y_prob, targets)

        df = pd.DataFrame(points)
        filepath = os.path.join(self.output_dir, filename)

        if os.path.exists(filepath):
            df.to_csv(filepath, mode='a', header=False, index=False)
            logger.info(f"📊 Appended operating points to file: {filepath}")
        else:
            df.to_csv(filepath, index=False)
            logger.info(f"📝 Saved operating points to new file: {filepath}")

        return points

