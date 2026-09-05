import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
import os
import logging
from typing import Dict, Union, List, Optional, Any
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

    This is the mechanism behind the cross-validation table ("pd_table"): every network is
    tuned to deliver the same PD, so the columns that actually differ between models are
    SP and FA.

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


def compute_metrics(
    y_true: Union[List[int], np.ndarray],
    y_prob: Union[List[float], np.ndarray],
    pos_weight: Optional[float] = None
) -> Dict[str, float]:
    """
    Computes the threshold-free metric set for one set of predictions.

    Everything here is a property of the score ranking rather than of any particular cut, so
    it is comparable across models without agreeing on a decision threshold first. Metrics at
    a cut belong to `compute_operating_points`, where the cut is derived from a target PD.

    Pure function with no I/O, so it can be reused by the pipeline, by notebooks and by
    the table builder without dragging a ModelSummary instance along.

    Args:
        y_true (Union[List[int], np.ndarray]): True target labels.
        y_prob (Union[List[float], np.ndarray]): Predicted probabilities.
        pos_weight (Optional[float]): Positive class weight used during training. Defaults to None.

    Returns:
        Dict[str, float]: AUC_ROC, AUC_PR, N_Positives, N_Negatives and (when given) Pos_Weight.
    """
    y_true_arr = np.asarray(y_true).flatten()
    y_prob_arr = np.asarray(y_prob).flatten()

    try:
        auc_roc = float(roc_auc_score(y_true_arr, y_prob_arr))
    except Exception:
        auc_roc = 0.0

    try:
        auc_pr = float(average_precision_score(y_true_arr, y_prob_arr))
    except Exception:
        auc_pr = 0.0

    metrics = {
        "AUC_ROC": auc_roc,
        "AUC_PR": auc_pr,
        "N_Positives": int((y_true_arr == 1).sum()),
        "N_Negatives": int((y_true_arr == 0).sum()),
    }

    if pos_weight is not None:
        metrics["Pos_Weight"] = float(pos_weight)

    return metrics


class ModelSummary:
    """
    Writes evaluation metrics as tidy CSV files, one row per (fold, operating point).

    Every write **overwrites** its target file. The previous append-mode behaviour silently
    accumulated one duplicate row set per re-run, which made any downstream aggregation
    (mean +/- std across folds) read stale rows from earlier runs as if they were extra folds.
    """

    def __init__(self, output_dir: str = "results/metrics") -> None:
        """
        Initializes ModelSummary instance.

        Args:
            output_dir (str): Directory where CSV metric files are saved. Defaults to 'results/metrics'.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _write(self, records: List[Dict[str, Any]], filename: str, description: str) -> pd.DataFrame:
        """
        Writes a list of record dicts to CSV, replacing any previous content.

        Args:
            records (List[Dict[str, Any]]): Rows to write.
            filename (str): CSV output filename, relative to output_dir.
            description (str): Human-readable label used in the log line.

        Returns:
            pd.DataFrame: The written frame.
        """
        df = pd.DataFrame(records)
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"📝 Saved {description} ({len(df)} rows) to: {filepath}")
        return df


    def save_operating_points(
        self,
        records: List[Dict[str, Any]],
        filename: str = "operating_points.csv"
    ) -> pd.DataFrame:
        """
        Saves the per-fold working point rows (as produced by compute_operating_points, plus a Fold key).

        Args:
            records (List[Dict[str, Any]]): One dict per (fold, operating point).
            filename (str): CSV output filename. Defaults to 'operating_points.csv'.

        Returns:
            pd.DataFrame: The written frame.
        """
        return self._write(records, filename, "operating points")

    def save_long_table(
        self,
        records: List[Dict[str, Any]],
        filename: str = "folds_long.csv"
    ) -> pd.DataFrame:
        """
        Saves the canonical tidy table for this kinematic region: one purely numeric row per
        (fold, operating point). This is the single source of truth the cross-validation table
        builder (ai.evaluation.pd_table) reads back; the LaTeX/figure renders are derived from it.

        Args:
            records (List[Dict[str, Any]]): One dict per (fold, operating point).
            filename (str): CSV output filename. Defaults to 'folds_long.csv'.

        Returns:
            pd.DataFrame: The written frame.
        """
        return self._write(records, filename, "long-format fold table")
