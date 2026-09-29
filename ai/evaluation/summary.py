from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from typing import Dict, Union, List, Optional
import numpy as np

from ai.evaluation.metrics import sp_index

DEFAULT_OPERATING_POINTS: Dict[str, float] = {"tight": 0.90, "medium": 0.95, "loose": 0.99}


def compute_operating_points(
    y_true: Union[List[int], np.ndarray],
    y_prob: Union[List[float], np.ndarray],
    targets: Optional[Dict[str, float]] = None
) -> List[Dict[str, float]]:
    """
    FA (background false alarm rate) at fixed PD (signal detection probability) working points,
    one dict per point in the long-table vocabulary (operating_point, target_pd, threshold, pd,
    fa, sp). For each target PD, the threshold is set to the (1 - PD) quantile of the
    signal-class score distribution, guaranteeing that exactly that fraction of signal is kept.

    This is the mechanism behind the cross-validation table ("pd_table"): every network is
    tuned to deliver the same PD, so the columns that actually differ between models are
    SP and FA.
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
            "operating_point": name,
            "target_pd": float(target_pd),
            "threshold": threshold,
            "pd": float(pd_rate),
            "fa": float(fa_rate),
            "sp": float(sp_index(pd_rate, fa_rate))
        })
    return points


def compute_metrics(
    y_true: Union[List[int], np.ndarray],
    y_prob: Union[List[float], np.ndarray]
) -> Dict[str, float]:
    """
    The threshold-free metrics of one set of predictions, in the long-table vocabulary
    (auc_roc, auc_pr, n_signal, n_background).

    Everything here is a property of the score ranking rather than of any particular cut, so
    it is comparable across models without agreeing on a decision threshold first. Metrics at
    a cut belong to `compute_operating_points`, where the cut is derived from a target PD.
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

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "n_signal": int((y_true_arr == 1).sum()),
        "n_background": int((y_true_arr == 0).sum()),
    }
