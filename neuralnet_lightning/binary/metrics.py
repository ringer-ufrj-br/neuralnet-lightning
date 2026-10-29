"""
Binary-task figures of merit: the SP index (in training and in evaluation), FA at fixed-PD
operating points, and the threshold-free AUCs.
"""

from typing import Dict, List, TypeVar

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

Number = TypeVar("Number", float, np.ndarray, torch.Tensor)


def sp_index(pd: Number, fa: Number) -> Number:
    """
    SP Index (Ringer): sqrt(sqrt(pd*(1-fa)) * (pd+1-fa)/2).

    pd = probability of detection (signal efficiency), fa = false alarm rate (background
    efficiency). Combines the geometric and arithmetic means of pd and (1-fa) so a threshold
    only scores well if signal efficiency and background rejection improve together.

    Uses `** 0.5` rather than np.sqrt/torch.sqrt so one implementation serves Python floats,
    numpy arrays and torch tensors - the per-epoch training metric and the evaluation report.
    """
    return ((pd * (1 - fa)) ** 0.5 * (pd + 1 - fa) / 2) ** 0.5


def max_sp_index(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Highest SP Index reachable over every decision threshold - the NeuralRinger stopping
    criterion - rather than SP at one arbitrary cut: with the class imbalance of this dataset
    and a pos_weight-weighted loss, the best operating point is nowhere near 0.5, so a fixed
    cut would rank epochs by where their scores happen to sit instead of by separability.

    Sweeps the thresholds the ROC visits - one per distinct score, so a cut never splits a
    group of equal scores - and returns the best SP among them (0 when either class is absent).
    """
    scores = preds.flatten()
    y = targets.flatten().long()

    n_pos = int((y == 1).sum())
    n_neg = y.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return torch.zeros((), dtype=torch.float32, device=preds.device)

    order = torch.argsort(scores, descending=True)
    scores, y = scores[order], y[order]

    pd_rate = torch.cumsum(y, 0).float() / n_pos
    fa_rate = torch.cumsum(1 - y, 0).float() / n_neg

    # Keep only the last row of each run of equal scores: those are the reachable cuts.
    reachable = torch.ones_like(scores, dtype=torch.bool)
    reachable[:-1] = scores[1:] != scores[:-1]

    return sp_index(pd_rate[reachable], fa_rate[reachable]).max()


def compute_operating_points(
    y_true: np.ndarray,
    y_score: np.ndarray,
    targets: Dict[str, float]
) -> List[Dict[str, float]]:
    """
    FA at fixed-PD working points, one dict per point (operating_point, target_pd, threshold,
    pd, fa, sp). For each target PD the threshold is the (1 - PD) quantile of the signal score
    distribution, so exactly that fraction of signal is kept.

    This is the mechanism behind the cross-validation table: every network is tuned to the
    same PD, so the columns that actually differ between models are SP and FA.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    signal_scores = y_score[y_true == 1]

    points = []
    for name, target_pd in targets.items():
        threshold = float(np.quantile(signal_scores, 1 - target_pd)) if len(signal_scores) else 0.5
        y_pred = (y_score >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        pd_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fa_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        points.append({
            "operating_point": name,
            "target_pd": float(target_pd),
            "threshold": threshold,
            "pd": float(pd_rate),
            "fa": float(fa_rate),
            "sp": float(sp_index(pd_rate, fa_rate)),
        })
    return points


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """
    The threshold-free metrics of one set of scores (auc_roc, auc_pr, n_signal, n_background).
    Properties of the score ranking rather than of a cut, so comparable across models without
    agreeing on a threshold first. A single-class input yields NaN AUCs.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    both_classes = len(np.unique(y_true)) == 2

    return {
        "auc_roc": float(roc_auc_score(y_true, y_score)) if both_classes else float("nan"),
        "auc_pr": float(average_precision_score(y_true, y_score)) if both_classes else float("nan"),
        "n_signal": int((y_true == 1).sum()),
        "n_background": int((y_true == 0).sum()),
    }
