from typing import Tuple, TypeVar

import numpy as np
import torch

Number = TypeVar("Number", float, np.ndarray, torch.Tensor)


def sp_index(pd: Number, fa: Number) -> Number:
    """
    SP Index (Ringer): sqrt(sqrt(pd*(1-fa)) * (pd+1-fa)/2).

    pd = probability of detection (recall/sensitivity), fa = false alarm rate (FPR).
    Combines the geometric and arithmetic means of pd and (1-fa) so that the score only
    rewards a threshold if signal efficiency and background rejection improve together.

    Uses the polymorphic `** 0.5` power operator instead of np.sqrt/torch.sqrt, so this
    single implementation works with Python floats, numpy arrays and torch tensors alike -
    it is shared by both the per-epoch training metric (models/*.py, torch tensors) and the
    post-hoc evaluation report (evaluation/summary.py, numpy arrays).

    Args:
        pd: Probability of detection, in [0, 1].
        fa: False alarm rate, in [0, 1].

    Returns:
        The SP Index, same type as the inputs.
    """
    return ((pd * (1 - fa)) ** 0.5 * (pd + 1 - fa) / 2) ** 0.5


def compute_pd_fa(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes PD (recall) and FA (false positive rate) from predicted probabilities and
    binary targets at a fixed decision threshold. Shared by every LightningModule that
    monitors the SP Index during validation (see ModelMLP/ModelCNN2D.on_validation_epoch_end).

    Args:
        preds (torch.Tensor): Predicted probabilities (post-sigmoid), any shape.
        targets (torch.Tensor): Binary ground-truth labels, same shape as preds.
        threshold (float): Decision threshold applied to preds. Defaults to 0.5.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (pd, fa) scalar tensors.
    """
    y_pred = (preds >= threshold).long()
    y_true = targets.long()

    tp = ((y_pred == 1) & (y_true == 1)).sum().float()
    fn = ((y_pred == 0) & (y_true == 1)).sum().float()
    fp = ((y_pred == 1) & (y_true == 0)).sum().float()
    tn = ((y_pred == 0) & (y_true == 0)).sum().float()

    pd_rate = tp / (tp + fn) if (tp + fn) > 0 else torch.zeros_like(tp)
    fa_rate = fp / (fp + tn) if (fp + tn) > 0 else torch.zeros_like(fp)
    return pd_rate, fa_rate
