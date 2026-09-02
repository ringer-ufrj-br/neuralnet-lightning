from typing import TypeVar

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


def max_sp_index(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Highest SP Index reachable over every decision threshold ("maximo do indice SP", the
    NeuralRinger stopping criterion), rather than SP at one arbitrary cut: with the class
    imbalance of this dataset and a pos_weight-weighted loss, the best operating point is
    nowhere near 0.5, so a fixed cut ranks epochs by where their scores happen to sit instead
    of by how separable the two classes are.

    Sweeps the thresholds the ROC visits - one per distinct score, so a cut never splits a
    group of equal scores - and returns the best SP among them.

    Args:
        preds (torch.Tensor): Predicted probabilities (post-sigmoid), any shape.
        targets (torch.Tensor): Binary ground-truth labels, same shape as preds.

    Returns:
        torch.Tensor: Scalar best SP Index, 0 when either class is absent.
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
