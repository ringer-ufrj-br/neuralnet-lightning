"""
Binary half of `evaluate`: one region's scores in, its slice of the cross-validation table
(`metrics/folds_long.csv`, see neuralnet_lightning.binary.report.LONG_COLUMNS) and its figures
out.
"""

import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from neuralnet_lightning.binary import plots
from neuralnet_lightning.binary.metrics import compute_metrics, compute_operating_points
from neuralnet_lightning.binning import Region
from neuralnet_lightning.paths import RunPaths

logger = logging.getLogger(__name__)


def summarize(
    name: str,
    region: Optional[Region],
    y_true: np.ndarray,
    fold_scores: Dict[int, np.ndarray],
    infos: Dict[int, Dict[str, Any]],
    operating_points: Dict[str, float],
    paths: RunPaths,
    make_plots: bool = True
) -> pd.DataFrame:
    """
    Writes and returns the long table of one region: one row per (fold, operating point) with
    PD, FA, SP and the threshold at that point, plus the fold's threshold-free metrics.
    """
    rows = []
    for fold in sorted(fold_scores):
        metrics = compute_metrics(y_true, fold_scores[fold])
        for point in compute_operating_points(y_true, fold_scores[fold], operating_points):
            rows.append({"name": name,
                         "et_bin": region.et_bin if region else None,
                         "eta_bin": region.eta_bin if region else None,
                         "fold": fold, **point, **metrics})
            logger.info(f"   fold {fold} {point['operating_point']:<7} PD={point['pd']:.4f} "
                        f"(target {point['target_pd']:.2f}) -> FA={point['fa']:.4f}, "
                        f"SP={point['sp']:.4f}, threshold={point['threshold']:.4f}")

    long_df = pd.DataFrame(rows)
    os.makedirs(paths.metrics, exist_ok=True)
    long_df.to_csv(paths.folds_long, index=False)
    logger.info(f"📝 Saved long-format fold table ({len(long_df)} rows) to: {paths.folds_long}")

    if make_plots:
        _render(name, region, y_true, fold_scores, infos, operating_points, paths)
    return long_df


def _render(
    name: str,
    region: Optional[Region],
    y_true: np.ndarray,
    fold_scores: Dict[int, np.ndarray],
    infos: Dict[int, Dict[str, Any]],
    operating_points: Dict[str, float],
    paths: RunPaths
) -> None:
    """The per-fold figures plus the fold-overlay ROC of this region."""
    logger.info(f"🖼️ Rendering plots into {paths.plots}...")
    for fold in sorted(fold_scores):
        y_score = fold_scores[fold]
        points = compute_operating_points(y_true, y_score, operating_points)
        # The confusion matrix needs one cut; the tightest working point is the one the trigger
        # would actually run at, so it is the cut worth picturing.
        cut = min(points, key=lambda point: point["target_pd"])["threshold"]

        plots.plot_roc_curve(y_true, y_score, os.path.join(paths.plots, f"roc_curve_fold_{fold}.pdf"), points)
        plots.plot_pr_curve(y_true, y_score, os.path.join(paths.plots, f"pr_curve_fold_{fold}.pdf"))
        plots.plot_confusion_matrix(y_true, (y_score >= cut).astype(int),
                                    os.path.join(paths.plots, f"confusion_matrix_fold_{fold}.pdf"))

        history_path = paths.resolve(infos[fold]["history"])
        if os.path.isfile(history_path):
            history = pd.read_csv(history_path)
            plots.plot_loss(history["train_loss"].dropna().tolist(), history["val_loss"].dropna().tolist(),
                            os.path.join(paths.plots, f"loss_curve_fold_{fold}.pdf"))

    if len(fold_scores) > 1:
        where = region.label if region else "full phase space"
        plots.plot_roc_folds(y_true, fold_scores, os.path.join(paths.plots, "roc_folds.pdf"),
                             title=f"ROC Curve — {name} ({where})")
