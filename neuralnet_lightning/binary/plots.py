"""Evaluation figures of the binary task: ROC, PR, confusion matrix, loss and per-fold ROC."""

import logging
import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # batch nodes have no display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, auc, roc_curve  # noqa: E402

logger = logging.getLogger(__name__)


def _save(path: str, what: str) -> None:
    """Saves and closes the current figure."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
    logger.info(f"📈 Saved {what} to: {path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    path: str,
    operating_points: Optional[List[Dict[str, float]]] = None
) -> None:
    """ROC curve with its AUC, the operating points marked as FA-vs-PD dots."""
    _, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_true, y_score, name="ROC curve", plot_chance_level=True, ax=ax)
    for point in operating_points or []:
        ax.scatter(point["fa"], point["pd"], color="crimson", zorder=5)
        ax.annotate(f"{point['operating_point']} (PD={point['pd']:.3f}, FA={point['fa']:.3f})",
                    (point["fa"], point["pd"]), textcoords="offset points", xytext=(8, -4), fontsize=8)
    ax.set(xlabel="False Alarm Rate (FA)", ylabel="Probability of Detection (PD)", title="ROC Curve")
    ax.grid(True, alpha=0.3)
    _save(path, "ROC curve")


def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray, path: str) -> None:
    """Precision-Recall curve with its Average Precision and the no-skill baseline."""
    _, ax = plt.subplots(figsize=(8, 6))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, name="PR curve", plot_chance_level=True, ax=ax)
    ax.set(title="Precision-Recall Curve")
    ax.grid(True, alpha=0.3)
    _save(path, "Precision-Recall curve")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: str) -> None:
    """Confusion matrix of the binary predictions at one cut."""
    _, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", colorbar=False, values_format="d", ax=ax)
    ax.set(xlabel="Predicted Class", ylabel="True Class", title="Confusion Matrix")
    _save(path, "confusion matrix")


def plot_loss(train_loss: List[float], val_loss: List[float], path: str) -> None:
    """Learning curve: training and validation loss per epoch, best validation epoch marked."""
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label="Train Loss", linewidth=2)
    plt.plot(val_loss, label="Validation Loss", linewidth=2)
    if val_loss:
        best_epoch = int(np.argmin(val_loss))
        plt.axvline(x=best_epoch, color="r", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
        plt.plot(best_epoch, val_loss[best_epoch], "ro", markersize=6)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save(path, "loss curve")


def plot_roc_folds(y_true: np.ndarray, fold_scores: Dict[int, np.ndarray], path: str, title: str) -> None:
    """
    Overlays the ROC curve of every fold, with the mean curve and a ±1σ band, so the spread
    quoted in the cross-validation table has a visual counterpart. Curves are interpolated onto
    a shared FA grid before averaging. A zoomed inset covers the high-PD / low-FA corner, the
    only region that matters at the tight working point.
    """
    grid = np.linspace(0.0, 1.0, 1001)
    curves, interpolated, aucs = [], [], []

    plt.figure(figsize=(8, 6))
    for fold in sorted(fold_scores):
        fpr, tpr, _ = roc_curve(y_true, fold_scores[fold])
        curves.append((fpr, tpr))
        aucs.append(auc(fpr, tpr))
        interpolated.append(np.interp(grid, fpr, tpr))
        plt.plot(fpr, tpr, lw=1.0, alpha=0.45, label=f"Fold {fold} (AUC = {aucs[-1]:.4f})")

    stacked = np.vstack(interpolated)
    mean_tpr, std_tpr = stacked.mean(axis=0), stacked.std(axis=0)
    plt.plot(grid, mean_tpr, color="crimson", lw=2.2, label=f"Mean (AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f})")
    plt.fill_between(grid, np.clip(mean_tpr - std_tpr, 0, 1), np.clip(mean_tpr + std_tpr, 0, 1),
                     color="crimson", alpha=0.18, label="±1 std. dev.")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Alarm Rate (FA)")
    plt.ylabel("Probability of Detection (PD)")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)

    # Mid-right of the axes: a good ROC hugs the top-left corner, so this area is empty, and
    # it stays clear of the lower-right legend.
    inset = plt.gca().inset_axes([0.44, 0.33, 0.52, 0.44])
    for fpr, tpr in curves:
        inset.plot(fpr, tpr, lw=1.0, alpha=0.45)
    inset.plot(grid, mean_tpr, color="crimson", lw=1.8)
    inset.set_xlim(0.0, 0.2)
    inset.set_ylim(0.8, 1.005)
    inset.grid(True, alpha=0.3)
    inset.tick_params(labelsize=7)
    inset.set_title("zoom", fontsize=8)
    _save(path, "per-fold ROC overlay")
