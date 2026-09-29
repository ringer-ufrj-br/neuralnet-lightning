import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    auc,
    roc_curve,
)
import os
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Renders the evaluation figures (ROC curve, PR curve, Confusion Matrix, Loss curves) into
    one directory.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _save(self, filename: str, what: str) -> None:
        """Saves and closes the current figure."""
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"📈 Saved {what} to: {filepath}")

    def plot_roc_curve(
        self,
        y_true: Union[List[int], np.ndarray],
        y_prob: Union[List[float], np.ndarray],
        filename: str,
        operating_points: Optional[List[Dict[str, float]]] = None
    ) -> None:
        """
        ROC curve with its AUC, the working points (as produced by
        ai.evaluation.summary.compute_operating_points) marked as FA-vs-PD dots.
        """
        _, ax = plt.subplots(figsize=(8, 6))
        RocCurveDisplay.from_predictions(y_true, y_prob, name="ROC curve", plot_chance_level=True, ax=ax)
        for point in operating_points or []:
            ax.scatter(point["fa"], point["pd"], color='crimson', zorder=5)
            ax.annotate(
                f"{point['operating_point']} (PD={point['pd']:.3f}, FA={point['fa']:.3f})",
                (point["fa"], point["pd"]),
                textcoords="offset points", xytext=(8, -4), fontsize=8
            )
        ax.set(xlabel='False Alarm Rate (FA)', ylabel='Probability of Detection (PD)', title='ROC Curve')
        ax.grid(True, alpha=0.3)
        self._save(filename, "ROC curve")

    def plot_pr_curve(self, y_true: Union[List[int], np.ndarray], y_prob: Union[List[float], np.ndarray], filename: str) -> None:
        """Precision-Recall curve with its Average Precision and the no-skill baseline."""
        _, ax = plt.subplots(figsize=(8, 6))
        PrecisionRecallDisplay.from_predictions(y_true, y_prob, name="PR curve", plot_chance_level=True, ax=ax)
        ax.set(title='Precision-Recall Curve')
        ax.grid(True, alpha=0.3)
        self._save(filename, "Precision-Recall curve")

    def plot_confusion_matrix(self, y_true: Union[List[int], np.ndarray], y_pred: Union[List[int], np.ndarray], filename: str) -> None:
        """Confusion Matrix heatmap of the binary predictions."""
        _, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", colorbar=False, values_format="d", ax=ax)
        ax.set(xlabel='Predicted Class', ylabel='True Class', title='Confusion Matrix')
        self._save(filename, "Confusion Matrix")

    def plot_loss(self, train_loss: List[float], val_loss: List[float], filename: str) -> None:
        """Learning curve comparing training and validation loss over epochs."""
        plt.figure(figsize=(8, 6))
        plt.plot(train_loss, label='Train Loss', linewidth=2)
        plt.plot(val_loss, label='Validation Loss', linewidth=2)

        if val_loss:
            best_epoch = np.argmin(val_loss)
            best_val = val_loss[best_epoch]
            plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
            plt.plot(best_epoch, best_val, 'ro', markersize=6)

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        self._save(filename, "Loss Curve")

    def plot_roc_folds(
        self,
        fold_scores: Dict[int, Tuple[np.ndarray, np.ndarray]],
        filename: str,
        title: str
    ) -> None:
        """
        Overlays the ROC curve of every fold (fold -> (y_true, y_prob)) in one figure, with the
        mean curve and a +/-1 sigma band, so the spread quoted in the cross-validation table
        has a visual counterpart. Curves are interpolated onto a shared FA grid before
        averaging, since each fold's roc_curve() returns its own set of thresholds. A zoomed
        inset covers the high-PD / low-FA corner, the only region that matters at the tight
        working point.
        """
        grid = np.linspace(0.0, 1.0, 1001)
        curves, interpolated, aucs = [], [], []

        plt.figure(figsize=(8, 6))
        for fold in sorted(fold_scores):
            fpr, tpr, _ = roc_curve(*fold_scores[fold])
            curves.append((fpr, tpr))
            aucs.append(auc(fpr, tpr))
            interpolated.append(np.interp(grid, fpr, tpr))
            plt.plot(fpr, tpr, lw=1.0, alpha=0.45, label=f"Fold {fold} (AUC = {aucs[-1]:.4f})")

        stacked = np.vstack(interpolated)
        mean_tpr, std_tpr = stacked.mean(axis=0), stacked.std(axis=0)

        plt.plot(
            grid, mean_tpr, color="crimson", lw=2.2,
            label=f"Mean (AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f})"
        )
        plt.fill_between(
            grid,
            np.clip(mean_tpr - std_tpr, 0, 1),
            np.clip(mean_tpr + std_tpr, 0, 1),
            color="crimson", alpha=0.18, label="±1 std. dev."
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=1.2, linestyle="--")

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Alarm Rate (FA)")
        plt.ylabel("Probability of Detection (PD)")
        plt.title(title)
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(True, alpha=0.3)

        # Placed in the mid-right of the axes: a well-performing ROC hugs the top-left
        # corner, so this region is empty, and it stays clear of the lower-right legend.
        inset = plt.gca().inset_axes([0.44, 0.33, 0.52, 0.44])
        for fpr, tpr in curves:
            inset.plot(fpr, tpr, lw=1.0, alpha=0.45)
        inset.plot(grid, mean_tpr, color="crimson", lw=1.8)
        inset.set_xlim(0.0, 0.2)
        inset.set_ylim(0.8, 1.005)
        inset.grid(True, alpha=0.3)
        inset.tick_params(labelsize=7)
        inset.set_title("zoom", fontsize=8)

        self._save(filename, "per-fold ROC overlay")
