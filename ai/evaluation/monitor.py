import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score
import os
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

try:
    import mplhep as hep
    plt.style.use(hep.style.ATLAS)
except ImportError:
    pass

class ModelMonitor:
    """
    Evaluator class for generating visual performance plots (ROC curve, PR curve, Confusion Matrix, Loss curves).
    """

    def __init__(self, output_dir: str = "results/plots") -> None:
        """
        Initializes ModelMonitor instance.

        Args:
            output_dir (str): Directory where plots will be saved. Defaults to 'results/plots'.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_roc_curve(
        self,
        y_true: Union[List[int], np.ndarray],
        y_prob: Union[List[float], np.ndarray],
        filename: str = "roc_curve.pdf",
        operating_points: Optional[List[Dict[str, float]]] = None
    ) -> None:
        """
        Plots and saves the Receiver Operating Characteristic (ROC) curve with AUC metric.

        Args:
            y_true (Union[List[int], np.ndarray]): True target labels.
            y_prob (Union[List[float], np.ndarray]): Predicted probabilities.
            filename (str): Output filename. Defaults to 'roc_curve.pdf'.
            operating_points (Optional[List[Dict[str, float]]]): Working points (e.g. tight/medium/loose)
                as produced by ai.evaluation.summary.compute_operating_points, marked as FA-vs-PD dots.

        Returns:
            None
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        if operating_points:
            for point in operating_points:
                plt.scatter(point["FA"], point["PD"], color='crimson', zorder=5)
                plt.annotate(
                    f"{point['Operating_Point']} (PD={point['PD']:.3f}, FA={point['FA']:.3f})",
                    (point["FA"], point["PD"]),
                    textcoords="offset points", xytext=(8, -4), fontsize=8
                )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Alarm Rate (FA)')
        plt.ylabel('Probability of Detection (PD)')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"📈 Saved ROC curve to: {filepath}")

    def plot_pr_curve(self, y_true: Union[List[int], np.ndarray], y_prob: Union[List[float], np.ndarray], filename: str = "pr_curve.pdf") -> None:
        """
        Plots and saves the Precision-Recall (PR) curve with Average Precision (AP / PR-AUC).

        Args:
            y_true (Union[List[int], np.ndarray]): True target labels.
            y_prob (Union[List[float], np.ndarray]): Predicted probabilities.
            filename (str): Output filename. Defaults to 'pr_curve.pdf'.

        Returns:
            None
        """
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        
        y_true_arr = np.asarray(y_true).flatten()
        pos_ratio = (y_true_arr == 1).sum() / max(len(y_true_arr), 1)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AP = {ap:.4f})')
        plt.axhline(y=pos_ratio, color='navy', lw=2, linestyle='--', label=f'No Skill Baseline ({pos_ratio:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"📈 Saved Precision-Recall curve to: {filepath}")
        
    def plot_confusion_matrix(self, y_true: Union[List[int], np.ndarray], y_pred: Union[List[int], np.ndarray], filename: str = "confusion_matrix.pdf") -> None:
        """
        Plots and saves the Confusion Matrix heatmap.

        Args:
            y_true (Union[List[int], np.ndarray]): True target labels.
            y_pred (Union[List[int], np.ndarray]): Predicted binary labels.
            filename (str): Output filename. Defaults to 'confusion_matrix.pdf'.

        Returns:
            None
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title('Confusion Matrix')
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"📊 Saved Confusion Matrix to: {filepath}")

    def plot_loss(self, train_loss: List[float], val_loss: List[float], filename: str = "loss_curve.pdf") -> None:
        """
        Plots learning curve comparing training and validation loss over epochs.

        Args:
            train_loss (List[float]): List of training loss values per epoch.
            val_loss (List[float]): List of validation loss values per epoch.
            filename (str): Output filename. Defaults to 'loss_curve.pdf'.

        Returns:
            None
        """
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
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"📉 Saved Loss Curve to: {filepath}")

    def plot_roc_folds(
        self,
        fold_scores: Dict[int, Tuple[np.ndarray, np.ndarray]],
        filename: str = "roc_folds.pdf",
        operating_points: Optional[List[Dict[str, float]]] = None,
        title: str = "ROC Curve — Cross Validation",
        zoom: bool = True
    ) -> Optional[str]:
        """
        Overlays the ROC curve of every fold in one figure, with the mean curve and a
        +/-1 sigma band, so the spread quoted in the cross-validation table has a visual
        counterpart. Curves are interpolated onto a shared FA grid before averaging, since
        each fold's roc_curve() returns its own set of thresholds.

        Args:
            fold_scores (Dict[int, Tuple[np.ndarray, np.ndarray]]): Mapping fold number ->
                (y_true, y_prob).
            filename (str): Output filename. Defaults to 'roc_folds.pdf'.
            operating_points (Optional[List[Dict[str, float]]]): Working points to mark,
                typically the fold-averaged ones.
            title (str): Figure title.
            zoom (bool): Also draw a zoomed inset over the high-PD / low-FA corner, which is
                the only region that matters at the tight working point. Defaults to True.

        Returns:
            Optional[str]: The written path, or None when there was nothing to plot.
        """
        if not fold_scores:
            logger.warning("⚠️ No fold scores supplied; skipping fold ROC overlay.")
            return None

        grid = np.linspace(0.0, 1.0, 1001)
        interpolated, aucs = [], []

        plt.figure(figsize=(8, 6))
        for fold in sorted(fold_scores):
            y_true, y_prob = fold_scores[fold]
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            fold_auc = auc(fpr, tpr)
            aucs.append(fold_auc)
            interpolated.append(np.interp(grid, fpr, tpr))
            plt.plot(fpr, tpr, lw=1.0, alpha=0.45, label=f"Fold {fold} (AUC = {fold_auc:.4f})")

        stacked = np.vstack(interpolated)
        mean_tpr, std_tpr = stacked.mean(axis=0), stacked.std(axis=0)

        plt.plot(
            grid, mean_tpr, color="crimson", lw=2.2,
            label=f"Mean (AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f})"
        )
        if len(interpolated) > 1:
            plt.fill_between(
                grid,
                np.clip(mean_tpr - std_tpr, 0, 1),
                np.clip(mean_tpr + std_tpr, 0, 1),
                color="crimson", alpha=0.18, label="±1 std. dev."
            )
        plt.plot([0, 1], [0, 1], color="navy", lw=1.2, linestyle="--")

        if operating_points:
            for point in operating_points:
                plt.scatter(point["FA"], point["PD"], color="black", marker="*", s=90, zorder=5)
                plt.annotate(
                    f"{point['Operating_Point']}",
                    (point["FA"], point["PD"]),
                    textcoords="offset points", xytext=(8, -10), fontsize=8
                )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Alarm Rate (FA)")
        plt.ylabel("Probability of Detection (PD)")
        plt.title(title)
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(True, alpha=0.3)

        if zoom:
            # Placed in the mid-right of the axes: a well-performing ROC hugs the top-left
            # corner, so this region is empty, and it stays clear of the lower-right legend.
            inset = plt.gca().inset_axes([0.44, 0.33, 0.52, 0.44])
            for fold in sorted(fold_scores):
                y_true, y_prob = fold_scores[fold]
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                inset.plot(fpr, tpr, lw=1.0, alpha=0.45)
            inset.plot(grid, mean_tpr, color="crimson", lw=1.8)
            inset.set_xlim(0.0, 0.2)
            inset.set_ylim(0.8, 1.005)
            inset.grid(True, alpha=0.3)
            inset.tick_params(labelsize=7)
            inset.set_title("zoom", fontsize=8)

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"📈 Saved per-fold ROC overlay to: {filepath}")
        return filepath
