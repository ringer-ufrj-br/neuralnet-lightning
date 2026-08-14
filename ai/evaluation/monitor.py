import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os
import logging
from typing import List, Union
import numpy as np

logger = logging.getLogger(__name__)

try:
    import mplhep as hep
    plt.style.use(hep.style.ATLAS)
except ImportError:
    pass

class ModelMonitor:
    """
    Evaluator class for generating visual performance plots (ROC curve, Confusion Matrix, Loss curves).
    """

    def __init__(self, output_dir: str = "results/plots") -> None:
        """
        Initializes ModelMonitor instance.

        Args:
            output_dir (str): Directory where plots will be saved. Defaults to 'results/plots'.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_roc_curve(self, y_true: Union[List[int], np.ndarray], y_prob: Union[List[float], np.ndarray], filename: str = "roc_curve.pdf") -> None:
        """
        Plots and saves the Receiver Operating Characteristic (ROC) curve with AUC metric.

        Args:
            y_true (Union[List[int], np.ndarray]): True target labels.
            y_prob (Union[List[float], np.ndarray]): Predicted probabilities.
            filename (str): Output filename. Defaults to 'roc_curve.pdf'.

        Returns:
            None
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"📈 Saved ROC curve to: {filepath}")
        
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
