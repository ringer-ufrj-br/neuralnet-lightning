import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class DataBalancer:
    """
    [DEPRECATED] Data balancer class using random undersampling to balance dataset class distribution.
    Note: The pipeline has migrated to cost-sensitive learning via weighted loss (BCEWithLogitsLoss with pos_weight)
    in ai.trainer.trainer to preserve 100% of training samples and avoid artificial subsampling.
    """

    def __init__(self, random_state: int = 42) -> None:
        """
        Initializes DataBalancer instance.

        Args:
            random_state (int): Seed for random number generator. Defaults to 42.
        """
        self.random_state = random_state

    def apply(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies undersampling to features X and target labels Y.

        Args:
            X (np.ndarray): Feature array.
            Y (np.ndarray): Target labels array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Balanced features (X_balanced) and labels (Y_balanced).
        """
        y_flat = Y.flatten() if Y.ndim > 1 else Y
        classes, counts = np.unique(y_flat, return_counts=True)
        
        if len(classes) < 2:
            logger.warning("⚠️ DataBalancer: Fewer than 2 classes found. Skipping data balancing.")
            return X, Y

        min_count = np.min(counts)
        logger.info(f"⚖️ DataBalancer: Balancing to {min_count} samples per class (Undersampling)...")
        
        rng = np.random.default_rng(seed=self.random_state)
        
        balanced_indices = []
        for cls in classes:
            cls_indices = np.where(y_flat == cls)[0]
            selected_indices = rng.choice(cls_indices, size=min_count, replace=False)
            balanced_indices.append(selected_indices)
            
        balanced_indices = np.concatenate(balanced_indices)
        rng.shuffle(balanced_indices)
        
        return X[balanced_indices], Y[balanced_indices]
