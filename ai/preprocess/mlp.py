import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ai.preprocess.base import BasePreprocessor, RING

logger = logging.getLogger(__name__)


def _selected_ring_columns(prefix: str = RING) -> List[str]:
    """
    Selected ring columns for MLP training - we selected 1/2 of rings in each layer (fixed,
    not parameterized). Mirrors the reference selection from prior Ringer trainings:

    pre-sample - 8 rings
    EM1 - 64 rings
    EM2 - 8 rings
    EM3 - 8 rings
    Had1 - 4 rings
    Had2 - 4 rings
    Had3 - 4 rings

    Args:
        prefix (str): printf-style column name template with one '%i' placeholder. Defaults
            to the canonical 'ring_%i', so the selection is the same whatever the dataset
            calls its rings.

    Returns:
        List[str]: The 50 selected column names, in ring order.
    """
    # rings presample
    presample = [prefix % iring for iring in range(8 // 2)]

    # EM1 list
    sum_rings = 8
    em1 = [prefix % iring for iring in range(sum_rings, sum_rings + (64 // 2))]

    # EM2 list
    sum_rings = 8 + 64
    em2 = [prefix % iring for iring in range(sum_rings, sum_rings + (8 // 2))]

    # EM3 list
    sum_rings = 8 + 64 + 8
    em3 = [prefix % iring for iring in range(sum_rings, sum_rings + (8 // 2))]

    # HAD1 list
    sum_rings = 8 + 64 + 8 + 8
    had1 = [prefix % iring for iring in range(sum_rings, sum_rings + (4 // 2))]

    # HAD2 list
    sum_rings = 8 + 64 + 8 + 8 + 4
    had2 = [prefix % iring for iring in range(sum_rings, sum_rings + (4 // 2))]

    # HAD3 list
    sum_rings = 8 + 64 + 8 + 8 + 4 + 4
    had3 = [prefix % iring for iring in range(sum_rings, sum_rings + (4 // 2))]

    return presample + em1 + em2 + em3 + had1 + had2 + had3


class PreprocessMLP(BasePreprocessor):
    """
    Baseline Ringer preprocessor: the leading half of every calorimeter layer's rings. It works
    in the canonical `ring_i` vocabulary, so it is identical for a dataset storing one column
    per ring and one storing all 100 in a single list column.

    Column selection and sensor-anomaly cleaning are the inherited BasePreprocessor defaults.
    Its normalisation is the NeuralRinger reference MLP scaling, overriding the base per-event
    norm1: log1p of the ring energies (negative noise clipped to zero), then a per-feature
    StandardScaler fitted on the training rows only.

    A different normalisation is a different model: subclass this, override `fit`/`normalize`
    (`BasePreprocessor.normalize(self, X)` gives the per-event norm1) and register a pipeline
    for it. Keeping it in the class rather than in a config means the normalisation a set of
    checkpoints was trained under is readable from the class that produced them.
    """

    feature_columns = _selected_ring_columns()

    def __init__(self) -> None:
        self.scaler = StandardScaler()

    @staticmethod
    def _log_energies(X: np.ndarray) -> np.ndarray:
        """log1p of the ring energies with negative noise clipped to zero."""
        return np.log1p(np.clip(X, 0.0, None)).astype(np.float32)

    def fit(self, df: pd.DataFrame) -> "PreprocessMLP":
        """Fits the StandardScaler on the log1p-compressed rings. MUST see the training split only."""
        X = self._log_energies(self.extract(df, self.feature_columns))
        logger.info(f"📐 Fitting StandardScaler on {len(X)} training rows...")
        self.scaler.fit(X)
        return self

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """log1p, then the StandardScaler learned in `fit` (which raises NotFittedError before it)."""
        return self.scaler.transform(self._log_energies(X)).astype(np.float32)

