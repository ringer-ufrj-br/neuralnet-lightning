"""Baseline Ringer input: the inner half of each calorimeter layer's rings, norm1."""

from typing import Any, Dict, List

import numpy as np

from neuralnet_lightning.data.schema import ring_name
from neuralnet_lightning.preprocess.base import BasePreprocessor

#: Rings per calorimeter layer, in storage order: PS, EM1, EM2, EM3, HAD1, HAD2, HAD3.
LAYER_RINGS = (8, 64, 8, 8, 4, 4, 4)


def selected_rings() -> List[str]:
    """The inner half of each layer's rings (4 + 32 + 4 + 4 + 2 + 2 + 2 = 50), in ring order."""
    names: List[str] = []
    start = 0
    for n_rings in LAYER_RINGS:
        names += [ring_name(i) for i in range(start, start + n_rings // 2)]
        start += n_rings
    return names


class PreprocessMLP(BasePreprocessor):
    """The selected rings, cleaned and normalised by the base class defaults."""

    feature_columns = selected_rings()

    def model_kwargs(self, X: np.ndarray) -> Dict[str, Any]:
        """The input dimension, from the preprocessed (N, n_features) matrix."""
        return {"input_dim": int(X.shape[1])}
