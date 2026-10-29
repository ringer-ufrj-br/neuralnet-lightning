"""The baseline Ringer input: the leading half of every calorimeter layer's rings, norm1."""

from typing import Any, Dict, List

import numpy as np

from neuralnet_lightning.data.schema import ring_name
from neuralnet_lightning.preprocess.base import BasePreprocessor

#: Rings per calorimeter layer, in storage order: PS, EM1, EM2, EM3, HAD1, HAD2, HAD3.
LAYER_RINGS = (8, 64, 8, 8, 4, 4, 4)


def selected_rings() -> List[str]:
    """
    The leading half of each layer's rings (4 + 32 + 4 + 4 + 2 + 2 + 2 = 50), in ring order.
    Mirrors the reference selection of prior Ringer trainings: the outer rings of a layer carry
    little energy and mostly noise.
    """
    names: List[str] = []
    start = 0
    for n_rings in LAYER_RINGS:
        names += [ring_name(i) for i in range(start, start + n_rings // 2)]
        start += n_rings
    return names


class PreprocessMLP(BasePreprocessor):
    """
    Cleaning and normalisation are the inherited defaults. It works in the canonical `ring_i`
    vocabulary, so it is identical for a dataset storing one column per ring and one storing
    all 100 in a single list column.

    A different normalisation is a different input representation: subclass this and override
    `normalize`, so the normalisation a set of checkpoints was trained under is readable from
    the class that produced them.
    """

    feature_columns = selected_rings()

    def model_kwargs(self, X: np.ndarray) -> Dict[str, Any]:
        """The input dimension, from the preprocessed (N, n_features) matrix."""
        return {"input_dim": int(X.shape[1])}
