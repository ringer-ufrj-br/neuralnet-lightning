from typing import List

from ai.preprocess.base import BasePreprocessor, RING


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

    Cleaning and normalisation (the per-event norm1) are the inherited defaults.

    A different normalisation is a different model: subclass this, override `normalize` and
    register a pipeline for it. Keeping it in the class rather than in a config means the
    normalisation a set of checkpoints was trained under is readable from the class that
    produced them.
    """

    feature_columns = _selected_ring_columns()
