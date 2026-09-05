"""
Labels for datasets whose target is only recoverable from the file name.

The mc25 tables are one file per Monte Carlo sample - Zee for signal, JF17 for background -
with no label column anywhere. Datasets that carry a real label column never reach this module.
"""

import logging
from typing import Dict, List, Sequence

import polars as pl

logger = logging.getLogger(__name__)

#: The mc25 convention: 'Zee' -> 1 (signal), 'JF17' -> 0 (background).
DEFAULT_PATTERNS: Dict[int, List[str]] = {1: ["zee"], 0: ["jf17"]}

#: In descending label order, so matching is deterministic regardless of dict ordering.
PATTERNS: Dict[int, List[str]] = dict(sorted(DEFAULT_PATTERNS.items(), reverse=True))


def label_from_path(file_path: str) -> int:
    """
    The label whose patterns match a file path, case-insensitively.

    Raises:
        ValueError: If the path matches nothing. A mislabelled file would otherwise train a
            network on silently wrong targets, so this is an error rather than a null.
    """
    lowered = file_path.lower()
    for label, names in PATTERNS.items():
        if any(pattern in lowered for pattern in names):
            return label
    raise ValueError(f"❌ Could not determine label for '{file_path}'. Patterns: {PATTERNS}")


def validate_files(files: Sequence[str]) -> None:
    """
    Checks up front that every file resolves to a label, so `label_expr` can run lazily over
    millions of rows without a per-row unknown-path check.

    Raises:
        ValueError: If any file path matches no pattern.
    """
    for file_path in files:
        label_from_path(file_path)


def label_expr(file_path_col: str = "file_path", label_col: str = "label") -> pl.Expr:
    """
    The label as a lazy expression over the file-path column, so the path strings are never
    materialized. Call `validate_files` first: an unmatched path yields null here.
    """
    lower = pl.col(file_path_col).str.to_lowercase()
    expr = pl
    for label, names in PATTERNS.items():
        condition = lower.str.contains(names[0], literal=True)
        for pattern in names[1:]:
            condition = condition | lower.str.contains(pattern, literal=True)
        expr = expr.when(condition).then(label)
    return expr.otherwise(None).cast(pl.Int8).alias(label_col)
