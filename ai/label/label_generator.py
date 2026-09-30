"""
Labels for datasets whose target is only recoverable from the file name.

The mc25 tables are one file per Monte Carlo sample - Zee for signal, JF17 for background -
with no label column anywhere. Datasets that carry a real label column never reach this module.
"""

from typing import Sequence

import polars as pl

#: The mc25 convention, matched case-insensitively against the file path. Checked in this
#: order, so a path matching both is signal.
PATTERNS = {1: "zee", 0: "jf17"}


def validate_files(files: Sequence[str]) -> None:
    """
    Checks up front that every file resolves to a label, so `label_expr` can run lazily over
    millions of rows without a per-row unknown-path check. Raises ValueError otherwise: a
    mislabelled file would train a network on silently wrong targets.
    """
    for file_path in files:
        if not any(pattern in file_path.lower() for pattern in PATTERNS.values()):
            raise ValueError(f"❌ Could not determine label for '{file_path}'. Patterns: {PATTERNS}")


def label_expr(label_col: str) -> pl.Expr:
    """
    The label as a lazy expression over the scan's file-path column, so the path strings are
    never materialized. Call `validate_files` first: an unmatched path yields null here.
    """
    lower = pl.col("file_path").str.to_lowercase()
    expr = pl
    for label, pattern in PATTERNS.items():
        expr = expr.when(lower.str.contains(pattern, literal=True)).then(label)
    return expr.otherwise(None).cast(pl.Int8).alias(label_col)
