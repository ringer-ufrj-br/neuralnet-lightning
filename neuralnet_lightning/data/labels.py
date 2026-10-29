"""Labels derived from the file name, for datasets without a label column (mc25: Zee = signal, JF17 = background)."""

from typing import Sequence

import polars as pl

#: Column holding each row's source file path.
FILE_PATH = "file_path"

#: Label -> case-insensitive file-path pattern, checked in order.
PATTERNS = {1: "zee", 0: "jf17"}


def validate_files(files: Sequence[str]) -> None:
    """Raises ValueError if a file matches no label pattern."""
    for file_path in files:
        if not any(pattern in file_path.lower() for pattern in PATTERNS.values()):
            raise ValueError(f"❌ Could not determine label for '{file_path}'. Patterns: {PATTERNS}")


def label_expr(alias: str) -> pl.Expr:
    """Lazy label expression over the file-path column; null for unmatched paths."""
    lower = pl.col(FILE_PATH).str.to_lowercase()
    expr = pl
    for label, pattern in PATTERNS.items():
        expr = expr.when(lower.str.contains(pattern, literal=True)).then(label)
    return expr.otherwise(None).cast(pl.Int8).alias(alias)
