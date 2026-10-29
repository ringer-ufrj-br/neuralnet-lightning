"""
Loads a dataset as a polars DataFrame with canonical column names:

    from neuralnet_lightning.data.schema import DatasetSchema
    from neuralnet_lightning.data.loader import load_dataframe
    df = load_dataframe(DatasetSchema(data_path=..., rings_col=..., et_col=..., eta_col=...),
                        columns=["ring_0", "ring_1"])
"""

import logging
from typing import List, Optional, Sequence

import polars as pl

from neuralnet_lightning.binning import Region
from neuralnet_lightning.data.labels import validate_files
from neuralnet_lightning.data.schema import ET, ETA, LABEL, ROW_ID, DatasetSchema

logger = logging.getLogger(__name__)


def load_dataframe(
    schema: DatasetSchema,
    columns: Optional[Sequence[str]] = None,
    region: Optional[Region] = None
) -> pl.DataFrame:
    """
    Loads `columns` (canonical names; None for all) plus et, eta, row_id and label, restricted
    to `region` (None for the whole phase space), in a single streaming scan. The row order is
    deterministic.

    Raises FileNotFoundError when no file matches, KeyError for an unknown column and
    ValueError when the region is empty or a row has no label.
    """
    files = schema.files()
    if not files:
        raise FileNotFoundError(f"❌ No parquet file found under '{schema.data_path}'.")
    if schema.needs_file_paths:
        validate_files(files)

    frame = schema.scan(files)
    available = schema.canonical_columns(frame.collect_schema().names())

    if columns is None:
        columns = [name for name in available if name != LABEL]
    missing = [name for name in columns if name not in available]
    if missing:
        raise KeyError(f"❌ Column(s) {missing} are not in the dataset (canonical names available: "
                       f"{_summarise(available)}).")

    keep: List[str] = list(dict.fromkeys(
        [*columns, *(name for name in (ET, ETA, ROW_ID) if name in available), LABEL]
    ))
    logger.info(f"🔎 Projecting scan down to {len(keep)} canonical column(s).")
    frame = schema.project(frame, keep)

    if region is not None:
        logger.info(f"✂️ Restricting to {region.label} ({region.description})...")
        frame = frame.filter(region.filter_expr(ET, ETA))

    df = frame.collect(engine="streaming")

    if df.is_empty():
        where = f"region {region.label}" if region else "the dataset"
        raise ValueError(f"❌ No rows in {where}.")

    if missing_labels := df[LABEL].null_count():
        raise ValueError(f"❌ {missing_labels} row(s) have no label. Check dataset.label_col in the "
                         f"config against the dataset's actual contents.")

    logger.info(f"   {df.height} rows loaded.")
    return df


def _summarise(names: Sequence[str], limit: int = 8) -> str:
    """`names` truncated to `limit` entries, for error messages."""
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown}, ... ({len(names)} total)"
