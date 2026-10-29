"""
Maps a dataset's source columns onto the canonical names `label`, `et`, `eta`,
`ring_0` ... `ring_99` and `row_id`. Other source columns keep their own names.
"""

import glob
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Sequence

import polars as pl

from neuralnet_lightning.data import labels

logger = logging.getLogger(__name__)

#: Canonical column names.
LABEL, ET, ETA, RING, ROW_ID = "label", "et", "eta", "ring_%i", "row_id"

#: Source column exposed as ROW_ID, when present.
ROW_ID_COL = "id"

#: Rings per event, across the pre-sampler, EM1-3 and HAD1-3.
N_RINGS: int = 100


def ring_name(index: int) -> str:
    """Canonical name of a ring feature, e.g. 'ring_37'."""
    return RING % index


@dataclass(frozen=True)
class DatasetSchema:
    """
    Column mapping of one dataset; also the config's `dataset:` block.

    `rings_col` is either a per-ring template containing '%i' ('cl_ring_%i') or a single list
    column holding all rings. `label_col` names the 0/1 target column; when unset, the label
    comes from the file name (see neuralnet_lightning.data.labels). `max_files` caps the files
    read per folder.
    """

    data_path: str
    rings_col: str
    et_col: str
    eta_col: str
    label_col: Optional[str] = None
    max_files: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_files is not None and self.max_files < 1:
            raise ValueError(f"❌ dataset.max_files must be >= 1 (got {self.max_files}).")

    def files(self) -> List[str]:
        """
        Parquet files under `data_path` (a directory, file or glob), sorted, at most
        `max_files` per folder.
        """
        path = self.data_path
        if os.path.isdir(path):
            path = os.path.join(path, "**", "*.parquet")
        per_folder = defaultdict(list)
        for file_path in glob.glob(path, recursive=True):
            if os.path.isfile(file_path):
                per_folder[os.path.dirname(file_path)].append(file_path)
        files = [f for folder in sorted(per_folder) for f in sorted(per_folder[folder])[:self.max_files]]
        logger.info(f"📂 Found {len(files)} parquet file(s) under '{self.data_path}'.")
        return files

    @property
    def rings_are_listed(self) -> bool:
        """Whether all rings live in one nested list column, rather than one column each."""
        return "%i" not in self.rings_col and "%d" not in self.rings_col

    @property
    def needs_file_paths(self) -> bool:
        """Whether the label can only be recovered from the originating file name."""
        return self.label_col is None

    def ring_source_columns(self) -> List[str]:
        """The source columns the rings live in."""
        if self.rings_are_listed:
            return [self.rings_col]
        return [self.rings_col % i for i in range(N_RINGS)]

    # ------------------------------------------------------------------ names

    def canonical_columns(self, source_names: Sequence[str]) -> List[str]:
        """Canonical names available in a frame with `source_names`, then the other source columns."""
        available = set(source_names)
        names: List[str] = []
        if self.rings_are_listed:
            if self.rings_col in available:
                names += [ring_name(i) for i in range(N_RINGS)]
        else:
            names += [ring_name(i) for i in range(N_RINGS) if self.rings_col % i in available]
        if self.et_col in available:
            names.append(ET)
        if self.eta_col in available:
            names.append(ETA)
        if ROW_ID_COL in available:
            names.append(ROW_ID)
        names.append(LABEL)

        hidden = self._aliased_columns() | set(names)
        names += [name for name in source_names if name not in hidden]
        return list(dict.fromkeys(names))

    def _aliased_columns(self) -> set:
        """Source columns exposed only through a canonical alias, plus the file-path column."""
        hidden = set(self.ring_source_columns())
        hidden |= {self.et_col, self.eta_col, ROW_ID_COL, labels.FILE_PATH}
        if self.label_col:
            hidden.add(self.label_col)
        return hidden

    # ------------------------------------------------------------------ exprs

    def select_expr(self, name: str) -> pl.Expr:
        """The expression producing one canonical column."""
        if name == ET:
            return pl.col(self.et_col).alias(ET)
        if name == ETA:
            return pl.col(self.eta_col).alias(ETA)
        if name == ROW_ID:
            return pl.col(ROW_ID_COL).alias(ROW_ID)
        if name == LABEL:
            return self.label_expr()

        prefix = RING.split("%")[0]
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            return self.ring_expr(int(name[len(prefix):]))
        return pl.col(name)

    def ring_expr(self, index: int) -> pl.Expr:
        """Ring `index` under its canonical name, from either storage shape."""
        if not 0 <= index < N_RINGS:
            raise IndexError(f"❌ Ring {index} is out of range for {N_RINGS} rings.")
        alias = ring_name(index)
        if self.rings_are_listed:
            return pl.col(self.rings_col).list.get(index).alias(alias)
        return pl.col(self.rings_col % index).alias(alias)

    def label_expr(self) -> pl.Expr:
        """The canonical Int8 label, from `label_col` or from the file path."""
        if self.label_col:
            return pl.col(self.label_col).cast(pl.Int8).alias(LABEL)
        return labels.label_expr(LABEL)

    # ------------------------------------------------------------------- scan

    def scan(self, files: Sequence[str]) -> pl.LazyFrame:
        """Lazy parquet scan over `files`, with a file-path column."""
        return pl.scan_parquet(list(files), include_file_paths=labels.FILE_PATH, low_memory=True)

    def project(self, frame: pl.LazyFrame, columns: Sequence[str]) -> pl.LazyFrame:
        """Selects the requested canonical columns from a scanned frame."""
        return frame.select([self.select_expr(name) for name in dict.fromkeys(columns)])
