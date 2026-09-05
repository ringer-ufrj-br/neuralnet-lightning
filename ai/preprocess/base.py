"""
Preprocessing: raw parquet columns in, model input out.

Two halves, in the order the data flows through them:

* `DatasetSchema` maps a *particular* dataset's columns onto a fixed canonical vocabulary -
  `label`, `et`, `eta`, `ring_0 ... ring_99`, `row_id` - as polars expressions applied inside
  the lazy scan. A ring stored as element `i` of a nested list therefore costs the same to
  read as one in its own column, and columns nobody asked for are never touched. Only what
  differs between datasets is configurable, so the block is flat:

      dataset:
        data_path: .../electron_ringer.parquet
        rings_col: "TrigEMClusterContainer.ringsE"
        et_col:    "TrigEMClusterContainer.et"
        eta_col:   "TrigEMClusterContainer.eta"
        label_col: target

  The defaults describe the mc25 layout, so a config that declares no `dataset:` block
  describes those tables.

* `BasePreprocessor` turns those canonical columns into the array a model consumes. The same
  architecture on a different dataset is a different model: it gets its own config, and its
  own preprocessor module when the transform differs too.
"""

from dataclasses import asdict, dataclass
import os
import logging
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


#: Canonical names. Et and eta keep the source column's unit (MeV / signed), matching the
#: edges in ai.binning.kinematics. Rings are addressed by index, never by the source dataset's
#: naming, which is what makes the preprocessors dataset agnostic.
LABEL, ET, ETA, RING, ROW_ID = "label", "et", "eta", "ring_%i", "row_id"

#: Stable per-row key, where a dataset has one. A dataset without it simply has no column of
#: this name, and ROW_ID is then never offered - the fingerprint falls back to Et.
ROW_ID_COL = "id"

#: The Ringer layout: 100 rings across pre-sample, EM1-3 and HAD1-3. The per-layer split is
#: spelled out in ai/preprocess/mlp.py, where the feature selection needs it.
N_RINGS: int = 100


def ring_name(index: int) -> str:
    """Canonical name of a ring feature, e.g. 'ring_37'."""
    return RING % index


@dataclass(frozen=True)
class DatasetSchema:
    """
    One dataset's on-disk layout. Defaults describe the mc25 tables.

    `rings_col` covers both storage shapes, told apart by the placeholder: a name containing
    '%i' is a printf template for one scalar column per ring ('cl_ring_%i'), anything else is
    a single nested list column holding all 100.

    `label_col` names the column holding the 0/1 target. Left unset, the dataset has no such
    column and the label is read off the source file name instead (see ai.label.label_generator).
    """

    data_path: Optional[str] = None
    max_files: Optional[int] = None
    rings_col: str = "cl_ring_%i"
    et_col: str = "cl_et"
    eta_col: str = "cl_eta"
    label_col: Optional[str] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DatasetSchema":
        """
        Builds a schema from a config's `dataset:` block. `data_path` and `max_files` are
        also accepted at the top level of the config, and every field defaults to the mc25
        layout, so a config that declares no `dataset:` block describes those tables.
        """
        block = dict(config.get("dataset") or {})
        return cls(
            data_path=block.get("data_path", config.get("data_path")),
            max_files=block.get("max_files", config.get("max_files")),
            rings_col=str(block.get("rings_col", "cl_ring_%i")),
            et_col=str(block.get("et_col", "cl_et")),
            eta_col=str(block.get("eta_col", "cl_eta")),
            label_col=block.get("label_col"),
        )

    def describe(self) -> Dict[str, Any]:
        """JSON form, recorded in the manifest so a run's layout is recoverable."""
        return asdict(self)

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
        """
        The canonical vocabulary available over a scanned frame: the rings, the kinematics, the
        label, plus any source column exposed under its own name (the cell images).

        This is what a preprocessor's `required_columns` is handed, so preprocessors never see
        - and never have to recognise - a source column name.
        """
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
        """
        Columns of the scanned frame reachable only through a canonical alias, plus the scan's
        own bookkeeping - never offered as features.
        """
        hidden = set(self.ring_source_columns())
        hidden |= {self.et_col, self.eta_col, ROW_ID_COL, "file_path"}
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
        """
        Ring `index` under its canonical name, from either storage shape.

        Raises:
            IndexError: If the index is outside the Ringer layout.
        """
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

        from ai.label.label_generator import label_expr

        return label_expr(label_col=LABEL)

    # ------------------------------------------------------------------- scan

    def scan(self, files: Sequence[str]) -> pl.LazyFrame:
        """Opens the lazy scan over the data files. Nothing is read here."""
        return pl.scan_parquet(list(files), include_file_paths="file_path", low_memory=True)

    def project(self, frame: pl.LazyFrame, columns: Sequence[str]) -> pl.LazyFrame:
        """
        Rewrites a scanned frame down to the requested canonical columns. Pruning happens here,
        inside the scan: the raw files carry hundreds of columns, most of them nested images.
        """
        return frame.select([self.select_expr(name) for name in dict.fromkeys(columns)])


class BasePreprocessor:
    """
    The contract every preprocessor honours, plus the parts that are the same for all of them
    (persistence, label extraction, fit_transform).

    The baseline preprocessor is a column selection: set `feature_columns` and the inherited
    `required_columns` / `transform` do the rest - extract those columns, zero the sensor
    anomalies, normalise each event by its own total. PreprocessMLP is exactly this.

    A preprocessor whose input is not a flat slice of dataset columns (the CNN2D image
    builder, the Fused rings+cells concatenation) instead overrides `transform`, and usually
    `required_columns`, and leaves `feature_columns` as None.

    Either kind overrides `fit` only if it has state to learn from the training split (a
    scaler, a mean, ...). The default `fit` is a no-op.

    Persistence is joblib pickling of the whole instance, so anything stored on `self` in
    `fit` is restored by `load` - no per-preprocessor save/load code is needed.
    """

    #: Dataset columns this preprocessor consumes, in feature order. When set, it drives the
    #: default `required_columns` and `transform`. Left None by preprocessors that build their
    #: input some other way and override `transform`.
    feature_columns: Optional[List[str]] = None

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Normalises each sample by the absolute sum of its own features, so the network sees the
        shape of the energy deposition and not its absolute scale (the Et binning already
        accounts for scale). This is the NeuralRinger reference normalisation,
        r'_i = r_i / |sum_j r_j|.

        Call this as the last step of `transform`. It is done here, once per dataset, rather
        than inside the model, where it would be recomputed for every batch of every epoch on
        data that never changes.

        Args:
            X (np.ndarray): Feature array, first dimension being the batch.

        Returns:
            np.ndarray: Float32 array of the same shape, each sample scaled by its own total.
                Samples summing to zero are left as they are rather than turned into NaNs.
        """
        axes = tuple(range(1, X.ndim))
        if not axes:
            return X.astype(np.float32)
        total = np.abs(X.sum(axis=axes, keepdims=True))
        return (X / np.where(total == 0.0, 1.0, total)).astype(np.float32)

    def required_columns(self, available: List[str]) -> Optional[List[str]]:
        """
        Declares which dataset columns this preprocessor consumes, so the loader can prune the
        rest during the parquet scan. The raw files carry 300+ columns, most of them nested
        calorimeter images; reading them all is what used to exhaust memory.

        The default returns `feature_columns` (None when unset, i.e. load everything).

        Args:
            available (List[str]): Column names present in the dataset files.

        Returns:
            Optional[List[str]]: Columns to load, or None to load everything.
        """
        return self.feature_columns

    def extract(self, df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        """
        Pulls `cols` out of `df` as a clean float32 matrix: NaNs and the -999 sensor-anomaly
        marker are zeroed. A missing column raises KeyError here rather than being silently
        worked around - a wrong column set is a bug, not something to recover from.

        Args:
            df (pd.DataFrame): Input rows.
            cols (List[str]): Column names to extract, in order.

        Returns:
            np.ndarray: Cleaned float32 array, first dimension being the batch. Not normalised.
        """
        X = df[cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0)
        return np.where(X == -999, 0.0, X)

    def fit(self, df: pd.DataFrame) -> "BasePreprocessor":
        """
        Learns whatever state this preprocessor needs from the training split. The default is
        a no-op, for preprocessors that are pure functions of their input.

        Args:
            df (pd.DataFrame): Training rows.

        Returns:
            BasePreprocessor: self, for chaining.
        """
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Turns a DataFrame into the model's input array.

        The default is the baseline path: extract `feature_columns`, clean sensor anomalies
        and normalise each event by its own total. Preprocessors that build their input some
        other way override this; those must leave `feature_columns` as None.

        Args:
            df (pd.DataFrame): Rows to transform.

        Returns:
            np.ndarray: Float32 features, first dimension being the batch.
        """
        if self.feature_columns is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set feature_columns or override transform()."
            )
        cols = self.feature_columns
        logger.info(f"🧪 Extracting {len(cols)} features ({cols[0]} ... {cols[-1]})...")
        return self.normalize(self.extract(df, cols))

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fits on the given rows and transforms them in one call.

        Args:
            df (pd.DataFrame): Training rows.

        Returns:
            np.ndarray: The transformed features.
        """
        return self.fit(df).transform(df)

    def save(self, filepath: str) -> str:
        """
        Persists the fitted preprocessor alongside the trained checkpoints.

        Args:
            filepath (str): Destination path (.joblib).

        Returns:
            str: The written path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"💾 Saved preprocessor to: {filepath}")
        return filepath

    @staticmethod
    def load(filepath: str) -> "BasePreprocessor":
        """
        Loads a preprocessor previously written by save().

        Args:
            filepath (str): Path to the .joblib file.

        Returns:
            BasePreprocessor: The restored instance.
        """
        preprocessor = joblib.load(filepath)
        logger.info(f"📂 Loaded preprocessor from: {filepath}")
        return preprocessor

    def get_labels(self, df: pd.DataFrame, label_col: str = 'label') -> Optional[np.ndarray]:
        """
        Extracts target labels from the DataFrame.

        Args:
            df (pd.DataFrame): Input rows.
            label_col (str): Label column name. Defaults to 'label'.

        Returns:
            Optional[np.ndarray]: Float32 labels, or None when the column is absent.
        """
        if label_col in df.columns:
            return df[label_col].values.astype(np.float32)
        logger.warning(f"⚠️ Label column '{label_col}' not found in DataFrame.")
        return None
