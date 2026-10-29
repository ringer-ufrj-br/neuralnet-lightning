"""Small synthetic parquet datasets in both ring layouts; signal and background are easily separable."""

import os
from typing import Tuple

import numpy as np
import polars as pl
import pytest

from neuralnet_lightning.data.schema import N_RINGS, DatasetSchema

#: The mc25 cell grid (rows, columns) of each calorimeter layer, as stored in cl_cells_*.
CELL_GRIDS = {"presampler": (3, 3), "em1": (3, 15), "em2": (7, 7), "em3": (5, 5),
              "had1": (5, 5), "had2": (5, 5), "had3": (5, 5)}

#: Et range of the synthetic events, in MeV: inside the grid, and leaving the top Et bin empty.
ET_RANGE = (15e3, 45e3)


def make_events(n: int, signal: bool, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(et, eta, rings) of `n` synthetic events of one class."""
    rng = np.random.default_rng(seed)
    et = rng.uniform(*ET_RANGE, n).astype(np.float32)
    eta = rng.uniform(-2.45, 2.45, n).astype(np.float32)
    rings = rng.exponential(1.0, (n, N_RINGS)).astype(np.float32)
    if signal:
        rings[:, 72:80] *= 8.0   # EM2 core
    else:
        rings[:, 88:100] *= 8.0  # hadronic leakage
    return et, eta, rings


def make_cells(n: int, signal: bool, seed: int) -> dict:
    """One List(List(Float64)) column per layer, as mc25 stores the cell images."""
    rng = np.random.default_rng(seed + 1000)
    hot = ("em2",) if signal else ("had1", "had2", "had3")
    columns = {}
    for layer, (h, w) in CELL_GRIDS.items():
        cells = rng.exponential(1.0, (n, h, w)) * (8.0 if layer in hot else 1.0)
        columns[f"cl_cells_{layer}"] = pl.Series(cells.tolist(), dtype=pl.List(pl.List(pl.Float64)))
    return columns


def per_ring_frame(n: int, signal: bool, seed: int) -> pl.DataFrame:
    """
    The mc25 layout: one column per ring, no label column, an extra shower-shape column and the
    cell images.
    """
    et, eta, rings = make_events(n, signal, seed)
    columns = {f"cl_ring_{i}": rings[:, i] for i in range(N_RINGS)}
    return pl.DataFrame({"cl_et": et, "cl_eta": eta, "cl_rhad": np.full(n, 0.1, np.float32), **columns,
                         **make_cells(n, signal, seed)})


@pytest.fixture
def per_ring_schema(tmp_path) -> DatasetSchema:
    """Two files whose names carry the label (Zee = signal, JF17 = background)."""
    folder = tmp_path / "mc25"
    folder.mkdir()
    per_ring_frame(300, signal=True, seed=1).write_parquet(folder / "sample.Zee.parquet")
    per_ring_frame(300, signal=False, seed=2).write_parquet(folder / "sample.JF17.parquet")
    return DatasetSchema(data_path=str(folder), rings_col="cl_ring_%i", et_col="cl_et", eta_col="cl_eta")


@pytest.fixture
def listed_schema(tmp_path) -> DatasetSchema:
    """The mc21 layout: all rings in one list column, a real label column and a row id."""
    sig, bkg = make_events(200, True, 3), make_events(200, False, 4)
    et, eta, rings = (np.concatenate([a, b]) for a, b in zip(sig, bkg))
    target = np.array([1] * 200 + [0] * 200, dtype=np.uint8)
    path = tmp_path / "electron_ringer.parquet"
    os.makedirs(path)
    pl.DataFrame({
        "id": np.arange(len(et), dtype=np.uint64),
        "target": target,
        "Cluster.et": et,
        "Cluster.eta": eta,
        "Cluster.ringsE": rings.tolist(),
    }).with_columns(pl.col("Cluster.ringsE").cast(pl.List(pl.Float32))).write_parquet(path / "00000000.parquet")
    return DatasetSchema(data_path=str(path), rings_col="Cluster.ringsE", et_col="Cluster.et",
                         eta_col="Cluster.eta", label_col="target")
