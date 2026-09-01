import numpy as np
import polars as pl
from typing import List, Optional, Tuple

# Et bin edges in GeV (lower edge of each bin; the last bin is open-ended). `cl_et` in the
# dataset is stored in MeV, so callers must convert (this module does it internally).
ET_BIN_EDGES_GEV: List[float] = [15.0, 20.0, 30.0, 40.0, 50.0]

# |eta| bin edges - closed at both ends (values >= ETA_BIN_EDGES[-1] fall outside ATLAS
# calorimeter acceptance and are excluded, unlike the open-ended top Et bin).
ETA_BIN_EDGES: List[float] = [0.0, 0.8, 1.37, 1.54, 2.37, 2.5]

N_ET_BINS = len(ET_BIN_EDGES_GEV)
N_ETA_BINS = len(ETA_BIN_EDGES) - 1
N_BINS = N_ET_BINS * N_ETA_BINS  # 25


def et_bin_index(et_mev: np.ndarray) -> np.ndarray:
    """
    Computes the Et bin index (0-4) for each value, or -1 if below the lowest edge (15 GeV).
    The top bin (index 4) is open-ended (>= 50 GeV).

    Args:
        et_mev (np.ndarray): Cluster transverse energy in MeV (as stored in `cl_et`).

    Returns:
        np.ndarray: Integer bin indices, same shape as input, -1 for out-of-range values.
    """
    et_gev = np.asarray(et_mev, dtype=np.float64) / 1000.0
    idx = np.searchsorted(ET_BIN_EDGES_GEV, et_gev, side="right") - 1
    idx = np.where(et_gev < ET_BIN_EDGES_GEV[0], -1, idx)
    return idx


def eta_bin_index(eta: np.ndarray) -> np.ndarray:
    """
    Computes the |eta| bin index (0-4) for each value, or -1 if outside [0, 2.5).

    Args:
        eta (np.ndarray): Cluster pseudorapidity, signed (as stored in `cl_eta`).

    Returns:
        np.ndarray: Integer bin indices, same shape as input, -1 for out-of-range values.
    """
    abs_eta = np.abs(np.asarray(eta, dtype=np.float64))
    lower_edges = ETA_BIN_EDGES[:-1]
    idx = np.searchsorted(lower_edges, abs_eta, side="right") - 1
    idx = np.where(abs_eta >= ETA_BIN_EDGES[-1], -1, idx)
    return idx


def bin_filter_expr(et_bin: int, eta_bin: int, et_col: str = "cl_et", eta_col: str = "cl_eta") -> pl.Expr:
    """
    Polars filter expression selecting exactly the rows of one kinematic bin, equivalent to
    `(et_bin_index(cl_et) == et_bin) & (eta_bin_index(cl_eta) == eta_bin)`. Meant for lazy
    queries so out-of-bin rows are dropped during the parquet scan instead of after loading
    the full dataset. Both computations promote to float64 first, like the numpy versions,
    so edge cases land in the same bin either way.

    Args:
        et_bin (int): Et bin index (0-4); the top bin is open-ended.
        eta_bin (int): |eta| bin index (0-4).
        et_col (str): Column with Et in MeV. Defaults to 'cl_et'.
        eta_col (str): Column with signed eta. Defaults to 'cl_eta'.

    Returns:
        pl.Expr: Boolean expression, True for rows inside the bin.
    """
    et_gev = pl.col(et_col).cast(pl.Float64) / 1000.0
    expr = et_gev >= ET_BIN_EDGES_GEV[et_bin]
    if et_bin + 1 < N_ET_BINS:
        expr = expr & (et_gev < ET_BIN_EDGES_GEV[et_bin + 1])

    abs_eta = pl.col(eta_col).cast(pl.Float64).abs()
    expr = expr & (abs_eta >= ETA_BIN_EDGES[eta_bin]) & (abs_eta < ETA_BIN_EDGES[eta_bin + 1])
    return expr


def bin_label(et_bin: int, eta_bin: int) -> str:
    """Directory/file-friendly label for a given (et_bin, eta_bin) pair, e.g. 'et2_eta0'."""
    return f"et{et_bin}_eta{eta_bin}"


def bin_description(et_bin: int, eta_bin: int) -> str:
    """Human-readable range description, e.g. 'Et in [30, 40) GeV, |eta| in [0.00, 0.80)'."""
    et_lo = ET_BIN_EDGES_GEV[et_bin]
    et_hi = ET_BIN_EDGES_GEV[et_bin + 1] if et_bin + 1 < N_ET_BINS else float("inf")
    eta_lo = ETA_BIN_EDGES[eta_bin]
    eta_hi = ETA_BIN_EDGES[eta_bin + 1]
    return f"Et in [{et_lo:g}, {et_hi:g}) GeV, |eta| in [{eta_lo:.2f}, {eta_hi:.2f})"


def all_bins() -> List[Tuple[int, int]]:
    """All 25 (et_bin, eta_bin) index pairs."""
    return [(et, eta) for et in range(N_ET_BINS) for eta in range(N_ETA_BINS)]


def et_range_str(et_bin: int, latex: bool = False) -> str:
    """
    Column header for an Et bin, e.g. '15 < Et[GeV] < 20' (or the open-ended 'Et[GeV] > 50').

    Args:
        et_bin (int): Et bin index (0-4).
        latex (bool): Render with LaTeX math markup instead of plain text. Defaults to False.

    Returns:
        str: The formatted range label.
    """
    lo = ET_BIN_EDGES_GEV[et_bin]
    is_open = et_bin + 1 >= N_ET_BINS
    if latex:
        et = r"E_T[\mathrm{GeV}]"
        return f"${et} > {lo:g}$" if is_open else f"${lo:g} < {et} < {ET_BIN_EDGES_GEV[et_bin + 1]:g}$"
    if is_open:
        return f"Et[GeV] > {lo:g}"
    return f"{lo:g} < Et[GeV] < {ET_BIN_EDGES_GEV[et_bin + 1]:g}"


def eta_range_str(eta_bin: int, latex: bool = False) -> str:
    """
    Row header for an |eta| bin, e.g. '0.00 < |eta| < 0.80'.

    Args:
        eta_bin (int): |eta| bin index (0-4).
        latex (bool): Render with LaTeX math markup instead of plain text. Defaults to False.

    Returns:
        str: The formatted range label.
    """
    lo = ETA_BIN_EDGES[eta_bin]
    hi = ETA_BIN_EDGES[eta_bin + 1]
    if latex:
        return rf"${lo:.2f} < |\eta| < {hi:.2f}$"
    return f"{lo:.2f} < |eta| < {hi:.2f}"
