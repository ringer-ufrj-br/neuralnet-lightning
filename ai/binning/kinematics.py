"""
Kinematic binning: the Et x |eta| grid the Ringer scheme trains one network per cell of.

The grid is fixed - it is the ATLAS standard binning, the same for every dataset here - so it
lives as module constants. Et is compared in the unit the dataset stores (MeV in every dataset
so far) and only converted for display.
"""

import math
from typing import List, Tuple

import polars as pl

#: Et bin edges in MeV, one more than there are bins; the last is open-ended.
ET_EDGES: List[float] = [15e3, 20e3, 30e3, 40e3, 50e3, float("inf")]

#: |eta| bin edges, closed at both ends - at or above the last is outside acceptance.
ETA_EDGES: List[float] = [0.0, 0.8, 1.37, 1.54, 2.37, 2.5]

N_ET_BINS, N_ETA_BINS = len(ET_EDGES) - 1, len(ETA_EDGES) - 1

#: Every (et_bin, eta_bin) pair in row-major order - what a launcher fans out over.
REGIONS: List[Tuple[int, int]] = [(et, eta) for et in range(N_ET_BINS) for eta in range(N_ETA_BINS)]

#: Et is stored in MeV and printed in GeV.
ET_SCALE, ET_UNIT = 1e-3, "GeV"


def validate(et_bin: int, eta_bin: int) -> None:
    """Raises ValueError if either index is outside the grid."""
    if not 0 <= et_bin < N_ET_BINS:
        raise ValueError(f"❌ et_bin {et_bin} is outside the grid (0-{N_ET_BINS - 1}).")
    if not 0 <= eta_bin < N_ETA_BINS:
        raise ValueError(f"❌ eta_bin {eta_bin} is outside the grid (0-{N_ETA_BINS - 1}).")


def filter_expr(et_bin: int, eta_bin: int, et_col: str, eta_col: str) -> pl.Expr:
    """
    Rows of one region, as a lazy filter - so out-of-region rows are dropped during the
    parquet scan rather than after loading.
    """
    validate(et_bin, eta_bin)
    et = pl.col(et_col).cast(pl.Float64)
    expr = et >= ET_EDGES[et_bin]
    if not math.isinf(ET_EDGES[et_bin + 1]):
        expr = expr & (et < ET_EDGES[et_bin + 1])
    abs_eta = pl.col(eta_col).cast(pl.Float64).abs()
    return expr & (abs_eta >= ETA_EDGES[eta_bin]) & (abs_eta < ETA_EDGES[eta_bin + 1])


def _et(index: int) -> float:
    """The Et edge at `index`, in display units."""
    return ET_EDGES[index] * ET_SCALE


def bin_label(et_bin: int, eta_bin: int) -> str:
    """Directory-friendly region label, e.g. 'et2_eta0'."""
    return f"et{et_bin}_eta{eta_bin}"


def bin_description(et_bin: int, eta_bin: int) -> str:
    """Human-readable ranges, e.g. 'Et in [30, 40) GeV, |eta| in [0.00, 0.80)'."""
    return (f"Et in [{_et(et_bin):g}, {_et(et_bin + 1):g}) {ET_UNIT}, "
            f"|eta| in [{ETA_EDGES[eta_bin]:.2f}, {ETA_EDGES[eta_bin + 1]:.2f})")


def et_range_str(et_bin: int, latex: bool = False) -> str:
    """Et column header: '$15 < E_T[\\mathrm{GeV}] < 20$' in LaTeX, '15-20 GeV' otherwise."""
    lo, hi = _et(et_bin), _et(et_bin + 1)
    if latex:
        et = rf"E_T[\mathrm{{{ET_UNIT}}}]"
        return f"${et} > {lo:g}$" if math.isinf(hi) else f"${lo:g} < {et} < {hi:g}$"
    return f"> {lo:g} {ET_UNIT}" if math.isinf(hi) else f"{lo:g}-{hi:g} {ET_UNIT}"


def eta_range_str(eta_bin: int, latex: bool = False) -> str:
    """|eta| row header: '$0.00 < |\\eta| < 0.80$' in LaTeX, '0.00-0.80' otherwise."""
    lo, hi = ETA_EDGES[eta_bin], ETA_EDGES[eta_bin + 1]
    return rf"${lo:.2f} < |\eta| < {hi:.2f}$" if latex else f"{lo:.2f}-{hi:.2f}"
