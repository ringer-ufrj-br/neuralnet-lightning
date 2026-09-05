"""
Kinematic binning: the Et x |eta| grid the Ringer scheme trains one network per cell of.

The grid is fixed - it is the ATLAS standard binning, the same for every dataset here - so it
lives as constants and there is exactly one `GRID` instance to read them from. Et is compared
in the unit the dataset stores (MeV in every dataset so far) and only converted for display.
"""

import logging
from typing import List, Tuple

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

#: Et bin edges in MeV, one more than there are bins; the last is open-ended.
ET_EDGES: List[float] = [15e3, 20e3, 30e3, 40e3, 50e3, float("inf")]

#: |eta| bin edges, closed at both ends - at or above the last is outside acceptance.
ETA_EDGES: List[float] = [0.0, 0.8, 1.37, 1.54, 2.37, 2.5]

#: Et is stored in MeV and printed in GeV.
ET_SCALE, ET_UNIT = 1e-3, "GeV"


class Binning:
    """The Et x |eta| grid. One instance, `GRID`, built from the constants above."""

    et_edges = ET_EDGES
    eta_edges = ETA_EDGES

    @property
    def n_et_bins(self) -> int:
        """Number of Et bins."""
        return len(self.et_edges) - 1

    @property
    def n_eta_bins(self) -> int:
        """Number of |eta| bins."""
        return len(self.eta_edges) - 1

    @property
    def n_bins(self) -> int:
        """Total regions, i.e. how many networks the grid asks for."""
        return self.n_et_bins * self.n_eta_bins

    @property
    def regions(self) -> List[Tuple[int, int]]:
        """Every (et_bin, eta_bin) pair in row-major order - what a launcher fans out over."""
        return [(et, eta) for et in range(self.n_et_bins) for eta in range(self.n_eta_bins)]

    def validate(self, et_bin: int, eta_bin: int) -> None:
        """
        Checks a region index pair against the grid.

        Raises:
            ValueError: If either index is outside the grid.
        """
        if not 0 <= et_bin < self.n_et_bins:
            raise ValueError(f"❌ et_bin {et_bin} is outside the grid (0-{self.n_et_bins - 1}).")
        if not 0 <= eta_bin < self.n_eta_bins:
            raise ValueError(f"❌ eta_bin {eta_bin} is outside the grid (0-{self.n_eta_bins - 1}).")

    # ------------------------------------------------------------------ index

    def et_bin_index(self, et_mev: np.ndarray) -> np.ndarray:
        """Et bin of each value (MeV), -1 outside the grid."""
        values = np.asarray(et_mev, dtype=np.float64)
        idx = np.searchsorted(self.et_edges[:-1], values, side="right") - 1
        return np.where((values < self.et_edges[0]) | (values >= self.et_edges[-1]), -1, idx)

    def eta_bin_index(self, eta: np.ndarray) -> np.ndarray:
        """|eta| bin of each signed eta, -1 outside the grid."""
        abs_eta = np.abs(np.asarray(eta, dtype=np.float64))
        idx = np.searchsorted(self.eta_edges[:-1], abs_eta, side="right") - 1
        return np.where(abs_eta >= self.eta_edges[-1], -1, idx)

    def filter_expr(self, et_bin: int, eta_bin: int, et_col: str, eta_col: str) -> pl.Expr:
        """
        Rows of one region, as a lazy filter - so out-of-region rows are dropped during the
        parquet scan rather than after loading. Promotes to float64 like the numpy versions
        above, so edge cases land in the same bin either way.
        """
        self.validate(et_bin, eta_bin)
        et = pl.col(et_col).cast(pl.Float64)
        expr = et >= self.et_edges[et_bin]
        if not np.isinf(self.et_edges[et_bin + 1]):
            expr = expr & (et < self.et_edges[et_bin + 1])
        abs_eta = pl.col(eta_col).cast(pl.Float64).abs()
        return expr & (abs_eta >= self.eta_edges[eta_bin]) & (abs_eta < self.eta_edges[eta_bin + 1])

    # ------------------------------------------------------------------ label

    def _et(self, index: int) -> float:
        """The Et edge at `index`, in display units."""
        return self.et_edges[index] * ET_SCALE

    def bin_label(self, et_bin: int, eta_bin: int) -> str:
        """Directory-friendly region label, e.g. 'et2_eta0'."""
        return f"et{et_bin}_eta{eta_bin}"

    def bin_description(self, et_bin: int, eta_bin: int) -> str:
        """Human-readable ranges, e.g. 'Et in [30, 40) GeV, |eta| in [0.00, 0.80)'."""
        return (f"Et in [{self._et(et_bin):g}, {self._et(et_bin + 1):g}) {ET_UNIT}, "
                f"|eta| in [{self.eta_edges[eta_bin]:.2f}, {self.eta_edges[eta_bin + 1]:.2f})")

    def et_range_str(self, et_bin: int, latex: bool = False) -> str:
        """Et column header, e.g. '15 < Et[GeV] < 20' or the open-ended 'Et[GeV] > 50'."""
        lo, hi = self._et(et_bin), self._et(et_bin + 1)
        if latex:
            et = rf"E_T[\mathrm{{{ET_UNIT}}}]"
            return f"${et} > {lo:g}$" if np.isinf(hi) else f"${lo:g} < {et} < {hi:g}$"
        et = f"Et[{ET_UNIT}]"
        return f"{et} > {lo:g}" if np.isinf(hi) else f"{lo:g} < {et} < {hi:g}"

    def et_range_compact(self, et_bin: int) -> str:
        """Short Et header for renders that size columns to content, e.g. '30-40 GeV'."""
        lo, hi = self._et(et_bin), self._et(et_bin + 1)
        return f"> {lo:g} {ET_UNIT}" if np.isinf(hi) else f"{lo:g}-{hi:g} {ET_UNIT}"

    def eta_range_str(self, eta_bin: int, latex: bool = False) -> str:
        """|eta| row header, e.g. '0.00 < |eta| < 0.80'."""
        lo, hi = self.eta_edges[eta_bin], self.eta_edges[eta_bin + 1]
        return rf"${lo:.2f} < |\eta| < {hi:.2f}$" if latex else f"{lo:.2f} < |eta| < {hi:.2f}"

    def eta_range_compact(self, eta_bin: int) -> str:
        """Short |eta| row header, e.g. '0.00-0.80'."""
        return f"{self.eta_edges[eta_bin]:.2f}-{self.eta_edges[eta_bin + 1]:.2f}"


#: The grid. There is only one.
GRID = Binning()
