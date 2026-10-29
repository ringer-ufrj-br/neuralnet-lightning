"""Et x |eta| kinematic grid; one network is trained per cell."""

import math
from dataclasses import dataclass
from typing import List

import polars as pl

#: Et bin edges in MeV; the last bin is open-ended.
ET_EDGES: List[float] = [15e3, 20e3, 30e3, 40e3, 50e3, float("inf")]

#: |eta| bin edges.
ETA_EDGES: List[float] = [0.0, 0.8, 1.37, 1.54, 2.37, 2.5]

N_ET_BINS, N_ETA_BINS = len(ET_EDGES) - 1, len(ETA_EDGES) - 1

#: Et is stored in MeV and printed in GeV.
ET_SCALE, ET_UNIT = 1e-3, "GeV"

#: Region label of a run over the whole phase space (no --et-bin/--eta-bin).
FULL_PHASE_SPACE = "all"


@dataclass(frozen=True)
class Region:
    """One (et_bin, eta_bin) cell of the grid."""

    et_bin: int
    eta_bin: int

    def __post_init__(self) -> None:
        if not 0 <= self.et_bin < N_ET_BINS:
            raise ValueError(f"❌ et_bin {self.et_bin} is outside the grid (0-{N_ET_BINS - 1}).")
        if not 0 <= self.eta_bin < N_ETA_BINS:
            raise ValueError(f"❌ eta_bin {self.eta_bin} is outside the grid (0-{N_ETA_BINS - 1}).")

    @property
    def label(self) -> str:
        """Directory-friendly label, e.g. 'et2_eta0'."""
        return f"et{self.et_bin}_eta{self.eta_bin}"

    @property
    def description(self) -> str:
        """Human-readable ranges, e.g. 'Et in [30, 40) GeV, |eta| in [0.00, 0.80)'."""
        return (f"Et in [{_et(self.et_bin):g}, {_et(self.et_bin + 1):g}) {ET_UNIT}, "
                f"|eta| in [{ETA_EDGES[self.eta_bin]:.2f}, {ETA_EDGES[self.eta_bin + 1]:.2f})")

    @property
    def cli_args(self) -> str:
        """The `--et-bin/--eta-bin` fragment that selects this region."""
        return f"--et-bin {self.et_bin} --eta-bin {self.eta_bin}"

    def filter_expr(self, et_col: str, eta_col: str) -> pl.Expr:
        """Polars filter selecting this region's rows."""
        et = pl.col(et_col).cast(pl.Float64)
        expr = et >= ET_EDGES[self.et_bin]
        if not math.isinf(ET_EDGES[self.et_bin + 1]):
            expr = expr & (et < ET_EDGES[self.et_bin + 1])
        abs_eta = pl.col(eta_col).cast(pl.Float64).abs()
        return expr & (abs_eta >= ETA_EDGES[self.eta_bin]) & (abs_eta < ETA_EDGES[self.eta_bin + 1])


#: Every region, in row-major order.
REGIONS: List[Region] = [Region(et, eta) for et in range(N_ET_BINS) for eta in range(N_ETA_BINS)]


def _et(index: int) -> float:
    """The Et edge at `index`, in display units."""
    return ET_EDGES[index] * ET_SCALE


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
