"""
Cross-validation table ("tabelao") builder.

The canonical artefact is a **long/tidy** table: one purely numeric row per
(model, et_bin, eta_bin, fold, operating_point). Every render - the wide mean+/-std table,
the LaTeX fragment and the matplotlib figure - is derived from it, so there is exactly one
place where numbers are produced and several places where they are formatted.

Layout of the rendered table mirrors the ATLAS/Ringer convention: rows are |eta| regions,
column groups are Et regions, and each group carries the PD / SP / FA triplet. When more than
one model has been evaluated, each |eta| region gets one row per model, so architectures are
compared side by side under identical working points.
"""

import glob
import json
import logging
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ai.evaluation.metrics import sp_index
from ai.binning.kinematics import (
    ET_BIN_EDGES_GEV,
    ETA_BIN_EDGES,
    N_ET_BINS,
    et_range_str,
    eta_range_str,
)

logger = logging.getLogger(__name__)

#: Column contract of the canonical long table. Anything writing `folds_long.csv` must emit these.
LONG_COLUMNS: List[str] = [
    "model", "et_bin", "eta_bin", "fold", "operating_point",
    "target_pd", "threshold", "pd", "fa", "sp", "auc_roc", "auc_pr",
    "n_signal", "n_background",
]

#: Label used for the margin row/column in the rendered tables.
INTEGRATED_LABEL: str = "Integrated"

#: Sentinel bin index marking a row/column that pools every bin of that axis (the table's
#: margins). Kept out of the canonical long CSV and only materialised for rendering.
INTEGRATED: int = -1

#: Metrics shown in each Et column group, in order, with their table headers.
TABLE_METRICS: List[Tuple[str, str, str]] = [
    ("pd", "PD[%]", r"$P_D$[\%]"),
    ("sp", "SP[%]", r"$SP$[\%]"),
    ("fa", "FA[%]", r"$F_A$[\%]"),
]


def _et_label(et_bin: Optional[int], compact: bool = False, latex: bool = False) -> str:
    """
    Column-group label for an Et bin, for the INTEGRATED margin, or for an unbinned run.

    The compact form is used by the figure render, where matplotlib sizes columns to their
    widest cell and a full '15 < Et[GeV] < 20' would either overflow its cell or blow the
    column up.

    Args:
        et_bin (Optional[int]): Et bin index, INTEGRATED for the margin, or None for an
            ungridded run.
        compact (bool): Use the short form ('30-40 GeV'). Defaults to False.
        latex (bool): Render with LaTeX math markup. Defaults to False.

    Returns:
        str: The label.
    """
    if et_bin is None:
        return "all $E_T$" if latex else "all Et"
    if et_bin == INTEGRATED:
        return rf"\textbf{{{INTEGRATED_LABEL}}}" if latex else INTEGRATED_LABEL
    if latex:
        return et_range_str(et_bin, latex=True)
    if not compact:
        return et_range_str(et_bin)
    lo = ET_BIN_EDGES_GEV[et_bin]
    if et_bin + 1 >= N_ET_BINS:
        return f"> {lo:g} GeV"
    return f"{lo:g}-{ET_BIN_EDGES_GEV[et_bin + 1]:g} GeV"


def _eta_label(eta_bin: Optional[int], compact: bool = False, latex: bool = False) -> str:
    """
    Row label for an |eta| bin, for the INTEGRATED margin, or for an unbinned run.

    Args:
        eta_bin (Optional[int]): |eta| bin index, INTEGRATED for the margin, or None for an
            ungridded run.
        compact (bool): Use the short form ('0.00-0.80'). Defaults to False.
        latex (bool): Render with LaTeX math markup. Defaults to False.

    Returns:
        str: The label.
    """
    if eta_bin is None:
        return r"all $|\eta|$" if latex else "all |eta|"
    if eta_bin == INTEGRATED:
        return rf"\textbf{{{INTEGRATED_LABEL}}}" if latex else INTEGRATED_LABEL
    if latex:
        return eta_range_str(eta_bin, latex=True)
    if not compact:
        return eta_range_str(eta_bin)
    return f"{ETA_BIN_EDGES[eta_bin]:.2f}-{ETA_BIN_EDGES[eta_bin + 1]:.2f}"


def discover_regions(
    results_root: str = "results",
    model_names: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """
    Inventories what exists on disk under a results tree: which models, which kinematic
    regions, how many folds were trained and how many were evaluated.

    Discovery is anchored on `artifacts/manifest.json` - the file `train` writes for every
    region - rather than on the metrics, so a region that was trained but never evaluated is
    still found and can be reported as missing instead of silently leaving a hole in the table.
    The model name comes from the manifest itself, not from the directory name.

    Args:
        results_root (str): Root results directory. Defaults to 'results'.
        model_names (Optional[Sequence[str]]): Restrict to these model subdirectories.
            Scans every model when None.

    Returns:
        pd.DataFrame: One row per region with columns 'model', 'region', 'path',
        'folds_trained', 'folds_evaluated' and 'evaluated'. Empty when nothing was trained.
    """
    targets = list(model_names) if model_names else ["*"]
    manifest_paths = sorted({
        path
        for target in targets
        for path in glob.glob(
            os.path.join(results_root, target, "**", "artifacts", "manifest.json"),
            recursive=True
        )
    })

    rows = []
    for manifest_path in manifest_paths:
        region_dir = os.path.dirname(os.path.dirname(manifest_path))
        try:
            with open(manifest_path) as handle:
                manifest = json.load(handle)
        except Exception as exc:
            logger.warning(f"⚠️ Skipping unreadable manifest '{manifest_path}': {exc}")
            continue

        sidecars = glob.glob(os.path.join(region_dir, "checkpoints", "fold_*.json"))
        long_path = os.path.join(region_dir, "metrics", "folds_long.csv")

        folds_evaluated = 0
        if os.path.exists(long_path):
            try:
                folds_evaluated = int(pd.read_csv(long_path)["fold"].nunique())
            except Exception:
                folds_evaluated = 0

        rows.append({
            "model": manifest.get("model", os.path.basename(region_dir)),
            "region": manifest.get("region", os.path.basename(region_dir)),
            "path": region_dir,
            "folds_trained": len(sidecars),
            "folds_evaluated": folds_evaluated,
            "evaluated": folds_evaluated > 0,
        })

    return pd.DataFrame(rows, columns=[
        "model", "region", "path", "folds_trained", "folds_evaluated", "evaluated"
    ])


def log_inventory(inventory: pd.DataFrame, results_root: str = "results") -> None:
    """
    Logs what was found on disk, one line per region, and spells out the exact command that
    fills each gap. A table with unexplained holes is worse than no table.

    Args:
        inventory (pd.DataFrame): As produced by discover_regions().
        results_root (str): Root results directory, for the message when nothing was found.
    """
    if inventory.empty:
        logger.warning(
            f"⚠️ No trained region found under '{results_root}'. Nothing to report — "
            "run `train` (and then `evaluate`) first."
        )
        return

    logger.info(f"🔎 Found {len(inventory)} trained region(s) under '{results_root}':")
    for _, entry in inventory.iterrows():
        status = (
            f"{entry.folds_evaluated}/{entry.folds_trained} folds evaluated"
            if entry.evaluated else "NOT EVALUATED"
        )
        logger.info(f"   {entry.model:<8} {entry.region:<18} {status:<24} {entry.path}")

    pending = inventory[~inventory["evaluated"]]
    for _, entry in pending.iterrows():
        region_args = _region_cli_args(entry.region)
        logger.warning(
            f"⚠️ {entry.model} / {entry.region} was trained but never evaluated, so it is "
            f"missing from the table. Fix with: "
            f"python ai/run.py evaluate --config <config> {region_args}".rstrip()
        )

    partial = inventory[inventory["evaluated"] & (inventory.folds_evaluated < inventory.folds_trained)]
    for _, entry in partial.iterrows():
        logger.warning(
            f"⚠️ {entry.model} / {entry.region}: {entry.folds_trained} folds trained but only "
            f"{entry.folds_evaluated} evaluated. Re-run `evaluate` for this region so the "
            "spread covers every fold."
        )


def _region_cli_args(region: str) -> str:
    """
    Turns a region label back into the `--et-bin/--eta-bin` arguments that select it.

    Args:
        region (str): Region label, e.g. 'et2_eta0' or 'full phase space'.

    Returns:
        str: The CLI fragment, or '' for an unbinned region.
    """
    match = re.fullmatch(r"et(\d+)_eta(\d+)", region)
    if not match:
        return ""
    return f"--et-bin {match.group(1)} --eta-bin {match.group(2)}"


def collect(
    results_root: str = "results",
    model_names: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """
    Scans a results tree for per-region `metrics/folds_long.csv` files and concatenates them.

    Both layouts are picked up: the ungridded `results/<MODEL>/metrics/` (et_bin/eta_bin are
    NaN, i.e. one network over the whole phase space) and the 5x5 grid
    `results/<MODEL>/et<i>_eta<j>/metrics/`.

    Args:
        results_root (str): Root results directory. Defaults to 'results'.
        model_names (Optional[Sequence[str]]): Restrict to these model subdirectories
            (e.g. ['MLP', 'CNN2D']). Scans every model when None.

    Returns:
        pd.DataFrame: Concatenated long table (empty frame with LONG_COLUMNS if nothing found).
    """
    targets = list(model_names) if model_names else ["*"]
    patterns = [
        os.path.join(results_root, target, "**", "metrics", "folds_long.csv")
        for target in targets
    ]
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern, recursive=True)})

    frames = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            logger.warning(f"⚠️ Skipping unreadable table '{path}': {exc}")
            continue
        missing = [col for col in LONG_COLUMNS if col not in frame.columns]
        if missing:
            logger.warning(f"⚠️ Skipping '{path}': missing columns {missing}.")
            continue
        frames.append(frame[LONG_COLUMNS])

    if not frames:
        logger.warning(f"⚠️ No 'folds_long.csv' found under {patterns}. Run `evaluate` first.")
        return pd.DataFrame(columns=LONG_COLUMNS)

    long_df = pd.concat(frames, ignore_index=True)

    gridded = long_df["et_bin"].notna().any()
    ungridded = long_df["et_bin"].isna().any()
    if gridded and ungridded:
        logger.warning(
            "⚠️ Both binned and unbinned regions were found. The table is built from the "
            "binned ones; the whole-phase-space rows are kept in the long CSV but not rendered, "
            "since they do not belong to any Et/|eta| cell."
        )

    logger.info(
        f"📚 Collected {len(long_df)} rows from {len(frames)} region(s): "
        f"{sorted(long_df['model'].unique())}, {long_df['fold'].nunique()} fold(s), "
        f"{long_df['operating_point'].nunique()} operating point(s)."
    )
    return long_df


def resolve_models(agg: pd.DataFrame, model_names: Optional[Sequence[str]] = None) -> List[str]:
    """
    Determines which models appear as rows, and in what order.

    An explicit list wins and is honoured verbatim (so `--models CNN2D,MLP` puts the CNN first),
    dropping any name that has no evaluated region; otherwise every model present is used,
    sorted for a stable table across runs.

    Args:
        agg (pd.DataFrame): Aggregate as produced by aggregate().
        model_names (Optional[Sequence[str]]): Requested model order.

    Returns:
        List[str]: Model names, in row order.
    """
    present = set(agg["model"].unique())
    if model_names:
        requested = [name for name in model_names if name in present]
        for name in model_names:
            if name not in present:
                logger.warning(f"⚠️ Model '{name}' has no evaluated region; leaving it out of the table.")
        return requested
    return sorted(present)


def _pool(long_df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """
    Pools per-region efficiencies into one row per key group, weighting by population.

    Each region's network carries its own threshold, so the integrated efficiency is the
    ratio of summed counts, not the average of the per-region rates:
    PD = sum(TP) / sum(P) = sum(pd_r * n_signal_r) / sum(n_signal_r), and likewise for FA
    over the background counts. Averaging the rates directly would let a sparsely populated
    bin weigh as much as a dense one.

    Pooling happens **per fold**, before any aggregation, so the spread quoted for the
    integrated row is the real fold-to-fold spread of the pooled number.

    Args:
        long_df (pd.DataFrame): Long table rows to pool.
        keys (List[str]): Grouping columns (always including model, fold, operating_point).

    Returns:
        pd.DataFrame: One pooled row per group, in long-table shape.
    """
    working = long_df.assign(
        _tp=long_df["pd"] * long_df["n_signal"],
        _fp=long_df["fa"] * long_df["n_background"],
    )
    pooled = working.groupby(keys, dropna=False).agg(
        n_signal=("n_signal", "sum"),
        n_background=("n_background", "sum"),
        target_pd=("target_pd", "first"),
        _tp=("_tp", "sum"),
        _fp=("_fp", "sum"),
    ).reset_index()

    pooled["pd"] = np.where(pooled["n_signal"] > 0, pooled["_tp"] / pooled["n_signal"], 0.0)
    pooled["fa"] = np.where(pooled["n_background"] > 0, pooled["_fp"] / pooled["n_background"], 0.0)
    pooled["sp"] = sp_index(pooled["pd"], pooled["fa"])

    # Meaningless once pooled: each region had its own threshold, and the AUCs cannot be
    # combined from summary numbers. Left as NaN rather than silently averaged.
    pooled["threshold"] = np.nan
    pooled["auc_roc"] = np.nan
    pooled["auc_pr"] = np.nan

    return pooled.drop(columns=["_tp", "_fp"])


def integrate(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pools every kinematic region into the phase-space total: one row per
    (model, fold, operating point), marked with et_bin/eta_bin = INTEGRATED.

    Rendered as its own table rather than as a margin of the per-region grid, so the grid
    keeps the reference layout and the integrated numbers stay readable on their own.

    Args:
        long_df (pd.DataFrame): Long table as produced by collect().

    Returns:
        pd.DataFrame: Pooled rows in long-table shape (empty when there is nothing to pool).
    """
    binned = long_df[long_df["et_bin"].notna() & long_df["eta_bin"].notna()]
    source = binned if not binned.empty else long_df
    if source.empty:
        return pd.DataFrame(columns=LONG_COLUMNS)

    pooled = _pool(source, ["model", "fold", "operating_point"])
    pooled["et_bin"] = INTEGRATED
    pooled["eta_bin"] = INTEGRATED
    return pooled[LONG_COLUMNS]


def aggregate(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduces the long table over folds into mean/std per (model, region, operating point).

    PD/SP/FA are converted to percent here - the tables are always quoted in percent, and doing
    the conversion once at aggregation keeps the renderers free of unit logic.

    Args:
        long_df (pd.DataFrame): Long table as produced by collect().

    Returns:
        pd.DataFrame: One row per (model, et_bin, eta_bin, operating_point) with
        `<metric>_mean` / `<metric>_std` columns (in %) plus `n_folds`.
    """
    if long_df.empty:
        return pd.DataFrame()

    keys = ["model", "et_bin", "eta_bin", "operating_point"]
    working = long_df.copy()
    for column in ("pd", "sp", "fa"):
        working[column] = working[column] * 100.0

    grouped = working.groupby(keys, dropna=False)
    agg = grouped.agg(
        pd_mean=("pd", "mean"), pd_std=("pd", "std"),
        sp_mean=("sp", "mean"), sp_std=("sp", "std"),
        fa_mean=("fa", "mean"), fa_std=("fa", "std"),
        auc_roc_mean=("auc_roc", "mean"), auc_roc_std=("auc_roc", "std"),
        n_folds=("fold", "nunique"),
    ).reset_index()

    # A single fold yields std=NaN; 0.0 is the honest reading (no spread observed).
    for column in [c for c in agg.columns if c.endswith("_std")]:
        agg[column] = agg[column].fillna(0.0)

    return agg


def format_cell(mean: float, std: float, decimals: int = 2, latex: bool = False) -> str:
    """
    Formats one mean+/-std cell.

    Args:
        mean (float): Mean value.
        std (float): Standard deviation.
        decimals (int): Decimal places. Defaults to 2.
        latex (bool): Emit a LaTeX '\\pm' instead of the unicode sign. Defaults to False.

    Returns:
        str: The formatted cell, or '--' when the value is missing.
    """
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "--"
    separator = r" $\pm$ " if latex else "±"
    return f"{mean:.{decimals}f}{separator}{std:.{decimals}f}"


def _resolve_regions(
    subset: pd.DataFrame
) -> Tuple[List[Optional[int]], List[Optional[int]], pd.DataFrame]:
    """
    Works out which |eta| rows and Et column groups the table has.

    Args:
        subset (pd.DataFrame): Aggregate rows for a single operating point.

    Returns:
        Tuple[List[Optional[int]], List[Optional[int]], pd.DataFrame]: (eta_bins, et_bins,
        ungridded rows). For a run with no kinematic binning both lists hold a single None,
        which renders as one 'all Et' / 'all |eta|' cell.
    """
    eta_bins: List[Optional[int]] = sorted({int(v) for v in subset["eta_bin"].dropna().unique()})
    et_bins: List[Optional[int]] = sorted({int(v) for v in subset["et_bin"].dropna().unique()})
    ungridded = subset[subset["et_bin"].isna() | subset["eta_bin"].isna()]

    if not eta_bins and not et_bins and not ungridded.empty:
        return [None], [None], ungridded
    return eta_bins, et_bins, ungridded


def _lookup(
    subset: pd.DataFrame,
    ungridded: pd.DataFrame,
    model: str,
    et_bin: Optional[int],
    eta_bin: Optional[int]
) -> Optional[pd.Series]:
    """
    Finds the aggregate row for one (model, Et bin, |eta| bin) cell.

    Args:
        subset (pd.DataFrame): Aggregate rows for a single operating point.
        ungridded (pd.DataFrame): The subset's unbinned rows.
        model (str): Model name.
        et_bin (Optional[int]): Et bin index, or None for the unbinned case.
        eta_bin (Optional[int]): |eta| bin index, or None for the unbinned case.

    Returns:
        Optional[pd.Series]: The matching row, or None when this model has no result there.
    """
    if et_bin is None or eta_bin is None:
        match = ungridded[ungridded["model"] == model]
    else:
        match = subset[
            (subset["model"] == model)
            & (subset["et_bin"] == et_bin)
            & (subset["eta_bin"] == eta_bin)
        ]
    return None if match.empty else match.iloc[0]


def _metric_cells(entry: Optional[pd.Series], decimals: int, latex: bool) -> List[str]:
    """
    Formats one cell per table metric for a single aggregate row.

    Args:
        entry (Optional[pd.Series]): Aggregate row, or None when the cell has no result.
        decimals (int): Decimal places.
        latex (bool): Emit LaTeX-styled cells.

    Returns:
        List[str]: One formatted string per entry of TABLE_METRICS.
    """
    if entry is None:
        return ["--"] * len(TABLE_METRICS)
    return [
        format_cell(entry[f"{metric}_mean"], entry[f"{metric}_std"], decimals, latex)
        for metric, _, _ in TABLE_METRICS
    ]


def _metric_columns(group_labels: Sequence[str], latex: bool) -> pd.MultiIndex:
    """
    Builds the (group, metric) column index shared by every rendered table.

    Args:
        group_labels (Sequence[str]): Column group labels (Et regions, or operating points).
        latex (bool): Use the LaTeX metric headers.

    Returns:
        pd.MultiIndex: Two-level column index.
    """
    return pd.MultiIndex.from_tuples(
        [
            (label, tex if latex else plain)
            for label in group_labels
            for _, plain, tex in TABLE_METRICS
        ],
        names=["", ""],
    )


def build_wide(
    agg: pd.DataFrame,
    operating_point: str,
    model_names: Optional[Sequence[str]] = None,
    decimals: int = 2,
    compact_labels: bool = False,
    latex: bool = False
) -> pd.DataFrame:
    """
    Pivots the aggregate into the printed table shape for a single operating point:
    one row per (|eta| region, model), one column per (Et region, metric) pair.

    Args:
        agg (pd.DataFrame): Aggregate as produced by aggregate().
        operating_point (str): Working point to render (e.g. 'tight').
        model_names (Optional[Sequence[str]]): Models to show, in row order. Every evaluated
            model when None.
        decimals (int): Decimal places in the formatted cells. Defaults to 2.
        compact_labels (bool): Use short region labels, for renders that size columns to content.
        latex (bool): Emit LaTeX-styled labels and cells. Defaults to False.

    Returns:
        pd.DataFrame: Table with a ('Det. Region', 'Model') MultiIndex on the rows and a
        (Et range, metric) MultiIndex on the columns. Empty frame when nothing matches.
    """
    subset = agg[
        (agg["operating_point"] == operating_point)
        & (agg["et_bin"] != INTEGRATED)
        & (agg["eta_bin"] != INTEGRATED)
    ]
    if subset.empty:
        return pd.DataFrame()

    models = resolve_models(subset, model_names)
    if not models:
        return pd.DataFrame()

    eta_bins, et_bins, ungridded = _resolve_regions(subset)
    columns = _metric_columns([_et_label(et, compact_labels, latex) for et in et_bins], latex)

    rows, index = [], []
    for eta in eta_bins:
        for model in models:
            index.append((_eta_label(eta, compact_labels, latex), model))
            row: List[str] = []
            for et in et_bins:
                row.extend(_metric_cells(_lookup(subset, ungridded, model, et, eta), decimals, latex))
            rows.append(row)

    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=["Det. Region", "Model"]),
        columns=columns
    )


def build_integrated_wide(
    agg: pd.DataFrame,
    model_names: Optional[Sequence[str]] = None,
    decimals: int = 2,
    latex: bool = False,
    operating_points: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """
    Pivots the phase-space totals into their own table: one row per model, one column group
    per operating point. Every region is already pooled, so this is the whole result set on
    a single line per model.

    Args:
        agg (pd.DataFrame): Aggregate built over integrate()'s pooled rows.
        model_names (Optional[Sequence[str]]): Models to show, in row order.
        decimals (int): Decimal places. Defaults to 2.
        latex (bool): Emit LaTeX-styled cells. Defaults to False.
        operating_points (Optional[Sequence[str]]): Column order. Defaults to the order found.

    Returns:
        pd.DataFrame: Table indexed by model, with a (operating point, metric) column index.
    """
    subset = agg[(agg["et_bin"] == INTEGRATED) & (agg["eta_bin"] == INTEGRATED)]
    if subset.empty:
        return pd.DataFrame()

    models = resolve_models(subset, model_names)
    if not models:
        return pd.DataFrame()

    points = list(operating_points or dict.fromkeys(subset["operating_point"]))
    columns = _metric_columns(points, latex)

    rows = []
    for model in models:
        row: List[str] = []
        for point in points:
            match = subset[(subset["model"] == model) & (subset["operating_point"] == point)]
            row.extend(_metric_cells(None if match.empty else match.iloc[0], decimals, latex))
        rows.append(row)

    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples([(model,) for model in models], names=["Model"]),
        columns=columns
    )


def _latex_from_wide(
    wide: pd.DataFrame,
    caption: str,
    label: str,
    corner: str = "",
    highlight_pd: bool = True
) -> str:
    """
    Emits a wide table as a standalone LaTeX fragment, ready for \\input{}.

    Written by hand rather than via DataFrame.to_latex because the target layout needs grouped
    \\multicolumn headers and per-cell colouring, neither of which pandas emits. Shared by the
    per-region and the integrated tables, which differ only in what their rows and column
    groups mean.

    Args:
        wide (pd.DataFrame): Table with a 1- or 2-level row MultiIndex and a (group, metric)
            column MultiIndex, cells already formatted for LaTeX.
        caption (str): Table caption.
        label (str): LaTeX label.
        corner (str): Text spanning the index columns in the first header row.
        highlight_pd (bool): Shade the PD column of each group. Defaults to True.

    Returns:
        str: The LaTeX source.
    """
    n_metrics = len(TABLE_METRICS)
    group_labels = list(dict.fromkeys(level for level, _ in wide.columns))
    n_index = wide.index.nlevels

    lines = [
        "% Generated by ai/evaluation/tabelao.py - do not edit by hand.",
        "% Requires: \\usepackage{booktabs}, \\usepackage[table]{xcolor}, \\usepackage{graphicx}",
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{'l' * n_index}{('c' * n_metrics) * len(group_labels)}}}",
        "\\toprule",
    ]

    header = [f"\\multicolumn{{{n_index}}}{{c}}{{{corner}}}"] if corner else [""] * n_index
    header += [f"\\multicolumn{{{n_metrics}}}{{c}}{{{group}}}" for group in group_labels]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("".join(
        f"\\cmidrule(lr){{{n_index + 1 + i * n_metrics}-{n_index + (i + 1) * n_metrics}}}"
        for i in range(len(group_labels))
    ))

    lines.append(" & ".join(list(wide.index.names) + [metric for _, metric in wide.columns]) + " \\\\")
    lines.append("\\midrule")

    previous_block = None
    for position, (keys, values) in enumerate(zip(wide.index, wide.to_numpy())):
        keys = list(keys)
        if n_index > 1:
            # The block label heads its group of rows and is blank below it, which reads as a
            # merged cell without pulling in the multirow package.
            is_new_block = keys[0] != previous_block
            if is_new_block and position > 0:
                lines.append("\\midrule")
            previous_block = keys[0]
            if not is_new_block:
                keys[0] = ""

        cells = list(keys)
        for column, value in enumerate(values):
            if highlight_pd and column % n_metrics == 0 and value != "--":
                value = f"\\cellcolor{{green!25}}{value}"
            cells.append(value)
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    return "\n".join(lines)


def _figure_from_wide(
    wide: pd.DataFrame,
    title: str,
    filepath: str,
    highlight_pd: bool = True
) -> str:
    """
    Renders a wide table as a picture (PDF/PNG, chosen by the file extension), so the numbers
    can be eyeballed without a LaTeX toolchain. Shared by both table kinds.

    Args:
        wide (pd.DataFrame): Table with a 1- or 2-level row MultiIndex and a (group, metric)
            column MultiIndex, cells already formatted as plain text.
        title (str): Figure title.
        filepath (str): Output path; the extension selects the format.
        highlight_pd (bool): Shade the PD column of each group. Defaults to True.

    Returns:
        str: The written path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_metrics = len(TABLE_METRICS)
    n_index = wide.index.nlevels
    group_labels = list(dict.fromkeys(level for level, _ in wide.columns))

    # Two header rows (group, then metric) are emulated as ordinary table rows, because
    # matplotlib tables cannot merge cells: the group label goes in the middle cell of its
    # block with clipping switched off, so it spills over the neighbouring (empty) cells of
    # the same block. Blocks are told apart by alternating background shades rather than by
    # hiding cell edges - a cell with only some of its edges visible is drawn from an open
    # path, and filling that path leaves a diagonal wedge across the cell.
    group_row = [""] * n_index
    for label in group_labels:
        block = [""] * n_metrics
        block[n_metrics // 2] = label
        group_row.extend(block)
    metric_row = list(wide.index.names) + [metric for _, metric in wide.columns]

    body, block_starts = [], []
    previous_block = None
    for keys, values in zip(wide.index, wide.to_numpy()):
        keys = list(keys)
        is_new_block = n_index == 1 or keys[0] != previous_block
        block_starts.append(is_new_block)
        if n_index > 1:
            previous_block = keys[0]
            if not is_new_block:
                keys[0] = ""
        body.append(keys + list(values))

    cell_text = [group_row, metric_row] + body
    n_cols = len(metric_row)

    fig, ax = plt.subplots(
        figsize=(max(6.0, 1.25 * n_cols), max(1.4, 0.40 * len(cell_text) + 0.35))
    )
    ax.axis("off")

    table = ax.table(cellText=cell_text, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    # Size every column to its widest entry, so labels are never truncated.
    table.auto_set_column_width(col=list(range(n_cols)))

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("0.7")
        if row == 0:
            block = 0 if col < n_index else (col - n_index) // n_metrics
            shade = "0.90" if col < n_index else ("0.86" if block % 2 == 0 else "0.78")
            cell.set_facecolor(shade)
            cell.set_edgecolor(shade)
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_clip_on(False)
        elif row == 1:
            cell.set_facecolor("0.92")
            cell.get_text().set_fontweight("bold")
        elif col < n_index:
            cell.set_facecolor("0.96")
            cell.get_text().set_fontweight("bold")
        elif highlight_pd and (col - n_index) % n_metrics == 0:
            cell.set_facecolor("#d6f5d6")

        # Thicker rule where a new row block starts, mirroring the LaTeX \\midrule.
        if row > 2 and block_starts[row - 2]:
            cell.set_linewidth(1.6)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    fig.savefig(filepath, bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info(f"🖼️ Saved table figure to: {filepath}")
    return filepath


def _fold_text(subset: pd.DataFrame) -> str:
    """
    Describes the fold count for a caption: the number when it is uniform, a generic phrase
    when models or regions disagree.

    Args:
        subset (pd.DataFrame): Aggregate rows being captioned.

    Returns:
        str: Caption fragment.
    """
    counts = sorted({int(value) for value in subset["n_folds"].unique()})
    return f"{counts[0]} folds" if len(counts) == 1 else "validação cruzada"


def to_latex(
    agg: pd.DataFrame,
    operating_point: str,
    model_names: Optional[Sequence[str]] = None,
    caption: Optional[str] = None,
    label: Optional[str] = None,
    decimals: int = 2,
    highlight_pd: bool = True
) -> str:
    """
    Renders the per-region table of one operating point as a LaTeX fragment.

    Args:
        agg (pd.DataFrame): Aggregate as produced by aggregate().
        operating_point (str): Working point to render (e.g. 'tight').
        model_names (Optional[Sequence[str]]): Models to show, in row order.
        caption (Optional[str]): Table caption. Generated when None.
        label (Optional[str]): LaTeX label. Defaults to 'tab:tabelao_<operating_point>'.
        decimals (int): Decimal places in the cells. Defaults to 2.
        highlight_pd (bool): Shade the PD column, which is the quantity every network was
            tuned to reproduce. Defaults to True.

    Returns:
        str: The LaTeX source. Empty string when there is nothing to render.
    """
    wide = build_wide(agg, operating_point, model_names=model_names, decimals=decimals, latex=True)
    if wide.empty:
        logger.warning(f"⚠️ No rows for operating point '{operating_point}'; skipping LaTeX render.")
        return ""

    subset = agg[agg["operating_point"] == operating_point]
    caption = caption or (
        f"Valores de eficiência ($P_D$, $SP$, $F_A$) obtidos a partir da validação cruzada "
        f"({_fold_text(subset)}), em cada região do espaço de fase, para o ponto de operação "
        f"\\textit{{{operating_point}}}."
    )
    return _latex_from_wide(
        wide, caption, label or f"tab:tabelao_{operating_point}",
        corner="kinematic region", highlight_pd=highlight_pd
    )


def integrated_to_latex(
    agg: pd.DataFrame,
    model_names: Optional[Sequence[str]] = None,
    caption: Optional[str] = None,
    label: str = "tab:tabelao_integrated",
    decimals: int = 2,
    highlight_pd: bool = True,
    operating_points: Optional[Sequence[str]] = None
) -> str:
    """
    Renders the integrated (phase-space total) table as a LaTeX fragment.

    Args:
        agg (pd.DataFrame): Aggregate built over integrate()'s pooled rows.
        model_names (Optional[Sequence[str]]): Models to show, in row order.
        caption (Optional[str]): Table caption. Generated when None.
        label (str): LaTeX label. Defaults to 'tab:tabelao_integrated'.
        decimals (int): Decimal places. Defaults to 2.
        highlight_pd (bool): Shade the PD column of each operating point. Defaults to True.
        operating_points (Optional[Sequence[str]]): Column order.

    Returns:
        str: The LaTeX source. Empty string when there is nothing to render.
    """
    wide = build_integrated_wide(
        agg, model_names=model_names, decimals=decimals, latex=True,
        operating_points=operating_points
    )
    if wide.empty:
        logger.warning("⚠️ No integrated rows to render.")
        return ""

    subset = agg[(agg["et_bin"] == INTEGRATED) & (agg["eta_bin"] == INTEGRATED)]
    caption = caption or (
        f"Valores de eficiência ($P_D$, $SP$, $F_A$) integrados em todo o espaço de fase, "
        f"obtidos a partir da validação cruzada ({_fold_text(subset)}), para cada ponto de "
        f"operação. Cada região contribui em proporção à sua população."
    )
    return _latex_from_wide(wide, caption, label, corner="", highlight_pd=highlight_pd)


def render_figure(
    agg: pd.DataFrame,
    operating_point: str,
    filepath: str,
    model_names: Optional[Sequence[str]] = None,
    decimals: int = 2,
    highlight_pd: bool = True
) -> Optional[str]:
    """
    Renders the per-region table of one operating point as a picture.

    Args:
        agg (pd.DataFrame): Aggregate as produced by aggregate().
        operating_point (str): Working point to render.
        filepath (str): Output path; the extension selects the format.
        model_names (Optional[Sequence[str]]): Models to show, in row order.
        decimals (int): Decimal places. Defaults to 2.
        highlight_pd (bool): Shade the PD columns. Defaults to True.

    Returns:
        Optional[str]: The written path, or None when there was nothing to render.
    """
    wide = build_wide(
        agg, operating_point, model_names=model_names, decimals=decimals, compact_labels=True
    )
    if wide.empty:
        logger.warning(f"⚠️ No rows for operating point '{operating_point}'; skipping figure render.")
        return None
    return _figure_from_wide(
        wide, f"Cross Validation — operating point: {operating_point}", filepath, highlight_pd
    )


def render_integrated_figure(
    agg: pd.DataFrame,
    filepath: str,
    model_names: Optional[Sequence[str]] = None,
    decimals: int = 2,
    highlight_pd: bool = True,
    operating_points: Optional[Sequence[str]] = None
) -> Optional[str]:
    """
    Renders the integrated table as a picture.

    Args:
        agg (pd.DataFrame): Aggregate built over integrate()'s pooled rows.
        filepath (str): Output path; the extension selects the format.
        model_names (Optional[Sequence[str]]): Models to show, in row order.
        decimals (int): Decimal places. Defaults to 2.
        highlight_pd (bool): Shade the PD columns. Defaults to True.
        operating_points (Optional[Sequence[str]]): Column order.

    Returns:
        Optional[str]: The written path, or None when there was nothing to render.
    """
    wide = build_integrated_wide(
        agg, model_names=model_names, decimals=decimals, operating_points=operating_points
    )
    if wide.empty:
        logger.warning("⚠️ No integrated rows to render.")
        return None
    return _figure_from_wide(
        wide, "Cross Validation — integrated over the phase space", filepath, highlight_pd
    )


def check_comparable(long_df: pd.DataFrame) -> List[str]:
    """
    Checks that, within each kinematic region, every model was scored on the same holdout.

    The comparison is only meaningful if the models saw identical test rows, which happens
    when their configs agree on data_path, max_files, n_splits and seed. The scored rows'
    signal/background counts are a cheap proxy for that: if they differ between two models in
    the same region, the configs drifted apart and the rows are not comparable.

    Args:
        long_df (pd.DataFrame): Long table as produced by collect().

    Returns:
        List[str]: One message per region where the models disagree (empty when all agree).
    """
    problems = []
    keys = ["et_bin", "eta_bin"]
    for region, group in long_df.groupby(keys, dropna=False):
        counts = group.groupby("model")[["n_signal", "n_background"]].first()
        if len(counts.drop_duplicates()) > 1:
            et_bin, eta_bin = region
            region_name = "full phase space" if pd.isna(et_bin) else f"et{int(et_bin)}_eta{int(eta_bin)}"
            detail = ", ".join(
                f"{model}: {int(row.n_signal)} sig / {int(row.n_background)} bkg"
                for model, row in counts.iterrows()
            )
            problems.append(f"{region_name} -> {detail}")
    return problems


def default_output_dir(results_root: str, model_names: Optional[Sequence[str]]) -> str:
    """
    Picks where a report's artefacts go: inside the model's own directory for a single-model
    table, and in a shared 'comparison' directory when several models share one table.

    Args:
        results_root (str): Root results directory.
        model_names (Optional[Sequence[str]]): Models being reported, or None for "all".

    Returns:
        str: The output directory path.
    """
    if model_names and len(model_names) == 1:
        return os.path.join(results_root, model_names[0], "tabelao")
    return os.path.join(results_root, "comparison", "tabelao")


def build_report(
    results_root: str = "results",
    model_names: Optional[Sequence[str]] = None,
    output_dir: Optional[str] = None,
    operating_points: Optional[Sequence[str]] = None,
    decimals: int = 2,
    integrated: bool = True,
    formats: Sequence[str] = ("tex", "pdf"),
    highlight_pd: bool = True
) -> Dict[str, List[str]]:
    """
    End-to-end table build: collect -> aggregate -> render every operating point.

    Pass several names in `model_names` to get one comparison table per operating point, with
    a row per (|eta| region, model). Since every network is tuned to the same target PD, the
    comparison reads straight down the SP and FA columns.

    Args:
        results_root (str): Root results directory. Defaults to 'results'.
        model_names (Optional[Sequence[str]]): Models to include, in row order. Scans every
            evaluated model when None.
        output_dir (str): Where the table artefacts go. Defaults to the model's own
            'tabelao' directory, or '<results_root>/comparison/tabelao' for several models.
        operating_points (Optional[Sequence[str]]): Which working points to render.
            Defaults to every one present in the collected data.
        decimals (int): Decimal places in the cells. Defaults to 2.
        integrated (bool): Also write the integrated table - every region pooled into the
            phase-space total - as its own set of files. Defaults to True.
        formats (Sequence[str]): Render formats - 'tex' plus any matplotlib-supported
            image extension ('pdf', 'png'). Defaults to ('tex', 'pdf').
        highlight_pd (bool): Shade the PD column. Defaults to True.

    Returns:
        Dict[str, List[str]]: Written paths, keyed by artefact kind ('long', 'tex', 'figure',
        'integrated').
    """
    written: Dict[str, List[str]] = {"long": [], "tex": [], "figure": [], "integrated": []}

    # Announce what was found before touching the numbers: which regions exist, which are
    # still unevaluated, and therefore why the table looks the way it does.
    log_inventory(discover_regions(results_root, model_names), results_root)

    long_df = collect(results_root, model_names)
    if long_df.empty:
        return written

    if long_df["model"].nunique() > 1:
        for problem in check_comparable(long_df):
            logger.warning(
                f"⚠️ Models were scored on different holdouts in {problem}. "
                "Their rows are not directly comparable — align data_path, max_files, "
                "n_splits and seed across the configs and re-run `evaluate`."
            )

    output_dir = output_dir or default_output_dir(results_root, model_names)
    os.makedirs(output_dir, exist_ok=True)

    long_path = os.path.join(output_dir, "tabelao_long.csv")
    long_df.to_csv(long_path, index=False)
    written["long"].append(long_path)
    logger.info(f"📝 Saved canonical long table to: {long_path}")

    # The canonical CSV above holds the measured per-region rows only; the margins are a
    # rendering-time derivation, so they never masquerade as data in the long table.
    agg = aggregate(long_df)
    points = operating_points or list(dict.fromkeys(long_df["operating_point"]))

    for point in points:
        if "tex" in formats:
            source = to_latex(
                agg, point, model_names=model_names,
                decimals=decimals, highlight_pd=highlight_pd
            )
            if source:
                tex_path = os.path.join(output_dir, f"tabelao_{point}.tex")
                with open(tex_path, "w") as handle:
                    handle.write(source)
                written["tex"].append(tex_path)
                logger.info(f"\U0001f4c4 Saved LaTeX table to: {tex_path}")

        for extension in (fmt for fmt in formats if fmt != "tex"):
            rendered = render_figure(
                agg, point, os.path.join(output_dir, f"tabelao_{point}.{extension}"),
                model_names=model_names, decimals=decimals, highlight_pd=highlight_pd
            )
            if rendered:
                written["figure"].append(rendered)

    if integrated:
        # The phase-space total is saved as its own artefact rather than folded into the grid
        # above: it answers a different question and is usually quoted on its own.
        integrated_long = integrate(long_df)
        if not integrated_long.empty:
            csv_path = os.path.join(output_dir, "tabelao_integrated_long.csv")
            integrated_long.to_csv(csv_path, index=False)
            written["integrated"].append(csv_path)
            logger.info(f"\U0001f4dd Saved integrated long table to: {csv_path}")

            agg_integrated = aggregate(integrated_long)

            if "tex" in formats:
                source = integrated_to_latex(
                    agg_integrated, model_names=model_names, decimals=decimals,
                    highlight_pd=highlight_pd, operating_points=points
                )
                if source:
                    tex_path = os.path.join(output_dir, "tabelao_integrated.tex")
                    with open(tex_path, "w") as handle:
                        handle.write(source)
                    written["integrated"].append(tex_path)
                    logger.info(f"\U0001f4c4 Saved integrated LaTeX table to: {tex_path}")

            for extension in (fmt for fmt in formats if fmt != "tex"):
                rendered = render_integrated_figure(
                    agg_integrated,
                    os.path.join(output_dir, f"tabelao_integrated.{extension}"),
                    model_names=model_names, decimals=decimals,
                    highlight_pd=highlight_pd, operating_points=points
                )
                if rendered:
                    written["integrated"].append(rendered)

    return written
