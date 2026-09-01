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

#: Metrics shown in each Et column group, in order, with their table headers.
TABLE_METRICS: List[Tuple[str, str, str]] = [
    ("pd", "PD[%]", r"$P_D$[\%]"),
    ("sp", "SP[%]", r"$SP$[\%]"),
    ("fa", "FA[%]", r"$F_A$[\%]"),
]


def _et_label(et_bin: Optional[int], compact: bool = False) -> str:
    """
    Column-group label for an Et bin. The compact form is used by the figure render, where
    matplotlib sizes columns to their widest cell and a full '15 < Et[GeV] < 20' would either
    overflow its cell or blow the column up.

    Args:
        et_bin (Optional[int]): Et bin index, or None for an ungridded run.
        compact (bool): Use the short form ('30-40 GeV'). Defaults to False.

    Returns:
        str: The label.
    """
    if et_bin is None:
        return "all Et" if compact else "all Et"
    if not compact:
        return et_range_str(et_bin)
    lo = ET_BIN_EDGES_GEV[et_bin]
    if et_bin + 1 >= N_ET_BINS:
        return f"> {lo:g} GeV"
    return f"{lo:g}-{ET_BIN_EDGES_GEV[et_bin + 1]:g} GeV"


def _eta_label(eta_bin: Optional[int], compact: bool = False) -> str:
    """
    Row label for an |eta| bin.

    Args:
        eta_bin (Optional[int]): |eta| bin index, or None for an ungridded run.
        compact (bool): Use the short form ('0.00-0.80'). Defaults to False.

    Returns:
        str: The label.
    """
    if eta_bin is None:
        return "all |eta|"
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


def format_cell(mean: float, std: float, decimals: int = 2) -> str:
    """
    Formats one mean+/-std cell, e.g. '98.16+/-0.04'.

    Args:
        mean (float): Mean value.
        std (float): Standard deviation.
        decimals (int): Decimal places. Defaults to 2.

    Returns:
        str: The formatted cell, or '--' when the value is missing.
    """
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "--"
    return f"{mean:.{decimals}f}±{std:.{decimals}f}"


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


def build_wide(
    agg: pd.DataFrame,
    operating_point: str,
    model_names: Optional[Sequence[str]] = None,
    decimals: int = 2,
    compact_labels: bool = False
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

    Returns:
        pd.DataFrame: Table with a ('Det. Region', 'Model') MultiIndex on the rows and a
        (Et range, metric) MultiIndex on the columns. Empty frame when nothing matches.
    """
    subset = agg[agg["operating_point"] == operating_point]
    if subset.empty:
        return pd.DataFrame()

    models = resolve_models(subset, model_names)
    if not models:
        return pd.DataFrame()

    eta_bins, et_bins, ungridded = _resolve_regions(subset)

    columns = pd.MultiIndex.from_tuples(
        [
            (_et_label(et, compact_labels), header)
            for et in et_bins
            for _, header, _ in TABLE_METRICS
        ],
        names=["kinematic region", ""],
    )

    rows, index = [], []
    for eta in eta_bins:
        for model in models:
            index.append((_eta_label(eta, compact_labels), model))
            row = []
            for et in et_bins:
                entry = _lookup(subset, ungridded, model, et, eta)
                for metric, _, _ in TABLE_METRICS:
                    if entry is None:
                        row.append("--")
                    else:
                        row.append(format_cell(entry[f"{metric}_mean"], entry[f"{metric}_std"], decimals))
            rows.append(row)

    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=["Det. Region", "Model"]),
        columns=columns
    )


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
    Renders one operating point as a standalone LaTeX table fragment, ready for \\input{}.

    Written by hand rather than via DataFrame.to_latex because the target layout needs grouped
    \\multicolumn headers over the Et regions and per-cell colouring, neither of which pandas emits.

    Args:
        agg (pd.DataFrame): Aggregate as produced by aggregate().
        operating_point (str): Working point to render (e.g. 'tight').
        model_names (Optional[Sequence[str]]): Models to show, in row order. Every evaluated
            model when None.
        caption (Optional[str]): Table caption. A sensible default is generated when None.
        label (Optional[str]): LaTeX label. Defaults to 'tab:tabelao_<operating_point>'.
        decimals (int): Decimal places in the cells. Defaults to 2.
        highlight_pd (bool): Shade the PD column, which is the quantity every network was
            tuned to reproduce. Defaults to True.

    Returns:
        str: The LaTeX source. Empty string when there is nothing to render.
    """
    subset = agg[agg["operating_point"] == operating_point]
    if subset.empty:
        logger.warning(f"⚠️ No rows for operating point '{operating_point}'; skipping LaTeX render.")
        return ""

    models = resolve_models(subset, model_names)
    if not models:
        logger.warning(f"⚠️ No models to render for operating point '{operating_point}'.")
        return ""

    eta_bins, et_bins, ungridded = _resolve_regions(subset)

    fold_counts = sorted({int(v) for v in subset["n_folds"].unique()})
    folds_text = f"{fold_counts[0]} folds" if len(fold_counts) == 1 else "validação cruzada"
    caption = caption or (
        f"Valores de eficiência ($P_D$, $SP$, $F_A$) obtidos a partir da validação cruzada "
        f"({folds_text}), em cada região do espaço de fase, para o ponto de operação "
        f"\\textit{{{operating_point}}}."
    )
    label = label or f"tab:tabelao_{operating_point}"

    n_metrics = len(TABLE_METRICS)
    column_spec = "ll" + ("c" * n_metrics) * len(et_bins)

    lines = [
        "% Generated by ai/evaluation/tabelao.py - do not edit by hand.",
        "% Requires: \\usepackage{booktabs}, \\usepackage[table]{xcolor}, \\usepackage{graphicx}",
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
    ]

    # Header row 1: Et group spans.
    header_groups = ["\\multicolumn{2}{c}{kinematic region}"]
    for et in et_bins:
        title = et_range_str(et, latex=True) if et is not None else "all $E_T$"
        header_groups.append(f"\\multicolumn{{{n_metrics}}}{{c}}{{{title}}}")
    lines.append(" & ".join(header_groups) + " \\\\")

    # cmidrule under each Et group (columns 3.. onwards, in blocks of n_metrics).
    rules = []
    for i in range(len(et_bins)):
        first = 3 + i * n_metrics
        rules.append(f"\\cmidrule(lr){{{first}-{first + n_metrics - 1}}}")
    lines.append("".join(rules))

    # Header row 2: metric names.
    header_metrics = ["Det. Region", "Model"]
    for _ in et_bins:
        header_metrics.extend(tex for _, _, tex in TABLE_METRICS)
    lines.append(" & ".join(header_metrics) + " \\\\")
    lines.append("\\midrule")

    for block, eta in enumerate(eta_bins):
        if block > 0 and len(models) > 1:
            # With several models per region the blocks would otherwise run together.
            lines.append("\\midrule")
        region = eta_range_str(eta, latex=True) if eta is not None else "all $|\\eta|$"
        for position, model in enumerate(models):
            # The region label heads its block and is left blank on the model rows below it,
            # which reads as a merged cell without pulling in the multirow package.
            cells = [region if position == 0 else "", model]
            for et in et_bins:
                entry = _lookup(subset, ungridded, model, et, eta)
                for metric, _, _ in TABLE_METRICS:
                    if entry is None:
                        cells.append("--")
                        continue
                    value = f"{entry[f'{metric}_mean']:.{decimals}f} $\\pm$ {entry[f'{metric}_std']:.{decimals}f}"
                    if highlight_pd and metric == "pd":
                        value = f"\\cellcolor{{green!25}}{value}"
                    cells.append(value)
            lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    return "\n".join(lines)


def render_figure(
    agg: pd.DataFrame,
    operating_point: str,
    filepath: str,
    model_names: Optional[Sequence[str]] = None,
    decimals: int = 2,
    highlight_pd: bool = True
) -> Optional[str]:
    """
    Renders one operating point as a picture of the table (PDF/PNG, chosen by the file
    extension), so the numbers can be eyeballed without a LaTeX toolchain.

    Args:
        agg (pd.DataFrame): Aggregate as produced by aggregate().
        operating_point (str): Working point to render.
        filepath (str): Output path; the extension selects the format.
        model_names (Optional[Sequence[str]]): Models to show, in row order. Every evaluated
            model when None.
        decimals (int): Decimal places in the cells. Defaults to 2.
        highlight_pd (bool): Shade the PD columns. Defaults to True.

    Returns:
        Optional[str]: The written path, or None when there was nothing to render.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wide = build_wide(
        agg, operating_point, model_names=model_names, decimals=decimals, compact_labels=True
    )
    if wide.empty:
        logger.warning(f"⚠️ No rows for operating point '{operating_point}'; skipping figure render.")
        return None

    n_metrics = len(TABLE_METRICS)
    et_labels = list(dict.fromkeys(level for level, _ in wide.columns))
    n_models = len({model for _, model in wide.index})

    # Two header rows (Et group, then metric) are emulated as ordinary table rows, because
    # matplotlib tables cannot merge cells: the Et label goes in the middle cell of its block
    # with clipping switched off, so it spills over the neighbouring (empty) cells of the same
    # block. Blocks are told apart by alternating background shades rather than by hiding cell
    # edges - a cell with only some of its edges visible is drawn from an open path, and
    # filling that path leaves a diagonal wedge across the cell.
    group_row = ["", ""]
    for lab in et_labels:
        block = [""] * n_metrics
        block[n_metrics // 2] = lab
        group_row.extend(block)
    metric_row = ["Det. Region", "Model"] + [
        header for _ in et_labels for _, header, _ in TABLE_METRICS
    ]

    body, region_starts = [], []
    previous_region = None
    for (region, model), row in zip(wide.index, wide.to_numpy()):
        is_new_region = region != previous_region
        region_starts.append(is_new_region)
        # As in the LaTeX render, the region label heads its block and is blank below it.
        body.append([region if is_new_region else "", model] + list(row))
        previous_region = region

    cell_text = [group_row, metric_row] + body

    n_cols = len(metric_row)
    fig_width = max(6.0, 1.25 * n_cols)
    fig_height = max(1.4, 0.40 * len(cell_text) + 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(cellText=cell_text, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    # Size every column to its widest entry, so region and model labels are never truncated.
    table.auto_set_column_width(col=list(range(n_cols)))

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("0.7")
        if row == 0:
            block = 0 if col < 2 else (col - 2) // n_metrics
            shade = "0.90" if col < 2 else ("0.86" if block % 2 == 0 else "0.78")
            cell.set_facecolor(shade)
            cell.set_edgecolor(shade)
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_clip_on(False)
        elif row == 1:
            cell.set_facecolor("0.92")
            cell.get_text().set_fontweight("bold")
        elif col < 2:
            cell.set_facecolor("0.96")
            cell.get_text().set_fontweight("bold")
        elif highlight_pd and (col - 2) % n_metrics == 0:
            cell.set_facecolor("#d6f5d6")

        # Thicker top rule where a new |eta| region starts, mirroring the LaTeX \midrule.
        if row >= 2 and n_models > 1 and region_starts[row - 2] and row > 2:
            cell.set_linewidth(1.6)

    title = f"Cross Validation — operating point: {operating_point}"
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    fig.savefig(filepath, bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info(f"🖼️ Saved table figure to: {filepath}")
    return filepath


def check_comparable(long_df: pd.DataFrame) -> List[str]:
    """
    Checks that, within each kinematic region, every model was scored on the same holdout.

    The comparison is only meaningful if the models saw identical test rows, which happens
    when their configs agree on data_path, max_files, test_size and seed. The holdout's
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
        formats (Sequence[str]): Render formats - 'tex' plus any matplotlib-supported
            image extension ('pdf', 'png'). Defaults to ('tex', 'pdf').
        highlight_pd (bool): Shade the PD column. Defaults to True.

    Returns:
        Dict[str, List[str]]: Written paths, keyed by artefact kind ('long', 'tex', 'figure').
    """
    written: Dict[str, List[str]] = {"long": [], "tex": [], "figure": []}

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
                "test_size and seed across the configs and re-run `evaluate`."
            )

    output_dir = output_dir or default_output_dir(results_root, model_names)
    os.makedirs(output_dir, exist_ok=True)

    long_path = os.path.join(output_dir, "tabelao_long.csv")
    long_df.to_csv(long_path, index=False)
    written["long"].append(long_path)
    logger.info(f"📝 Saved canonical long table to: {long_path}")

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
                logger.info(f"📄 Saved LaTeX table to: {tex_path}")

        for extension in (fmt for fmt in formats if fmt != "tex"):
            figure_path = os.path.join(output_dir, f"tabelao_{point}.{extension}")
            rendered = render_figure(
                agg, point, figure_path, model_names=model_names,
                decimals=decimals, highlight_pd=highlight_pd
            )
            if rendered:
                written["figure"].append(rendered)

    return written
