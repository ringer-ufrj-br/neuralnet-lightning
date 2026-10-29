"""
Cross-validation table ("tabelão" / pd_table).

Concatenates every region's `metrics/folds_long.csv` into a long table, aggregates it over
folds and renders it as LaTeX and HTML: rows are |eta| regions (one per experiment), column
groups are Et regions with PD / SP / FA.
"""

import glob
import logging
import os
import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from neuralnet_lightning import binning
from neuralnet_lightning.binary.metrics import sp_index

logger = logging.getLogger(__name__)

#: Columns of the long table (`folds_long.csv`).
LONG_COLUMNS: List[str] = [
    "name", "et_bin", "eta_bin", "fold", "operating_point",
    "target_pd", "threshold", "pd", "fa", "sp", "auc_roc", "auc_pr",
    "n_signal", "n_background",
]

#: Bin index the integrated (phase-space total) rows carry in their long CSV.
INTEGRATED: int = -1

#: Metrics shown in each column group, in order, with their plain and LaTeX headers.
TABLE_METRICS = [
    ("pd", "PD[%]", r"$P_D$[\%]"),
    ("sp", "SP[%]", r"$SP$[\%]"),
    ("fa", "FA[%]", r"$F_A$[\%]"),
]


def _et_label(et_bin: float, latex: bool) -> str:
    """Column-group label for an Et bin, or for a whole-phase-space run (NaN)."""
    if pd.isna(et_bin):
        return "all $E_T$" if latex else "all Et"
    return binning.et_range_str(int(et_bin), latex)


def _eta_label(eta_bin: float, latex: bool) -> str:
    """Row label for an |eta| bin, or for a whole-phase-space run (NaN)."""
    if pd.isna(eta_bin):
        return r"all $|\eta|$" if latex else "all |eta|"
    return binning.eta_range_str(int(eta_bin), latex)


# ---------------------------------------------------------------- inventory

def discover_regions(results_root: str, names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    One row per trained region under `<results_root>/<name>/<region>/` (name, region, path,
    folds_trained, folds_evaluated, evaluated), evaluated or not.
    """
    rows = []
    for target in names or ["*"]:
        for region_dir in sorted(glob.glob(os.path.join(results_root, target, "*"))):
            sidecars = glob.glob(os.path.join(region_dir, "checkpoints", "fold_*_init_*.json"))
            if not sidecars:
                continue
            long_path = os.path.join(region_dir, "metrics", "folds_long.csv")
            folds_evaluated = int(pd.read_csv(long_path)["fold"].nunique()) if os.path.exists(long_path) else 0
            rows.append({
                "name": os.path.basename(os.path.dirname(region_dir)),
                "region": os.path.basename(region_dir),
                "path": region_dir,
                "folds_trained": len({re.match(r"fold_(\d+)", os.path.basename(p)).group(1) for p in sidecars}),
                "folds_evaluated": folds_evaluated,
                "evaluated": folds_evaluated > 0,
            })
    return pd.DataFrame(rows, columns=["name", "region", "path", "folds_trained", "folds_evaluated", "evaluated"])


def log_inventory(inventory: pd.DataFrame, results_root: str) -> None:
    """Logs each region found and the command that evaluates the missing ones."""
    if inventory.empty:
        logger.warning(f"⚠️ No trained region found under '{results_root}'. Run `train` (then `evaluate`) first.")
        return

    logger.info(f"🔎 Found {len(inventory)} trained region(s) under '{results_root}':")
    for entry in inventory.itertuples():
        status = f"{entry.folds_evaluated}/{entry.folds_trained} folds evaluated" if entry.evaluated else "NOT EVALUATED"
        logger.info(f"   {entry.name:<20} {entry.region:<10} {status:<24} {entry.path}")

    for entry in inventory[~inventory["evaluated"]].itertuples():
        match = re.fullmatch(r"et(\d+)_eta(\d+)", entry.region)
        region_args = f" --et-bin {match.group(1)} --eta-bin {match.group(2)}" if match else ""
        logger.warning(f"⚠️ {entry.name} / {entry.region} was trained but never evaluated, so it is missing "
                       f"from the table. Fix with: python -m neuralnet_lightning evaluate --config <config>{region_args}")

    partial = inventory[inventory["evaluated"] & (inventory["folds_evaluated"] < inventory["folds_trained"])]
    for entry in partial.itertuples():
        logger.warning(f"⚠️ {entry.name} / {entry.region}: {entry.folds_trained} folds trained but only "
                       f"{entry.folds_evaluated} evaluated. Re-run `evaluate` so the spread covers every fold.")


def collect(results_root: str, names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Concatenates every region's `metrics/folds_long.csv` (empty frame if none)."""
    paths = sorted({path for target in names or ["*"]
                    for path in glob.glob(os.path.join(results_root, target, "*", "metrics", "folds_long.csv"))})
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = [col for col in LONG_COLUMNS if col not in frame.columns]
        if missing:
            logger.warning(f"⚠️ Skipping '{path}': missing columns {missing}.")
            continue
        frames.append(frame[LONG_COLUMNS])

    if not frames:
        logger.warning(f"⚠️ No 'folds_long.csv' found under '{results_root}'. Run `evaluate` first.")
        return pd.DataFrame(columns=LONG_COLUMNS)

    long_df = pd.concat(frames, ignore_index=True)
    if long_df["et_bin"].notna().any() and long_df["et_bin"].isna().any():
        logger.warning("⚠️ Both binned and whole-phase-space runs were found. The table shows the binned "
                       "ones; the others stay in the long CSV, since they belong to no Et/|eta| cell.")
    logger.info(f"📚 Collected {len(long_df)} rows from {len(frames)} region(s): "
                f"{sorted(long_df['name'].unique())}, {long_df['fold'].nunique()} fold(s), "
                f"{long_df['operating_point'].nunique()} operating point(s).")
    return long_df


def check_comparable(long_df: pd.DataFrame) -> List[str]:
    """One message per region where experiments differ in signal/background counts."""
    problems = []
    for (et_bin, eta_bin), group in long_df.groupby(["et_bin", "eta_bin"], dropna=False):
        counts = group.groupby("name")[["n_signal", "n_background"]].first()
        if len(counts.drop_duplicates()) > 1:
            region = "full phase space" if pd.isna(et_bin) else f"et{int(et_bin)}_eta{int(eta_bin)}"
            detail = ", ".join(f"{name}: {int(row.n_signal)} sig / {int(row.n_background)} bkg"
                               for name, row in counts.iterrows())
            problems.append(f"{region} -> {detail}")
    return problems


# --------------------------------------------------------------- aggregation

def integrate(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pools every region into one row per (name, fold, operating point), marked
    et_bin = eta_bin = INTEGRATED. PD = sum(pd_r * n_signal_r) / sum(n_signal_r), likewise FA.
    """
    binned = long_df[long_df["et_bin"].notna() & long_df["eta_bin"].notna()]
    source = binned if not binned.empty else long_df
    if source.empty:
        return pd.DataFrame(columns=LONG_COLUMNS)

    pooled = source.assign(_tp=source["pd"] * source["n_signal"], _fp=source["fa"] * source["n_background"]) \
        .groupby(["name", "fold", "operating_point"], dropna=False).agg(
            n_signal=("n_signal", "sum"), n_background=("n_background", "sum"),
            target_pd=("target_pd", "first"), _tp=("_tp", "sum"), _fp=("_fp", "sum"),
        ).reset_index()

    pooled["pd"] = np.where(pooled["n_signal"] > 0, pooled["_tp"] / pooled["n_signal"], 0.0)
    pooled["fa"] = np.where(pooled["n_background"] > 0, pooled["_fp"] / pooled["n_background"], 0.0)
    pooled["sp"] = sp_index(pooled["pd"], pooled["fa"])
    # Not defined once regions are pooled.
    pooled[["threshold", "auc_roc", "auc_pr"]] = np.nan
    pooled["et_bin"] = INTEGRATED
    pooled["eta_bin"] = INTEGRATED
    return pooled[LONG_COLUMNS]


def aggregate(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean and std over folds of pd, sp and fa (in percent) plus `n_folds`, one row per
    (name, et_bin, eta_bin, operating_point).
    """
    working = long_df.assign(**{column: long_df[column] * 100.0 for column in ("pd", "sp", "fa")})
    agg = working.groupby(["name", "et_bin", "eta_bin", "operating_point"], dropna=False).agg(
        pd_mean=("pd", "mean"), pd_std=("pd", "std"),
        sp_mean=("sp", "mean"), sp_std=("sp", "std"),
        fa_mean=("fa", "mean"), fa_std=("fa", "std"),
        n_folds=("fold", "nunique"),
    ).reset_index()
    # A single fold gives std NaN.
    std_columns = [column for column in agg.columns if column.endswith("_std")]
    agg[std_columns] = agg[std_columns].fillna(0.0)
    return agg


def resolve_names(agg: pd.DataFrame, names: Optional[Sequence[str]] = None) -> List[str]:
    """Experiments shown as rows: `names` in order (dropping those without results), else all, sorted."""
    present = set(agg["name"].unique())
    if not names:
        return sorted(present)
    for name in names:
        if name not in present:
            logger.warning(f"⚠️ '{name}' has no evaluated region; leaving it out of the table.")
    return [name for name in names if name in present]


# ----------------------------------------------------------------- rendering

def format_cell(mean: float, std: float, decimals: int = 2, latex: bool = False) -> str:
    """One mean±std cell, or '--' when the value is missing."""
    if pd.isna(mean):
        return "--"
    return f"{mean:.{decimals}f}{r' $\pm$ ' if latex else '±'}{std:.{decimals}f}"


def _pivot(agg: pd.DataFrame, rows: List[str], group: str, index: pd.Index,
           groups: Sequence[str], decimals: int, latex: bool) -> pd.DataFrame:
    """Pivots aggregate rows into mean±std cells: `index` rows, one (group, metric) column per entry of `groups`."""
    labels = [tex if latex else plain for _, plain, tex in TABLE_METRICS]
    cells = pd.DataFrame(
        {label: [format_cell(mean, std, decimals, latex) for mean, std in zip(agg[f"{key}_mean"], agg[f"{key}_std"])]
         for label, (key, _, _) in zip(labels, TABLE_METRICS)},
        index=pd.MultiIndex.from_frame(agg[rows + [group]]),
    )
    wide = cells.unstack(group).swaplevel(axis=1)
    return wide.reindex(index=index, columns=pd.MultiIndex.from_product([groups, labels])).fillna("--")


def build_wide(agg: pd.DataFrame, operating_point: str, names: Optional[Sequence[str]] = None,
               decimals: int = 2, latex: bool = False) -> pd.DataFrame:
    """
    Table of one operating point: one row per (|eta| region, experiment), one column per
    (Et region, metric). Binned regions take precedence over whole-phase-space runs.
    """
    subset = agg[agg["operating_point"] == operating_point]
    rows = resolve_names(subset, names)
    if not rows:
        return pd.DataFrame()

    gridded = subset[subset["et_bin"].notna() & subset["eta_bin"].notna()]
    subset = (gridded if not gridded.empty else subset).sort_values(["eta_bin", "et_bin"])
    subset = subset.assign(eta=[_eta_label(value, latex) for value in subset["eta_bin"]],
                           et=[_et_label(value, latex) for value in subset["et_bin"]])
    index = pd.MultiIndex.from_product([subset["eta"].unique(), rows], names=["Det. Region", "Experiment"])
    return _pivot(subset, ["eta", "name"], "et", index, subset["et"].unique(), decimals, latex)


def build_integrated_wide(agg: pd.DataFrame, names: Optional[Sequence[str]], points: Sequence[str],
                          decimals: int = 2, latex: bool = False) -> pd.DataFrame:
    """The phase-space totals: one row per experiment, one column group per operating point."""
    rows = resolve_names(agg, names)
    if not rows:
        return pd.DataFrame()
    return _pivot(agg, ["name"], "operating_point", pd.Index(rows, name="Experiment"), points, decimals, latex)


def _to_latex(wide: pd.DataFrame, caption: str, label: str) -> str:
    """A wide table as a LaTeX fragment for \\input{}, with the PD columns shaded."""
    # Escape '_' in experiment names; the region labels are already LaTeX.
    wide = wide.rename(index=lambda value: str(value).replace("_", r"\_"), level="Experiment")
    pd_columns = [column for column in wide.columns if column[1] == TABLE_METRICS[0][2]]
    tabular = wide.style.map(lambda value: "" if value == "--" else "cellcolor:{green!25};", subset=pd_columns).to_latex(
        hrules=True,
        multicol_align="c",
        multirow_align="naive",
        clines="skip-last;data" if wide.index.nlevels > 1 else None,
        column_format="l" * wide.index.nlevels + "c" * len(wide.columns),
    )
    # Replace Styler's \cline separators with \midrule, dropping the one before \bottomrule.
    tabular = re.sub(r"\\cline\{[^}]*\}\n(\\bottomrule)?", lambda m: m.group(1) or "\\midrule\n", tabular)
    return "\n".join([
        "% Generated by neuralnet_lightning/binary/report.py - do not edit by hand.",
        "% Requires: \\usepackage{booktabs}, \\usepackage[table]{xcolor}, \\usepackage{graphicx}",
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        tabular.rstrip(),
        "}",
        "\\end{table}",
        "",
    ])


def _to_html(wide: pd.DataFrame, title: str) -> str:
    """A wide table as HTML, with the PD columns shaded."""
    pd_columns = [column for column in wide.columns if column[1] == TABLE_METRICS[0][1]]
    return wide.style.map(lambda value: "" if value == "--" else "background-color: #d6f5d6;", subset=pd_columns) \
        .set_caption(title) \
        .set_table_styles([{"selector": "th, td", "props": "border: 1px solid #bbb; padding: 2px 8px; text-align: center;"}]) \
        .to_html()


def _write_table(output_dir: str, stem: str, tex: pd.DataFrame, plain: pd.DataFrame,
                 caption: str, label: str, title: str) -> List[str]:
    """Writes one table as `<stem>.tex` and `<stem>.html`."""
    paths = [os.path.join(output_dir, f"{stem}.tex"), os.path.join(output_dir, f"{stem}.html")]
    for path, source in zip(paths, (_to_latex(tex, caption, label), _to_html(plain, title))):
        with open(path, "w") as handle:
            handle.write(source)
        logger.info(f"📄 Saved table to: {path}")
    return paths


def _fold_text(agg: pd.DataFrame) -> str:
    """The fold count for a caption: the number when it is uniform, a generic phrase otherwise."""
    counts = sorted({int(value) for value in agg["n_folds"].unique()})
    return f"{counts[0]} folds" if len(counts) == 1 else "validação cruzada"


def build_report(
    results_root: str,
    names: Optional[Sequence[str]] = None,
    output_dir: Optional[str] = None,
    decimals: int = 2,
    integrated: bool = True
) -> Dict[str, List[str]]:
    """
    Writes the long CSV and a LaTeX + HTML table per operating point, plus the integrated table
    unless `integrated` is False. Returns the written paths by kind.

    Output goes to `output_dir`, else `<results_root>/<name>/pd_table` for a single experiment
    and `<results_root>/comparison/pd_table` for several.
    """
    written: Dict[str, List[str]] = {"long": [], "tables": [], "integrated": []}

    log_inventory(discover_regions(results_root, names), results_root)

    long_df = collect(results_root, names)
    if long_df.empty:
        return written

    if long_df["name"].nunique() > 1:
        for problem in check_comparable(long_df):
            logger.warning(f"⚠️ Experiments were scored on different rows in {problem}. Their rows are not "
                           "directly comparable — align the dataset, max_files, n_splits and seed.")

    if output_dir is None:
        output_dir = os.path.join(results_root, names[0] if names and len(names) == 1 else "comparison", "pd_table")
    os.makedirs(output_dir, exist_ok=True)

    long_path = os.path.join(output_dir, "pd_table_long.csv")
    long_df.to_csv(long_path, index=False)
    written["long"].append(long_path)
    logger.info(f"📝 Saved long table to: {long_path}")

    agg = aggregate(long_df)
    points = list(dict.fromkeys(long_df["operating_point"]))

    for point in points:
        tex = build_wide(agg, point, names, decimals, latex=True)
        if tex.empty:
            logger.warning(f"⚠️ No rows for operating point '{point}'; skipping its table.")
            continue
        caption = (f"Valores de eficiência ($P_D$, $SP$, $F_A$) obtidos a partir da validação cruzada "
                   f"({_fold_text(agg[agg['operating_point'] == point])}), em cada região do espaço de "
                   f"fase, para o ponto de operação \\textit{{{point}}}.")
        written["tables"] += _write_table(output_dir, f"pd_table_{point}", tex, build_wide(agg, point, names, decimals),
                                          caption, f"tab:pd_table_{point}", f"Cross Validation — operating point: {point}")

    if integrated:
        integrated_long = integrate(long_df)
        csv_path = os.path.join(output_dir, "pd_table_integrated_long.csv")
        integrated_long.to_csv(csv_path, index=False)
        written["integrated"].append(csv_path)

        agg_integrated = aggregate(integrated_long)
        tex = build_integrated_wide(agg_integrated, names, points, decimals, latex=True)
        if not tex.empty:
            caption = (f"Valores de eficiência ($P_D$, $SP$, $F_A$) integrados em todo o espaço de fase, "
                       f"obtidos a partir da validação cruzada ({_fold_text(agg_integrated)}), para cada "
                       f"ponto de operação. Cada região contribui em proporção à sua população.")
            written["integrated"] += _write_table(
                output_dir, "pd_table_integrated", tex, build_integrated_wide(agg_integrated, names, points, decimals),
                caption, "tab:pd_table_integrated", "Cross Validation — integrated over the phase space")

    return written
