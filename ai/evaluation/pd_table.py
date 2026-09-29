"""
Cross-validation table ("pd_table") builder.

The canonical artefact is a **long/tidy** table: one purely numeric row per
(model, et_bin, eta_bin, fold, operating_point). Every render - the LaTeX fragment and the
HTML view - is derived from it, so there is exactly one place where numbers are produced and
several places where they are formatted.

Layout of the rendered table mirrors the ATLAS/Ringer convention: rows are |eta| regions,
column groups are Et regions, and each group carries the PD / SP / FA triplet. When more than
one model has been evaluated, each |eta| region gets one row per model, so architectures are
compared side by side under identical working points.
"""

import glob
import logging
import os
import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ai.binning import kinematics
from ai.evaluation.metrics import sp_index

logger = logging.getLogger(__name__)

#: Column contract of the canonical long table. Anything writing `folds_long.csv` must emit these.
LONG_COLUMNS: List[str] = [
    "model", "et_bin", "eta_bin", "fold", "operating_point",
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
    """Column-group label for an Et bin, or for an ungridded run (NaN)."""
    if pd.isna(et_bin):
        return "all $E_T$" if latex else "all Et"
    return kinematics.et_range_str(int(et_bin), latex)


def _eta_label(eta_bin: float, latex: bool) -> str:
    """Row label for an |eta| bin, or for an ungridded run (NaN)."""
    if pd.isna(eta_bin):
        return r"all $|\eta|$" if latex else "all |eta|"
    return kinematics.eta_range_str(int(eta_bin), latex)


def discover_regions(
    results_root: str = "results",
    model_names: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """
    Inventories what exists on disk under a results tree: one row per region with 'model',
    'region', 'path', 'folds_trained', 'folds_evaluated' and 'evaluated'.

    Discovery is anchored on each region's `checkpoints/` directory rather than on its metrics,
    so a region that was trained but never evaluated is still found and reported as missing
    instead of silently leaving a hole in the table. Model and region are read from the path:
    `<results_root>/<model>/<region>`, or `<results_root>/<model>` for an ungridded run.
    """
    targets = list(model_names) if model_names else ["*"]
    checkpoint_dirs = sorted({
        path
        for target in targets
        for path in glob.glob(os.path.join(results_root, target, "**", "checkpoints"), recursive=True)
    })

    rows = []
    for checkpoint_dir in checkpoint_dirs:
        region_dir = os.path.dirname(checkpoint_dir)
        parts = os.path.relpath(region_dir, results_root).split(os.sep)
        model, region = parts[0], (parts[1] if len(parts) > 1 else "full phase space")

        sidecars = glob.glob(os.path.join(checkpoint_dir, "fold_*.json"))
        if not sidecars:
            continue
        long_path = os.path.join(region_dir, "metrics", "folds_long.csv")

        folds_evaluated = 0
        if os.path.exists(long_path):
            try:
                folds_evaluated = int(pd.read_csv(long_path)["fold"].nunique())
            except Exception:
                folds_evaluated = 0

        rows.append({
            "model": model,
            "region": region,
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
    """The `--et-bin/--eta-bin` arguments selecting a region label, '' for an unbinned one."""
    match = re.fullmatch(r"et(\d+)_eta(\d+)", region)
    if not match:
        return ""
    return f"--et-bin {match.group(1)} --eta-bin {match.group(2)}"


def collect(
    results_root: str = "results",
    model_names: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """
    Scans a results tree for per-region `metrics/folds_long.csv` files and concatenates them
    (an empty frame with LONG_COLUMNS if none is found).

    Both layouts are picked up: the ungridded `results/<MODEL>/metrics/` (et_bin/eta_bin are
    NaN, i.e. one network over the whole phase space) and the grid
    `results/<MODEL>/et<i>_eta<j>/metrics/`.
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
    Which models appear as rows, and in what order.

    An explicit list wins and is honoured verbatim (so `--models CNN2D,MLP` puts the CNN first),
    dropping any name that has no evaluated region; otherwise every model present is used,
    sorted for a stable table across runs.
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
    Reduces the long table over folds into `<metric>_mean` / `<metric>_std` plus `n_folds`, one
    row per (model, et_bin, eta_bin, operating_point).

    PD/SP/FA are converted to percent here - the tables are always quoted in percent, and doing
    the conversion once at aggregation keeps the renderers free of unit logic.
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
        n_folds=("fold", "nunique"),
    ).reset_index()

    # A single fold yields std=NaN; 0.0 is the honest reading (no spread observed).
    for column in [c for c in agg.columns if c.endswith("_std")]:
        agg[column] = agg[column].fillna(0.0)

    return agg


def format_cell(mean: float, std: float, decimals: int = 2, latex: bool = False) -> str:
    """One mean+/-std cell, or '--' when the value is missing."""
    if pd.isna(mean):
        return "--"
    separator = r" $\pm$ " if latex else "±"
    return f"{mean:.{decimals}f}{separator}{std:.{decimals}f}"


def _pivot(
    agg: pd.DataFrame,
    rows: List[str],
    group: str,
    index: pd.Index,
    groups: Sequence[str],
    decimals: int,
    latex: bool
) -> pd.DataFrame:
    """
    Pivots aggregate rows into a printed table: the `index` rows, one (group, metric) column
    pair per entry of `groups`, cells formatted as mean±std and '--' where there is no result.
    """
    names = [tex if latex else plain for _, plain, tex in TABLE_METRICS]
    cells = pd.DataFrame(
        {
            name: [format_cell(mean, std, decimals, latex)
                   for mean, std in zip(agg[f"{key}_mean"], agg[f"{key}_std"])]
            for name, (key, _, _) in zip(names, TABLE_METRICS)
        },
        index=pd.MultiIndex.from_frame(agg[rows + [group]]),
    )
    wide = cells.unstack(group).swaplevel(axis=1)
    return wide.reindex(index=index, columns=pd.MultiIndex.from_product([groups, names])).fillna("--")


def build_wide(
    agg: pd.DataFrame,
    operating_point: str,
    model_names: Optional[Sequence[str]] = None,
    decimals: int = 2,
    latex: bool = False
) -> pd.DataFrame:
    """
    The printed table of one operating point: one row per (|eta| region, model), one column
    per (Et region, metric) pair. A run with no kinematic binning renders as a single
    'all Et' / 'all |eta|' cell; mixed with binned regions, only the binned ones are shown.
    """
    subset = agg[agg["operating_point"] == operating_point]
    models = resolve_models(subset, model_names)
    if not models:
        return pd.DataFrame()

    gridded = subset[subset["et_bin"].notna() & subset["eta_bin"].notna()]
    subset = (gridded if not gridded.empty else subset).sort_values(["eta_bin", "et_bin"])
    subset = subset.assign(
        eta=[_eta_label(value, latex) for value in subset["eta_bin"]],
        et=[_et_label(value, latex) for value in subset["et_bin"]],
    )
    index = pd.MultiIndex.from_product([subset["eta"].unique(), models], names=["Det. Region", "Model"])
    et_groups = subset.sort_values("et_bin")["et"].unique()
    return _pivot(subset, ["eta", "model"], "et", index, et_groups, decimals, latex)


def build_integrated_wide(
    agg: pd.DataFrame,
    model_names: Optional[Sequence[str]] = None,
    decimals: int = 2,
    latex: bool = False,
    operating_points: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """
    The phase-space totals as their own table: one row per model, one column group per
    operating point (in `operating_points` order, else as found). Every region is already
    pooled, so this is the whole result set on a single line per model.
    """
    models = resolve_models(agg, model_names)
    if not models:
        return pd.DataFrame()

    points = list(operating_points or dict.fromkeys(agg["operating_point"]))
    return _pivot(agg, ["model"], "operating_point", pd.Index(models, name="Model"), points, decimals, latex)


def _to_latex(wide: pd.DataFrame, caption: str, label: str) -> str:
    """
    A wide table as a standalone LaTeX fragment, ready for \\input{}, with the PD column of
    each group shaded - PD is the quantity every network was tuned to reproduce.
    """
    pd_columns = [column for column in wide.columns if column[1] == TABLE_METRICS[0][2]]
    tabular = wide.style.map(
        lambda value: "" if value == "--" else "cellcolor:{green!25};", subset=pd_columns
    ).to_latex(
        hrules=True,
        multicol_align="c",
        multirow_align="naive",
        clines="skip-last;data" if wide.index.nlevels > 1 else None,
        column_format="l" * wide.index.nlevels + "c" * len(wide.columns),
    )
    # Styler separates the |eta| blocks with \cline, the last one included; booktabs wants a
    # \midrule between blocks and nothing right above \bottomrule.
    tabular = re.sub(r"\\cline\{[^}]*\}\n(\\bottomrule)?", lambda m: m.group(1) or "\\midrule\n", tabular)
    return "\n".join([
        "% Generated by ai/evaluation/pd_table.py - do not edit by hand.",
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
    """The same table as an HTML page, so the numbers can be read without a LaTeX toolchain."""
    pd_columns = [column for column in wide.columns if column[1] == TABLE_METRICS[0][1]]
    return wide.style.map(
        lambda value: "" if value == "--" else "background-color: #d6f5d6;", subset=pd_columns
    ).set_caption(title).set_table_styles(
        [{"selector": "th, td", "props": "border: 1px solid #bbb; padding: 2px 8px; text-align: center;"}]
    ).to_html()


def _write_table(
    output_dir: str,
    stem: str,
    tex: pd.DataFrame,
    plain: pd.DataFrame,
    caption: str,
    label: str,
    title: str
) -> List[str]:
    """Writes one table as `<stem>.tex` and `<stem>.html`."""
    paths = [os.path.join(output_dir, f"{stem}.tex"), os.path.join(output_dir, f"{stem}.html")]
    for path, source in zip(paths, (_to_latex(tex, caption, label), _to_html(plain, title))):
        with open(path, "w") as handle:
            handle.write(source)
        logger.info(f"\U0001f4c4 Saved table to: {path}")
    return paths


def _fold_text(subset: pd.DataFrame) -> str:
    """The fold count for a caption: the number when it is uniform, a generic phrase otherwise."""
    counts = sorted({int(value) for value in subset["n_folds"].unique()})
    return f"{counts[0]} folds" if len(counts) == 1 else "validação cruzada"


def check_comparable(long_df: pd.DataFrame) -> List[str]:
    """
    Checks that, within each kinematic region, every model was scored on the same holdout,
    returning one message per region where the models disagree.

    The comparison is only meaningful if the models saw identical test rows, which happens
    when their configs agree on data_path, max_files, n_splits and seed. The scored rows'
    signal/background counts are a cheap proxy for that: if they differ between two models in
    the same region, the configs drifted apart and the rows are not comparable.
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


def build_report(
    results_root: str = "results",
    model_names: Optional[Sequence[str]] = None,
    output_dir: Optional[str] = None,
    decimals: int = 2,
    integrated: bool = True
) -> Dict[str, List[str]]:
    """
    End-to-end table build: collect -> aggregate -> a LaTeX fragment and an HTML view per
    operating point, plus the integrated table (every region pooled into the phase-space
    total) unless `integrated` is False. Returns the written paths by artefact kind.

    Pass several names in `model_names` to get one comparison table per operating point, with
    a row per (|eta| region, model) in that order. Since every network is tuned to the same
    target PD, the comparison reads straight down the SP and FA columns. The artefacts go to
    the model's own 'pd_table' directory, or '<results_root>/comparison/pd_table' for several
    models, unless `output_dir` is given.
    """
    written: Dict[str, List[str]] = {"long": [], "tables": [], "integrated": []}

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

    if output_dir is None:
        single = model_names and len(model_names) == 1
        output_dir = os.path.join(results_root, model_names[0] if single else "comparison", "pd_table")
    os.makedirs(output_dir, exist_ok=True)

    long_path = os.path.join(output_dir, "pd_table_long.csv")
    long_df.to_csv(long_path, index=False)
    written["long"].append(long_path)
    logger.info(f"📝 Saved canonical long table to: {long_path}")

    agg = aggregate(long_df)
    points = list(dict.fromkeys(long_df["operating_point"]))

    for point in points:
        tex = build_wide(agg, point, model_names, decimals, latex=True)
        if tex.empty:
            logger.warning(f"⚠️ No rows for operating point '{point}'; skipping its table.")
            continue
        caption = (
            f"Valores de eficiência ($P_D$, $SP$, $F_A$) obtidos a partir da validação cruzada "
            f"({_fold_text(agg[agg['operating_point'] == point])}), em cada região do espaço de "
            f"fase, para o ponto de operação \\textit{{{point}}}."
        )
        written["tables"] += _write_table(
            output_dir, f"pd_table_{point}", tex, build_wide(agg, point, model_names, decimals),
            caption, f"tab:pd_table_{point}", f"Cross Validation — operating point: {point}"
        )

    if integrated:
        # The phase-space total is saved as its own artefact rather than folded into the grid
        # above: it answers a different question and is usually quoted on its own.
        integrated_long = integrate(long_df)
        if not integrated_long.empty:
            csv_path = os.path.join(output_dir, "pd_table_integrated_long.csv")
            integrated_long.to_csv(csv_path, index=False)
            written["integrated"].append(csv_path)
            logger.info(f"\U0001f4dd Saved integrated long table to: {csv_path}")

            agg_integrated = aggregate(integrated_long)
            tex = build_integrated_wide(agg_integrated, model_names, decimals, latex=True, operating_points=points)
            if tex.empty:
                logger.warning("⚠️ No integrated rows to render.")
            else:
                caption = (
                    f"Valores de eficiência ($P_D$, $SP$, $F_A$) integrados em todo o espaço de fase, "
                    f"obtidos a partir da validação cruzada ({_fold_text(agg_integrated)}), para cada "
                    f"ponto de operação. Cada região contribui em proporção à sua população."
                )
                written["integrated"] += _write_table(
                    output_dir, "pd_table_integrated", tex,
                    build_integrated_wide(agg_integrated, model_names, decimals, operating_points=points),
                    caption, "tab:pd_table_integrated", "Cross Validation — integrated over the phase space"
                )

    return written
