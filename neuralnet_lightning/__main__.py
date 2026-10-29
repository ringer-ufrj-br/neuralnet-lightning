"""
Command line entrypoint.

    python -m neuralnet_lightning train    --config configs/mlp.yaml [--et-bin i --eta-bin j]
                                           [--fold N [--init M]]
    python -m neuralnet_lightning evaluate --config configs/mlp.yaml [--et-bin i --eta-bin j]
    python -m neuralnet_lightning report   --config configs/mlp.yaml   (or --names a,b)
    python -m neuralnet_lightning grid

`train` only produces models, every (fold, initialisation) pair kept on disk - with
`--fold`/`--init` it trains exactly one, which is what lets a scheduler run one training per
job. `evaluate` picks each fold's best initialisation and turns it into scores, metrics and
plots for one region. `report` aggregates every evaluated region into the cross-validation
table. `grid` prints one 'et eta' line per region, for launchers that fan out over it.
"""

import argparse
import logging
import sys
from typing import List, Optional, Union

logger = logging.getLogger("neuralnet_lightning")


def _devices(value: str) -> Union[int, str]:
    """--devices as Lightning expects it: an int count, or a string ('auto', '0,1')."""
    return int(value) if value.lstrip("-").isdigit() else value


def build_parser() -> argparse.ArgumentParser:
    """The argument parser with the train/evaluate/report/grid subcommands."""
    parser = argparse.ArgumentParser(prog="python -m neuralnet_lightning", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_run_arguments(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--config", required=True, help="Path to the YAML experiment config.")
        sub.add_argument("--et-bin", type=int, default=None, help="Et bin index (with --eta-bin): one network per region.")
        sub.add_argument("--eta-bin", type=int, default=None, help="|eta| bin index (with --et-bin).")
        sub.add_argument("--accelerator", default=None, help="Overrides training.accelerator (auto, cpu, gpu).")
        sub.add_argument("--devices", type=_devices, default=None, help="Overrides training.devices (auto, 1, '0,1').")

    train = subparsers.add_parser("train", help="Train the cross-validation folds and persist the models.")
    add_run_arguments(train)
    train.add_argument("--fold", type=int, default=None, help="Train one fold only (1-indexed).")
    train.add_argument("--init", type=int, default=None, help="Train one initialisation only (1-indexed; needs --fold).")

    evaluate = subparsers.add_parser("evaluate", help="Score each fold's best initialisation; metrics and plots.")
    add_run_arguments(evaluate)
    evaluate.add_argument("--reuse-scores", action="store_true", help="Reuse scores/fold_N.parquet instead of re-running inference.")
    evaluate.add_argument("--no-plots", action="store_true", help="Skip figure rendering.")

    report = subparsers.add_parser("report", help="Aggregate every evaluated region into the cross-validation table.")
    source = report.add_mutually_exclusive_group()
    source.add_argument("--config", default=None, help="Report the experiment this config names.")
    source.add_argument("--names", default=None, help="Comma-separated experiments to compare, in row order.")
    report.add_argument("--results-root", default=None, help="Results tree to scan (default: the config's, else 'results').")
    report.add_argument("--output-dir", default=None, help="Where to write the tables.")
    report.add_argument("--no-integrated", action="store_false", dest="integrated", help="Skip the integrated table.")
    report.add_argument("--decimals", type=int, default=2, help="Decimal places in the table cells.")
    report.add_argument("--list", action="store_true", dest="list_only", help="List trained/evaluated regions and exit.")

    subparsers.add_parser("grid", help="Print one 'et eta' line per region of the kinematic grid.")
    return parser


def _region(args: argparse.Namespace):
    """The --et-bin/--eta-bin region, or None for the whole phase space."""
    from neuralnet_lightning.binning import Region

    if (args.et_bin is None) != (args.eta_bin is None):
        raise ValueError("❌ --et-bin and --eta-bin must be given together.")
    return None if args.et_bin is None else Region(args.et_bin, args.eta_bin)


def _report(args: argparse.Namespace) -> int:
    """The `report` subcommand."""
    from neuralnet_lightning.binary.report import build_report, discover_regions, log_inventory
    from neuralnet_lightning.config import load_config

    names: Optional[List[str]] = None
    results_root = args.results_root
    if args.config:
        config = load_config(args.config)
        names = [config.name]
        results_root = results_root or config.results_root
    elif args.names:
        names = [name.strip() for name in args.names.split(",") if name.strip()]
    results_root = results_root or "results"

    if args.list_only:
        log_inventory(discover_regions(results_root, names), results_root)
        return 0

    written = build_report(results_root, names, args.output_dir, args.decimals, args.integrated)
    total = sum(len(paths) for paths in written.values())
    if total == 0:
        logger.error("❌ Nothing to report. Run `evaluate` for at least one region first.")
        return 1
    logger.info(f"🎉 Cross-validation table written ({total} file(s)).")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Parses `argv` and runs one subcommand; returns the process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args(argv)

    if args.command == "grid":
        from neuralnet_lightning.binning import REGIONS
        for region in REGIONS:
            print(f"{region.et_bin} {region.eta_bin}")
        return 0

    try:
        if args.command == "report":
            return _report(args)

        from neuralnet_lightning.config import load_config
        from neuralnet_lightning.pipeline import Pipeline

        if args.command == "train" and args.init is not None and args.fold is None:
            raise ValueError("❌ --init needs --fold too: one training is one (fold, init) pair.")

        config = load_config(args.config, accelerator=args.accelerator, devices=args.devices)
        logger.info(f"⚙️ Loaded '{args.config}': experiment '{config.name}', model '{config.model}'")
        pipeline = Pipeline(config, _region(args))

        if args.command == "train":
            pipeline.train(fold=args.fold, init=args.init)
        else:
            pipeline.evaluate(reuse_scores=args.reuse_scores, make_plots=not args.no_plots)
        return 0
    except (FileNotFoundError, KeyError, ValueError) as exc:
        # The expected "holding it wrong" failures - a bad config, an untrained region, a
        # missing column - are raised by this package with a ❌ message saying what to fix; a
        # stack trace adds nothing. Anything else is a bug and keeps its traceback.
        message = str(exc).strip("'\"")
        if not message.startswith("❌"):
            raise
        logger.error(message)
        return 1


if __name__ == "__main__":
    sys.exit(main())
