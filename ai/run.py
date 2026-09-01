"""
Entrypoint for the ATLAS/CERN neural network experiments.

Three subcommands, deliberately separate:

    python ai/run.py train    --config ai/configs/mlp.yaml [--fold N] [--et-bin i --eta-bin j]
    python ai/run.py evaluate --config ai/configs/mlp.yaml [--et-bin i --eta-bin j]
    python ai/run.py report   --config ai/configs/mlp.yaml

`train` only produces models and the artefacts needed to reload them; `evaluate` turns those
models into scores, metrics and plots for one kinematic region; `report` aggregates every
evaluated region into the cross-validation table ("tabelao") as LaTeX and as a figure.
"""

import argparse
import yaml
import sys
import os
import logging
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure Python finds the ai package from project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def build_pipeline(config: Dict[str, Any], args: argparse.Namespace) -> Any:
    """
    Instantiates the pipeline for the configured model and kinematic region.

    Args:
        config (Dict[str, Any]): Parsed configuration.
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        Any: A BasePipeline subclass instance.

    Raises:
        ValueError: If the configured model has no pipeline.
    """
    accelerator = args.accelerator or config.get("accelerator", "auto")
    devices = args.devices or config.get("devices", "auto")
    model_type = config.get("model", "CNN2D")

    if model_type == "CNN2D":
        from ai.pipeline.pipeline_cnn2d import PipelineCNN2D
        pipeline_class = PipelineCNN2D
    elif model_type == "MLP":
        from ai.pipeline.pipeline_mlp import PipelineMLP
        pipeline_class = PipelineMLP
    else:
        raise ValueError(f"❌ Model '{model_type}' is not supported or not implemented in pipeline.")

    return pipeline_class(
        data_path=config.get("data_path"),
        max_files=config.get("max_files"),
        label_col=config.get("label_col", "label"),
        model_name=model_type,
        max_epochs=config.get("max_epochs", 20),
        batch_size=config.get("batch_size", 32),
        patience=config.get("patience", 5),
        num_workers=config.get("num_workers", 0),
        accelerator=accelerator,
        devices=devices,
        et_bin=args.et_bin,
        eta_bin=args.eta_bin
    )


def resolve_operating_points(config: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Reads the working point definition from the config, if present.

    Expected shape (name -> target PD, the signal efficiency each network is tuned to hit):

        operating_points:
          tight: 0.90
          medium: 0.95
          loose: 0.99

    Args:
        config (Dict[str, Any]): Parsed configuration.

    Returns:
        Optional[Dict[str, float]]: The mapping, or None to use the built-in defaults.
    """
    points = config.get("operating_points")
    if not points:
        return None
    return {str(name): float(target) for name, target in points.items()}


def resolve_report_models(config: Dict[str, Any], models_arg: Optional[str]) -> Optional[List[str]]:
    """
    Decides which models the table covers. --models and --config are alternatives, not
    companions: either names the models directly, the other names one through the YAML.

    Args:
        config (Dict[str, Any]): Parsed configuration (empty when no --config was given).
        models_arg (Optional[str]): Raw --models value, comma separated.

    Returns:
        Optional[List[str]]: Model names in row order, or None to include every model found.
    """
    if models_arg:
        names = [name.strip() for name in models_arg.split(',') if name.strip()]
        if config.get("model") and config["model"] not in names:
            logger.info(f"ℹ️ --models overrides the config's model ('{config['model']}').")
        return names or None
    if config.get("model"):
        return [config["model"]]
    logger.info("ℹ️ No --config or --models given; including every evaluated model found.")
    return None


def add_common_arguments(parser: argparse.ArgumentParser, config_default: Optional[str] = 'config.yaml') -> None:
    """
    Adds the arguments shared by every subcommand.

    Args:
        parser (argparse.ArgumentParser): Subcommand parser to extend.
        config_default (Optional[str]): Default for --config. None makes the config genuinely
            optional, which is what `report` wants: it reads the results tree, not the data.
    """
    parser.add_argument('--config', type=str, default=config_default, help="Path to YAML configuration file.")
    parser.add_argument('--et-bin', type=int, default=None, help="Et bin index 0-4 (requires --eta-bin too, useful for SLURM parallelism of the 25-network grid).")
    parser.add_argument('--eta-bin', type=int, default=None, help="|eta| bin index 0-4 (requires --et-bin too).")
    parser.add_argument('--accelerator', type=str, default=None, help="PyTorch Lightning accelerator (e.g. auto, cpu, cuda).")
    parser.add_argument('--devices', type=str, default=None, help="Devices to use (e.g. auto, 1, 0).")


def build_parser() -> argparse.ArgumentParser:
    """
    Builds the argument parser with the train/evaluate/report subcommands.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Neural Network Training Orchestrator (ATLAS CERN).",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the cross-validation folds and persist the models.")
    add_common_arguments(train_parser)
    train_parser.add_argument('--fold', type=int, default=None, help="Train a specific fold only (1-indexed; useful for SLURM parallelism).")

    evaluate_parser = subparsers.add_parser("evaluate", help="Score the trained folds and produce metrics and plots for one region.")
    add_common_arguments(evaluate_parser)
    evaluate_parser.add_argument('--reuse-scores', action='store_true', help="Reuse the cached scores/fold_N.parquet instead of re-running inference.")
    evaluate_parser.add_argument('--no-plots', action='store_true', help="Skip figure rendering (metrics and tables only).")

    report_parser = subparsers.add_parser("report", help="Aggregate every evaluated region into the cross-validation table.")
    add_common_arguments(report_parser, config_default=None)
    report_parser.add_argument('--results-root', type=str, default='results', help="Root results directory to scan. Defaults to 'results'.")
    report_parser.add_argument('--output-dir', type=str, default=None, help="Where to write the table. Defaults to results/<MODEL>/tabelao, or results/comparison/tabelao when comparing models.")
    report_parser.add_argument('--models', type=str, default=None, help="Comma-separated models to compare, in row order (e.g. 'MLP,CNN2D'). Alternative to --config; without either, every evaluated model is included.")
    report_parser.add_argument('--no-integrated', action='store_false', dest='integrated', help="Skip the separate integrated table (phase-space total), leaving only the per-region tables.")
    report_parser.add_argument('--formats', type=str, default='tex,pdf', help="Comma-separated render formats: 'tex' plus image extensions such as pdf/png. Defaults to 'tex,pdf'.")
    report_parser.add_argument('--decimals', type=int, default=2, help="Decimal places in the table cells. Defaults to 2.")
    report_parser.add_argument('--list', action='store_true', dest='list_only', help="List the trained/evaluated regions found on disk and exit, without building the table.")

    return parser


def main() -> None:
    """
    Main orchestrator function for Neural Network Training (ATLAS CERN).

    Returns:
        None
    """
    argv = sys.argv[1:]
    # Backwards compatibility with the pre-subcommand CLI (`run.py --config ... --fold N`),
    # which is what older SLURM scripts and notebooks still invoke.
    if argv and argv[0].startswith('-') and argv[0] not in ('-h', '--help'):
        logger.warning("⚠️ No subcommand given; assuming 'train'. Use `run.py train|evaluate|report` explicitly.")
        argv.insert(0, 'train')

    args = build_parser().parse_args(argv)

    # `report` reads the results tree, not the data, so a config is only ever a shorthand for
    # "the model named in it" - and --models says the same thing directly. Requiring both was
    # redundant; requiring either was arbitrary.
    if args.config is None:
        config = {}
    elif os.path.exists(args.config):
        logger.info(f"⚙️ Loading configuration from: {args.config}")
        config = load_config(args.config)
    else:
        logger.error(f"❌ Configuration file '{args.config}' not found.")
        sys.exit(1)

    if args.command == "report":
        from ai.evaluation.tabelao import build_report, discover_regions, log_inventory

        model_names = resolve_report_models(config, args.models)

        if args.list_only:
            log_inventory(discover_regions(args.results_root, model_names), args.results_root)
            return

        written = build_report(
            results_root=args.results_root,
            model_names=model_names,
            output_dir=args.output_dir,
            decimals=args.decimals,
            integrated=args.integrated,
            formats=tuple(fmt.strip() for fmt in args.formats.split(',') if fmt.strip())
        )
        total = sum(len(paths) for paths in written.values())
        if total == 0:
            logger.error("❌ Nothing to report. Run `evaluate` for at least one region first.")
            sys.exit(1)
        logger.info(f"🎉 Cross-validation table written ({total} file(s)):")
        for kind, paths in written.items():
            for path in paths:
                logger.info(f"   [{kind}] {path}")
        return

    pipeline = build_pipeline(config, args)

    if args.command == "train":
        pipeline.train(
            n_splits=config.get("n_splits", 5),
            test_size=config.get("test_size", 0.15),
            learning_rate=config.get("learning_rate", 0.001),
            target_fold=args.fold,
            seed=config.get("seed", 42)
        )
    elif args.command == "evaluate":
        try:
            pipeline.evaluate(
                threshold=config.get("threshold", 0.5),
                operating_points=resolve_operating_points(config),
                reuse_scores=args.reuse_scores,
                make_plots=not args.no_plots
            )
        except (FileNotFoundError, RuntimeError) as exc:
            # These are the expected "you are holding it wrong" failures - region not trained,
            # or the data moved under the stored holdout indices. A stack trace adds nothing.
            logger.error(str(exc))
            sys.exit(1)


if __name__ == "__main__":
    main()
