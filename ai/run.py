import argparse
import yaml
import sys
import os
import logging
from typing import Dict, Any

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


def main() -> None:
    """
    Main orchestrator function for Neural Network Training (ATLAS CERN).

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Neural Network Training Orchestrator (ATLAS CERN).")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to YAML configuration file.")
    parser.add_argument('--fold', type=int, default=None, help="Execute a specific fold (useful for SLURM parallelism).")
    parser.add_argument('--accelerator', type=str, default=None, help="PyTorch Lightning accelerator (e.g. auto, cpu, cuda).")
    parser.add_argument('--devices', type=str, default=None, help="Devices to use (e.g. auto, 1, 0).")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"❌ Configuration file '{args.config}' not found.")
        sys.exit(1)

    logger.info(f"⚙️ Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    accelerator = args.accelerator or config.get("accelerator", "auto")
    devices = args.devices or config.get("devices", "auto")
    
    model_type = config.get("model", "CNN2D")
    
    if model_type == "CNN2D":
        from ai.pipeline.pipeline_cnn2d import PipelineCNN2D
        pipeline = PipelineCNN2D(
            data_path=config.get("data_path"),
            max_files=config.get("max_files"),
            label_col=config.get("label_col", "label"),
            model_name=model_type,
            max_epochs=config.get("max_epochs", 20),
            batch_size=config.get("batch_size", 32),
            patience=config.get("patience", 5),
            num_workers=config.get("num_workers", 0),
            accelerator=accelerator,
            devices=devices
        )
    elif model_type == "MLP":
        from ai.pipeline.pipeline_mlp import PipelineMLP
        pipeline = PipelineMLP(
            data_path=config.get("data_path"),
            max_files=config.get("max_files"),
            label_col=config.get("label_col", "has_truth_clus"),
            model_name=model_type,
            max_epochs=config.get("max_epochs", 20),
            batch_size=config.get("batch_size", 32),
            patience=config.get("patience", 5),
            num_workers=config.get("num_workers", 0),
            accelerator=accelerator,
            devices=devices
        ) 
    else:
        raise ValueError(f"❌ Model '{model_type}' is not supported or not implemented in pipeline.")

    # Start complete workflow: load, train, and evaluate
    pipeline.run(
        use_kfold=config.get("use_kfold", False),
        n_splits=config.get("n_splits", 5),
        test_size=config.get("test_size", 0.15),
        learning_rate=config.get("learning_rate", 0.001),
        threshold=config.get("threshold", 0.5),
        target_fold=args.fold
    )

if __name__ == "__main__":
    main()
