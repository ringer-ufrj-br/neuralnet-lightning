"""
Registry mapping a config's `model:` name to its class:

    @register_model("MyNet")
    class MyNet(BaseBinaryClassifier):
        preprocessor = PreprocessMLP
        def build_network(self, input_dim: int = 50) -> nn.Module: ...

Every module of neuralnet_lightning.models is imported on first lookup. A registered class
provides (BaseBinaryClassifier implements all of them):

    preprocessor        BasePreprocessor subclass producing the model input
    monitor             metric for early stopping and checkpointing; `monitor_mode` "max" or "min"
    fold_kwargs(y)      constructor kwargs derived from a fold's training labels
    stratify_labels(y)  labels the K-Fold stratifies on (None: plain KFold)
    predict_scores      model input -> scores
    summarize           one region's scores -> metrics and plots
"""

import importlib
import logging
import pkgutil
from typing import Dict, List

logger = logging.getLogger(__name__)

_MODELS: Dict[str, type] = {}
_DISCOVERED = False


def register_model(name: str):
    """Class decorator registering a model under the name used in a config's `model:` field."""
    def decorator(cls: type) -> type:
        existing = _MODELS.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(f"Model name '{name}' is already registered to {existing.__name__}; "
                             f"{cls.__name__} cannot claim it too.")
        _MODELS[name] = cls
        return cls
    return decorator


def _discover() -> None:
    """Imports every neuralnet_lightning.models module once; import failures are logged as warnings."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True

    import neuralnet_lightning.models
    for module in pkgutil.iter_modules(neuralnet_lightning.models.__path__):
        if module.name == "registry":
            continue
        try:
            importlib.import_module(f"neuralnet_lightning.models.{module.name}")
        except Exception as exc:
            logger.warning(f"⚠️ Could not import neuralnet_lightning.models.{module.name}: {exc}")


def available_models() -> List[str]:
    """Every registered model name, sorted."""
    _discover()
    return sorted(_MODELS)


def get_model(name: str) -> type:
    """Resolves a config's `model:` string to its class (ValueError if unknown)."""
    _discover()
    if name not in _MODELS:
        raise ValueError(f"❌ Model '{name}' is not registered. Available: {', '.join(available_models())}.\n"
                         f"   To add one, create neuralnet_lightning/models/<name>.py with a @register_model "
                         f"decorated class - see neuralnet_lightning/models/mlp.py.")
    return _MODELS[name]
