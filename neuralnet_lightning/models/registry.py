"""
Name -> model class registry.

The `model:` string in a config is looked up here, so adding an architecture never means
editing the CLI or the pipeline. A model registers itself with a decorator:

    @register_model("MyNet")
    class MyNet(BaseBinaryClassifier):
        preprocessor = PreprocessMLP
        def build_network(self, input_dim: int = 50) -> nn.Module: ...

Every module of neuralnet_lightning.models is imported on first lookup, so dropping the file
into the package is enough for its decorator to run - there is no list to append to.

What the task-independent core (training, evaluation, pipeline) reads from a registered class:

    preprocessor    the BasePreprocessor subclass producing the model's input
    monitor         the logged metric early stopping, checkpointing and the choice between
                    initialisations follow; `monitor_mode` says whether higher ("max") is better
    fold_kwargs(y)  extra constructor kwargs derived from one fold's training targets (e.g. a
                    class weight); stored in the fold sidecar and fed back on reload
    stratify_labels(y)  what the K-Fold stratifies on (None: plain KFold)
    predict_scores  model input -> the scores evaluation persists
    summarize       one region's scores -> metrics table and plots

neuralnet_lightning.binary.model.BaseBinaryClassifier provides all of them for the electron x
jet task; another task supplies its own base class and the core needs no change.
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
    """
    Imports every neuralnet_lightning.models module once, so their decorators populate the
    registry. A module that fails to import is warned about rather than fatal: one broken
    architecture must not stop the others from running.
    """
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
