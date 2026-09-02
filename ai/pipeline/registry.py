"""
Name -> pipeline registry.

The `model:` string in a YAML config is looked up here, so adding an architecture never means
editing run.py. A pipeline registers itself with a decorator:

    @register_pipeline("MyNet")
    class PipelineMyNet(BasePipeline):
        ...

Every `ai/pipeline/pipeline_*.py` module is imported on first lookup, so simply dropping the
file into the package is enough for its decorator to run - there is no list to append to.
"""

import importlib
import logging
import pkgutil
from typing import Dict, List, Type

logger = logging.getLogger(__name__)

_PIPELINES: Dict[str, type] = {}
_DISCOVERED = False


def register_pipeline(name: str):
    """
    Registers a pipeline class under the name used in a config's `model:` field.

    Args:
        name (str): Config name, e.g. 'MLP'. Also becomes the results/<NAME>/ directory.

    Returns:
        Callable: The class decorator.

    Raises:
        ValueError: If the name is already taken by a different class.
    """
    def decorator(cls: type) -> type:
        existing = _PIPELINES.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Pipeline name '{name}' is already registered to {existing.__name__}; "
                f"{cls.__name__} cannot claim it too."
            )
        _PIPELINES[name] = cls
        cls.model_name = name
        return cls
    return decorator


def _discover() -> None:
    """
    Imports every ai.pipeline.pipeline_* module once, so their decorators populate the
    registry. A module that fails to import is warned about rather than fatal: one broken
    architecture must not stop the others from running.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True

    import ai.pipeline
    for module in pkgutil.iter_modules(ai.pipeline.__path__):
        if not module.name.startswith("pipeline_"):
            continue
        try:
            importlib.import_module(f"ai.pipeline.{module.name}")
        except Exception as exc:
            logger.warning(f"⚠️ Could not import ai.pipeline.{module.name}: {exc}")


def available_pipelines() -> List[str]:
    """
    Lists every registered pipeline name.

    Returns:
        List[str]: Registered names, sorted.
    """
    _discover()
    return sorted(_PIPELINES)


def get_pipeline(name: str) -> Type:
    """
    Resolves a config's `model:` string to its pipeline class.

    Args:
        name (str): The registered name.

    Returns:
        Type: The pipeline class.

    Raises:
        ValueError: If no pipeline is registered under that name.
    """
    _discover()
    if name not in _PIPELINES:
        raise ValueError(
            f"❌ Model '{name}' is not registered. Available: {', '.join(available_pipelines())}.\n"
            f"   To add one, create ai/pipeline/pipeline_<name>.py with a @register_pipeline "
            f"decorated class - see ai/pipeline/pipeline_mlp.py."
        )
    return _PIPELINES[name]
