"""
Typed experiment configuration, loaded from YAML and validated on load.

The YAML mirrors these dataclasses one to one:

    name: mlp_mc21                 # the experiment: results go to <results_root>/<name>/<region>/
    model: MLP                     # the architecture, looked up in models/registry.py
    dataset: datasets/mc21.yaml    # DatasetSchema: a mapping, a YAML file, or {file: ..., <overrides>}
    model_args: {hidden: 5}        # optional overrides of the model's build_network defaults
    training: {...}                # TrainingConfig
    evaluation: {...}              # EvaluationConfig

`dataset:` names a file so that several experiments share one dataset description instead of
each repeating it; `{file: ..., max_files: 100}` takes the file and overrides some of its keys.
A relative file path is resolved against the folder of the config that names it, so a config
means the same thing whatever directory the command runs from.

The defaults below are the only defaults in the package. An unknown key - typically a typo -
is an error rather than a silently ignored setting, and so is a value of the wrong type.
"""

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, get_args, get_origin, get_type_hints

import yaml

from neuralnet_lightning.data.schema import DatasetSchema


class ConfigError(ValueError):
    """A config that does not describe a valid experiment."""


@dataclass(frozen=True)
class TrainingConfig:
    """
    Everything that shapes how the networks are fitted. `n_splits: 1` opts out of
    cross-validation (one model on a stratified 80/20 split); `n_inits` trains each fold that
    many times from different weights, evaluation keeps the best.
    """

    max_epochs: int = 5000
    batch_size: int = 1024
    learning_rate: float = 0.001
    patience: int = 50
    n_splits: int = 10
    n_inits: int = 5
    seed: int = 42
    accelerator: str = "auto"
    devices: Union[int, str] = "auto"

    def __post_init__(self) -> None:
        for name in ("max_epochs", "batch_size", "patience", "n_splits", "n_inits"):
            if getattr(self, name) < 1:
                raise ConfigError(f"❌ training.{name} must be >= 1 (got {getattr(self, name)}).")
        if self.learning_rate <= 0:
            raise ConfigError(f"❌ training.learning_rate must be > 0 (got {self.learning_rate}).")


@dataclass(frozen=True)
class EvaluationConfig:
    """Operating points: name -> target detection probability (PD) the threshold is set to."""

    operating_points: Dict[str, float] = field(
        default_factory=lambda: {"tight": 0.90, "medium": 0.95, "loose": 0.99}
    )

    def __post_init__(self) -> None:
        if not self.operating_points:
            raise ConfigError("❌ evaluation.operating_points must name at least one point.")
        for name, target in self.operating_points.items():
            if not 0.0 < target < 1.0:
                raise ConfigError(f"❌ evaluation.operating_points.{name} must be in (0, 1) (got {target}).")


@dataclass(frozen=True)
class ExperimentConfig:
    """
    One experiment. `name` and `model` are deliberately separate: the same architecture on
    another dataset, or with other model_args, is another experiment under another name - not
    another class.
    """

    name: str
    model: str
    dataset: DatasetSchema
    results_root: str = "results"
    model_args: Dict[str, Any] = field(default_factory=dict)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or self.name.startswith("."):
            raise ConfigError(f"❌ name must be a plain directory name (got '{self.name}').")


def load_config(path: str, **training_overrides: Any) -> ExperimentConfig:
    """
    Parses and validates a YAML config. `training_overrides` replace training fields after
    loading - the CLI's --accelerator/--devices; None values are ignored.
    """
    config = from_dict(_read_yaml(path, "config"), base_dir=os.path.dirname(os.path.abspath(path)))
    overrides = {key: value for key, value in training_overrides.items() if value is not None}
    if overrides:
        config = dataclasses.replace(config, training=dataclasses.replace(config.training, **overrides))
    return config


def from_dict(raw: Dict[str, Any], base_dir: Optional[str] = None) -> ExperimentConfig:
    """
    Builds an ExperimentConfig from an already-parsed mapping (ConfigError if invalid).
    `base_dir` is where a relative `dataset:` file is looked up (default: the working directory).
    """
    if isinstance(raw, dict) and "dataset" in raw:
        raw = {**raw, "dataset": _resolve_dataset(raw["dataset"], base_dir or os.getcwd())}
    return _build(ExperimentConfig, raw, "config")


def _read_yaml(path: str, what: str) -> Dict[str, Any]:
    """A YAML file as a mapping; a missing file is a ConfigError rather than a traceback."""
    if not os.path.isfile(path):
        raise ConfigError(f"❌ {what.capitalize()} file '{path}' not found.")
    with open(path) as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"❌ {what.capitalize()} file '{path}' must hold a mapping, got {type(raw).__name__}.")
    return raw


def _resolve_dataset(raw: Any, base_dir: str) -> Any:
    """
    The `dataset:` block as one mapping: a string is a YAML file of DatasetSchema keys, and a
    mapping with a `file` key is that file with the mapping's other keys taking precedence.
    Anything else is returned unchanged for _build to validate.
    """
    if isinstance(raw, str):
        raw = {"file": raw}
    if not isinstance(raw, dict) or "file" not in raw:
        return raw
    overrides = {key: value for key, value in raw.items() if key != "file"}
    if not isinstance(raw["file"], str):
        raise ConfigError(f"❌ 'file' in 'dataset' must be a path, got {raw['file']!r}.")
    path = os.path.join(base_dir, raw["file"])
    return {**_read_yaml(path, "dataset"), **overrides}


def _build(cls: type, raw: Any, where: str) -> Any:
    """Instantiates dataclass `cls` from `raw`, recursing into nested dataclass fields."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"❌ '{where}' must be a mapping, got {type(raw).__name__}.")

    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ConfigError(f"❌ Unknown key(s) in '{where}': {unknown}. Valid keys: {sorted(fields)}.")

    hints = get_type_hints(cls)
    kwargs = {}
    for name, f in fields.items():
        if name not in raw:
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                raise ConfigError(f"❌ Missing required key '{name}' in '{where}'.")
            continue
        value, hint = raw[name], hints[name]
        if dataclasses.is_dataclass(hint):
            value = _build(hint, value, name)
        elif not _matches(value, hint):
            raise ConfigError(f"❌ '{name}' in '{where}' must be {_describe(hint)}, got {value!r}.")
        kwargs[name] = value

    try:
        return cls(**kwargs)
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        message = str(exc)
        raise ConfigError(message if message.startswith("❌") else f"❌ {message}") from exc


def _matches(value: Any, hint: Any) -> bool:
    """Whether `value` fits the (simple) type hint. Ints are accepted where floats are."""
    if hint is Any:
        return True
    origin = get_origin(hint)
    if origin is Union:
        return any(_matches(value, arg) for arg in get_args(hint))
    if origin is dict:
        key_hint, value_hint = get_args(hint)
        return isinstance(value, dict) and all(
            _matches(k, key_hint) and _matches(v, value_hint) for k, v in value.items()
        )
    if hint is type(None):
        return value is None
    if isinstance(value, bool):
        return hint is bool
    if hint is float:
        return isinstance(value, (int, float))
    return isinstance(value, hint)


def _describe(hint: Any) -> str:
    """A readable rendering of a type hint, for error messages."""
    if get_origin(hint) is Union:
        return " or ".join(_describe(arg) for arg in get_args(hint))
    if hint is type(None):
        return "null"
    return getattr(hint, "__name__", str(hint))

