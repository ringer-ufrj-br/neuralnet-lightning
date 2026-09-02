"""
TEMPLATE: how to add a new architecture.

This file is a worked example, not live code - nothing imports it. Copy the three classes
below into the three files named in their comments, rename them, and you are done. There is
no registry list to append to and no `if/elif` in run.py to edit.

The three files, and the ONLY things you have to write:

    ai/models/<name>.py       build_network()          -> the layers
    ai/preprocess/<name>.py   required_columns(), transform()  -> DataFrame -> array
    ai/pipeline/pipeline_<name>.py   two class attributes      -> wires them together

Then set `model: "<Name>"` in your YAML config and run:

    python ai/run.py train    --config ai/configs/<name>.yaml
    python ai/run.py evaluate --config ai/configs/<name>.yaml
    python ai/run.py report   --config ai/configs/<name>.yaml

Everything else - k-fold splitting, kinematic binning, GPU staging, batching, metrics, the SP
index, EarlyStopping, checkpointing, scoring, plots, the tabelao and the SLURM grid - already
works for your architecture without you touching it.

To check your architecture is registered:

    python -c "from ai.pipeline.registry import available_pipelines; print(available_pipelines())"
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch.nn as nn

from ai.models.base import BaseBinaryClassifier
from ai.pipeline.base import BasePipeline
from ai.pipeline.registry import register_pipeline
from ai.preprocess.base import BasePreprocessor


# =============================================================================
# 1. THE MODEL  ->  goes in  ai/models/<name>.py
# =============================================================================

class ModelTemplate(BaseBinaryClassifier):
    """
    Your architecture. Implement build_network and nothing else.

    Do NOT write an __init__: the base builds the loss, the metrics, the SP index hook and
    the optimizer for you, and saves your hyperparameters so load_from_checkpoint works.
    """

    def build_network(self, input_dim: int = 100, hidden: int = 16) -> nn.Module:
        """
        Builds the layers.

        Every keyword argument here is automatically a saved hyperparameter, available later
        as `self.hparams.hidden` and restored from the checkpoint. Give each one a default;
        the pipeline can override any of them via build_model_kwargs (step 3).

        Args:
            input_dim (int): Number of input features.
            hidden (int): Width of the hidden layer.

        Returns:
            nn.Module: Maps (Batch, input_dim) to (Batch, 1) RAW LOGITS.
                Do not apply a sigmoid - the loss does it. The input arrives already
                normalised by the preprocessor.
        """
        return nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    # Optional, only if you need them - delete if you do not:
    #
    # def forward(self, x):              override when the net is not one callable module
    # def compute_loss(self, batch):     override to add auxiliary losses (see models/fused.py)
    # def build_metrics(self):           override to add or drop metrics
    # def configure_optimizers(self):    override for a different optimizer or a scheduler


# =============================================================================
# 2. THE PREPROCESSOR  ->  goes in  ai/preprocess/<name>.py
# =============================================================================

class PreprocessTemplate(BasePreprocessor):
    """
    Turns the raw parquet DataFrame into the array your model consumes.

    save / load / fit_transform / get_labels are all inherited - do not rewrite them. The
    whole instance is pickled, so anything you set on `self` in fit() is restored later.
    """

    def required_columns(self, available: List[str]) -> List[str]:
        """
        Which dataset columns to actually read. The raw files carry 300+ columns; listing only
        what you need here is what keeps a full-dataset run inside memory.

        Args:
            available (List[str]): Column names present in the dataset files.

        Returns:
            List[str]: The columns to load. Return None to load everything (rarely a good idea).
        """
        return [c for c in available if c.startswith("cl_ring_")]

    def fit(self, df: pd.DataFrame) -> "PreprocessTemplate":
        """
        Learns any state from the TRAINING split only (a scaler, a mean, ...). Delete this
        method entirely if your preprocessing is stateless - the base's no-op fit is correct.

        Args:
            df (pd.DataFrame): The rows the pipeline fits on.

        Returns:
            PreprocessTemplate: self.
        """
        self.columns_ = self.required_columns(list(df.columns))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Builds the feature array. End with `self.normalize(...)`, which scales each event by
        its own total so the network sees the shape of the deposition and not its scale.

        Args:
            df (pd.DataFrame): Rows to transform.

        Returns:
            np.ndarray: Float32, first dimension is the batch. The shape here must match what
                your build_network expects.
        """
        return self.normalize(df[self.columns_].to_numpy(dtype=np.float32))


# =============================================================================
# 3. THE PIPELINE  ->  goes in  ai/pipeline/pipeline_<name>.py
#
#    The filename MUST start with "pipeline_" - that is how the registry finds it.
# =============================================================================

@register_pipeline("Template")          # <- the string you put in the config's `model:` field
class PipelineTemplate(BasePipeline):
    """
    Wires the model to the preprocessor. Usually just these two attributes.
    """

    model_class = ModelTemplate
    preprocessor_class = PreprocessTemplate

    def build_model_kwargs(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Optional. Passes values derived from the real data into build_network - typically the
        input dimension, which you only know after preprocessing. Delete this method if your
        architecture's defaults are already correct (see pipeline_cnn2d.py).

        Args:
            X (np.ndarray): The preprocessed training features.

        Returns:
            Dict[str, Any]: Keyword arguments for build_network.
        """
        return {"input_dim": int(X.shape[1])}
