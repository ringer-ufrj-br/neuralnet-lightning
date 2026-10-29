"""
Task-independent half of `evaluate`: pick each fold's best initialisation, run its checkpoint
over the whole region and persist the scores. Turning scores into figures of merit is the
task layer's job (neuralnet_lightning.binary.summary).
"""

import glob
import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import polars as pl
import torch

from neuralnet_lightning.paths import RunPaths

logger = logging.getLogger(__name__)

#: Rows per forward pass at inference. A region can hold millions of rows, too many for one pass.
INFERENCE_BATCH = 8192


def best_inits(paths: RunPaths, mode: str) -> Dict[int, Dict[str, Any]]:
    """
    The sidecar of each fold's best initialisation - by the monitored score, in `mode`
    ("max" or "min") - keyed by fold number. Training keeps every (fold, init) model; this is
    where one is picked per fold. An init without a score (no validation epoch completed)
    ranks last.
    """
    worst = float("-inf") if mode == "max" else float("inf")
    pick = max if mode == "max" else min

    candidates: Dict[int, list] = {}
    for path in sorted(glob.glob(os.path.join(paths.checkpoints, "fold_*_init_*.json"))):
        with open(path) as handle:
            info = json.load(handle)
        candidates.setdefault(int(info["fold"]), []).append(info)
    return {
        fold: pick(infos, key=lambda info: worst if info.get("best_score") is None else info["best_score"])
        for fold, infos in sorted(candidates.items())
    }


def predict(model: torch.nn.Module, X: np.ndarray, batch_size: int = INFERENCE_BATCH) -> np.ndarray:
    """`model.predict_scores` over X in batches, as a flat float array."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            chunk = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32)
            outputs.append(model.predict_scores(chunk).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


def score_folds(
    model_cls: type,
    X: np.ndarray,
    y: np.ndarray,
    infos: Dict[int, Dict[str, Any]],
    paths: RunPaths,
    extra_columns: Optional[Dict[str, np.ndarray]] = None
) -> Dict[int, np.ndarray]:
    """
    Scores every fold's chosen checkpoint over the WHOLE region, in-sample rows included, and
    writes `scores/fold_N.parquet` (y_true, y_score, in_sample, plus `extra_columns`, e.g. the
    kinematics, so the table can be re-cut later without re-running inference).

    Train and validation are separated during training - that drives early stopping and model
    selection - but the reported figures deliberately cover the full phase space. The
    `in_sample` column keeps an out-of-sample-only cut available.
    """
    os.makedirs(paths.scores, exist_ok=True)
    scores: Dict[int, np.ndarray] = {}
    for fold, info in sorted(infos.items()):
        checkpoint = paths.resolve(info["checkpoint"])
        if not os.path.exists(checkpoint):
            logger.warning(f"⚠️ Fold {fold}: checkpoint '{checkpoint}' is missing; skipping.")
            continue
        logger.info(f"🧠 Fold {fold}: init {info['init']} ({info['monitor']}={info['best_score']})")

        # fold_kwargs (e.g. pos_weight) are not hyperparameters - they come from the training
        # split - so Lightning cannot rebuild them from the checkpoint. Fed back from the sidecar.
        model = model_cls.load_from_checkpoint(checkpoint, map_location="cpu", **info.get("fold_kwargs", {}))
        y_score = predict(model, X)
        scores[fold] = y_score

        in_sample = np.ones(len(y), dtype=bool)
        in_sample[np.load(paths.resolve(info["val_indices"]))] = False
        pl.DataFrame({"y_true": y, "y_score": y_score, "in_sample": in_sample, **(extra_columns or {})}) \
            .write_parquet(paths.scores_file(fold))
        logger.info(f"💾 Fold {fold}: {int((~in_sample).sum())} of {len(y)} rows out of sample; "
                    f"scores saved to {paths.scores_file(fold)}")
    return scores


def load_scores(infos: Dict[int, Dict[str, Any]], paths: RunPaths) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    (y_true, fold -> y_score) of an earlier evaluation, without touching data or models - so
    working points and plots can be recut in seconds.
    """
    y_true: Optional[np.ndarray] = None
    scores: Dict[int, np.ndarray] = {}
    for fold in sorted(infos):
        path = paths.scores_file(fold)
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ --reuse-scores was given but '{path}' does not exist.")
        frame = pl.read_parquet(path, columns=["y_true", "y_score"])
        y_true = frame["y_true"].to_numpy()
        scores[fold] = frame["y_score"].to_numpy()
        logger.info(f"📂 Reusing cached scores for fold {fold}: {path}")
    return y_true, scores
