"""K-Fold training: one Lightning fit per (fold, initialisation), plus the artefacts evaluation reads."""

import glob
import json
import logging
import os
import sys
import warnings
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from neuralnet_lightning.config import TrainingConfig
from neuralnet_lightning.paths import RunPaths

logger = logging.getLogger(__name__)

#: Validation fraction of the single split used when n_splits < 2.
VALIDATION_SPLIT = 0.2

# Outside a terminal, progress bars and Lightning's info logs are disabled; LossHistoryCallback
# logs one line per epoch instead.
_INTERACTIVE = sys.stderr.isatty()
if not _INTERACTIVE:
    for _noisy in ("pytorch_lightning", "lightning_fabric"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

# Lightning warnings that are expected in this setup.
for _expected in (r"The `srun` command is available", r".*`tensorboardX` has been removed",
                  r".*smaller than the logging interval", r"Checkpoint directory .* exists and is not empty",
                  r"`isinstance\(treespec, LeafSpec\)` is deprecated"):
    warnings.filterwarnings("ignore", message=_expected)


def atomic_write_json(payload: Dict[str, Any], filepath: str) -> str:
    """Writes JSON atomically (temporary file + os.replace)."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    tmp_path = f"{filepath}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, filepath)
    return filepath


class TensorBatchLoader:
    """
    Batch iterator over in-memory tensors (CPU or GPU) that indexes a whole batch at once.
    `indices` restricts it to a row subset (None: every row); `shuffle` reshuffles every epoch.
    """

    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        batch_size: int = 128,
        shuffle: bool = False
    ) -> None:
        self.X = X
        self.Y = Y
        self.indices = None if indices is None else indices.to(X.device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(X) if indices is None else len(indices)

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if self.shuffle:
            order = torch.randperm(self.n, device=self.X.device)
            order = order if self.indices is None else self.indices[order]
        else:
            order = self.indices
        for start in range(0, self.n, self.batch_size):
            if order is None:
                yield self.X[start:start + self.batch_size], self.Y[start:start + self.batch_size]
            else:
                sel = order[start:start + self.batch_size]
                yield self.X[sel], self.Y[sel]


class LossHistoryCallback(Callback):
    """Records the loss history; outside a terminal, also logs one line per epoch."""

    def __init__(self, monitor: str) -> None:
        super().__init__()
        self.monitor = monitor
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics
        self.train_loss.append(metrics["train_loss_epoch"].item())

        if not _INTERACTIVE:
            shown = ("train_loss_epoch", "val_loss", self.monitor)
            logger.info(f"   epoch {trainer.current_epoch} | " + " | ".join(
                f"{name}={metrics[name].item():.6f}" for name in shown if name in metrics))

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        loss = trainer.callback_metrics.get("val_loss")
        if loss is not None:
            self.val_loss.append(loss.item())


def _gpu_device(cfg: TrainingConfig) -> Optional[torch.device]:
    """The CUDA device to stage the dataset on, or None when not training on a single GPU."""
    if cfg.accelerator not in ("auto", "gpu", "cuda") or not torch.cuda.is_available():
        return None
    if isinstance(cfg.devices, int) and (cfg.devices > 1 or (cfg.devices == -1 and torch.cuda.device_count() > 1)):
        return None
    if isinstance(cfg.devices, str) and "," in cfg.devices.strip(","):  # an explicit list, e.g. "0,1"
        return None
    return torch.device("cuda")


def stage_dataset(X: torch.Tensor, Y: torch.Tensor, cfg: TrainingConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """Moves (X, Y) to the GPU if they fit in half of its free memory; otherwise returns them unchanged."""
    device = _gpu_device(cfg)
    if device is None:
        return X, Y

    dataset_bytes = X.element_size() * X.nelement() + Y.element_size() * Y.nelement()
    free_bytes, _ = torch.cuda.mem_get_info()

    if dataset_bytes > free_bytes * 0.5:
        logger.info(f"↔️ Dataset ({dataset_bytes / 1e6:.0f}MB) too large to keep GPU-resident "
                    f"({free_bytes / 1e6:.0f}MB free) — batches will transfer per step instead.")
        return X, Y

    logger.info(f"🚀 Staging full dataset ({dataset_bytes / 1e6:.0f}MB) on {device} — no per-batch copy.")
    return X.to(device), Y.to(device)


def make_splits(
    n_rows: int,
    n_splits: int,
    seed: int,
    stratify: Optional[np.ndarray]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    (train, validation) row indices of every fold, deterministic given the arguments. Stratified
    on `stratify` when given; n_splits=1 gives a single train/validation split.
    """
    if n_splits < 2:
        train_ids, val_ids = train_test_split(np.arange(n_rows), test_size=VALIDATION_SPLIT,
                                              random_state=seed, shuffle=True, stratify=stratify)
        return [(np.sort(train_ids), np.sort(val_ids))]
    if stratify is not None:
        return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                    .split(np.zeros(n_rows), stratify))
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(n_rows)))


def _build_trainer(cfg: TrainingConfig, paths: RunPaths, stem: str,
                   monitor: str, mode: str) -> Tuple[pl.Trainer, LossHistoryCallback]:
    """Trainer with early stopping, loss history and the best checkpoint saved to `<checkpoints>/<stem>.ckpt`."""
    log_dir = os.path.join(paths.logs, stem)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(paths.checkpoints, exist_ok=True)

    loss_callback = LossHistoryCallback(monitor)
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        callbacks=[
            EarlyStopping(monitor=monitor, patience=cfg.patience, mode=mode, verbose=_INTERACTIVE),
            ModelCheckpoint(dirpath=paths.checkpoints, monitor=monitor, save_top_k=1, mode=mode,
                            filename=stem, auto_insert_metric_name=False),
            loss_callback,
        ],
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        default_root_dir=log_dir,
        enable_progress_bar=_INTERACTIVE,
        enable_model_summary=_INTERACTIVE,
    )
    return trainer, loss_callback


def _remove_stale(paths: RunPaths, fold: int, init: Optional[int]) -> None:
    """Deletes the checkpoints and sidecars left by earlier runs of `fold` (only of `init`, if given)."""
    pattern = f"fold_{fold}_init_{init if init is not None else '*'}.*"
    for stale in glob.glob(os.path.join(paths.checkpoints, pattern)):
        logger.info(f"🧹 Removing a file left by an earlier run: {stale}")
        os.remove(stale)


def fit_kfold(
    model_cls: type,
    model_kwargs: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    cfg: TrainingConfig,
    paths: RunPaths,
    stratify: Optional[np.ndarray] = None,
    fold: Optional[int] = None,
    init: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Trains the requested (fold, init) models, all by default (`fold`/`init` are 1-indexed).
    Writes the checkpoint, loss history and JSON sidecar of each model and the validation
    indices of each fold. Returns the sidecars.
    """
    splits = make_splits(len(y), cfg.n_splits, cfg.seed, stratify)
    if fold is not None and not 1 <= fold <= len(splits):
        raise ValueError(f"❌ --fold {fold} is outside 1-{len(splits)}.")
    if init is not None and not 1 <= init <= cfg.n_inits:
        raise ValueError(f"❌ --init {init} is outside 1-{cfg.n_inits}.")

    X_t, Y_t = stage_dataset(torch.as_tensor(X), torch.as_tensor(y), cfg)
    monitor, mode = model_cls.monitor, model_cls.monitor_mode
    logger.info(f"🔁 {len(splits)} fold(s) x {cfg.n_inits} init(s), seed={cfg.seed}, "
                f"stratified={stratify is not None}, monitoring {monitor} ({mode}).")

    sidecars: List[Dict[str, Any]] = []
    for fold_id, (train_ids, val_ids) in enumerate(splits, start=1):
        if fold is not None and fold_id != fold:
            continue
        logger.info(f"📌 ==================== Fold {fold_id}/{len(splits)} ====================")

        fold_kwargs = model_cls.fold_kwargs(y[train_ids])
        train_loader = TensorBatchLoader(X_t, Y_t, torch.as_tensor(train_ids), cfg.batch_size, shuffle=True)
        val_loader = TensorBatchLoader(X_t, Y_t, torch.as_tensor(val_ids), cfg.batch_size, shuffle=False)

        _remove_stale(paths, fold_id, init)

        # Used by evaluation to flag out-of-sample predictions.
        os.makedirs(paths.artifacts, exist_ok=True)
        np.save(paths.val_indices(fold_id), np.sort(val_ids))

        for init_id in [init] if init is not None else range(1, cfg.n_inits + 1):
            # Reproducible, distinct initial weights per (seed, fold, init).
            pl.seed_everything(cfg.seed * 100_000 + fold_id * 1_000 + init_id, workers=True, verbose=False)
            if cfg.n_inits > 1:
                logger.info(f"🎲 Fold {fold_id}: initialisation {init_id}/{cfg.n_inits}")

            model = model_cls(**model_kwargs, **fold_kwargs)
            stem = paths.stem(fold_id, init_id)
            trainer, losses = _build_trainer(cfg, paths, stem, monitor, mode)
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

            os.makedirs(paths.history, exist_ok=True)
            history_path = paths.history_file(fold_id, init_id)
            pd.DataFrame({
                "epoch": range(max(len(losses.train_loss), len(losses.val_loss))),
                "train_loss": pd.Series(losses.train_loss, dtype=float),
                "val_loss": pd.Series(losses.val_loss, dtype=float),
            }).to_csv(history_path, index=False)

            best = trainer.checkpoint_callback.best_model_score
            sidecar = {
                "fold": fold_id,
                "init": init_id,
                "checkpoint": paths.rel(trainer.checkpoint_callback.best_model_path),
                "val_indices": paths.rel(paths.val_indices(fold_id)),
                "history": paths.rel(history_path),
                "monitor": monitor,
                "monitor_mode": mode,
                "best_score": float(best) if best is not None else None,
                "epochs": int(trainer.current_epoch),
                "n_train": int(len(train_ids)),
                "n_val": int(len(val_ids)),
                "model_kwargs": model_kwargs,
                "fold_kwargs": fold_kwargs,
            }
            atomic_write_json(sidecar, paths.sidecar(fold_id, init_id))
            sidecars.append(sidecar)
            logger.info(f"✅ Fold {fold_id}, init {init_id}: {monitor}={sidecar['best_score']} "
                        f"-> {sidecar['checkpoint']}")

    logger.info(f"🎉 Training of {len(sidecars)} model(s) completed.")
    return sidecars
