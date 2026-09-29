import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from dataclasses import dataclass
import glob
import numpy as np
import os
import sys
import logging
from typing import Iterator, Tuple, List, Dict, Any, Type, Optional, Union
from sklearn.model_selection import StratifiedKFold, train_test_split

logger = logging.getLogger(__name__)

#: What EarlyStopping, ModelCheckpoint and the choice between initialisations follow: the SP
#: index maximised over every threshold (see BaseBinaryClassifier). Higher is better.
MONITOR = "val_sp"

#: Validation fraction of the single split used when n_splits < 2 opts out of cross-validation.
VALIDATION_SPLIT = 0.2

# tqdm repaints its bar with a carriage return: a terminal overwrites the line in place, but a
# redirected log keeps every repaint, and Lightning re-announces its banners on every fit. Off
# the terminal both are dropped and LossHistoryCallback prints one line per epoch instead.
_INTERACTIVE = sys.stderr.isatty()
if not _INTERACTIVE:
    for _noisy in ("pytorch_lightning", "lightning_fabric"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)


class TensorBatchLoader:
    """
    Batch iterator over in-memory tensors that yields whole batches by fancy-indexing the
    underlying tensors, instead of torch's DataLoader-over-TensorDataset path which fetches
    every row individually and re-stacks 128 one-row tensors per batch in Python. For the
    small models in this project that per-item overhead - not the forward pass - was the
    training bottleneck. Works equally for CPU tensors and GPU-staged tensors (indexing
    happens on whatever device the tensors live on, so nothing is copied per batch).

    `indices` is the row subset drawn from (one fold's train or validation split), None for
    every row; `shuffle` re-shuffles the order on every epoch.
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


def compute_pos_weight(y: torch.Tensor) -> float:
    """pos_weight = n_negatives / n_positives over the given (training) labels, 1.0 without positives."""
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    if n_pos == 0:
        logger.warning("⚠️ compute_pos_weight: No positive samples found in training split. Defaulting pos_weight to 1.0.")
        return 1.0

    pos_weight = n_neg / n_pos
    logger.info(f"⚖️ Class Weight Calculation (Train Split Only): Negatives={n_neg}, Positives={n_pos} -> pos_weight={pos_weight:.4f}")
    return pos_weight


class LossHistoryCallback(Callback):
    """Stores the loss history, and off the terminal logs the line replacing the progress bar."""
    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.val_loss = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        self.train_loss.append(metrics['train_loss_epoch'].item())

        # Runs after on_validation_epoch_end, so this epoch's validation metrics are in hand too.
        if not _INTERACTIVE:
            shown = ('train_loss_epoch', 'val_loss', MONITOR)
            logger.info(f"   epoch {trainer.current_epoch} | " + " | ".join(
                f"{name}={metrics[name].item():.6f}" for name in shown if name in metrics))

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        loss = metrics.get('val_loss')
        if loss is not None:
            self.val_loss.append(loss.item())


@dataclass
class ModelTrainer:
    """
    Runs stratified K-Fold cross-validation, recomputing the positive class weight
    (weighted loss) from each fold's own training split.

    Every (fold, initialisation) model is written to `<checkpoint_dir>/fold_N_init_M.ckpt`, with
    Lightning's logs under `<log_dir>/fold_N_init_M`.
    """

    max_epochs: int
    batch_size: int
    patience: int
    log_dir: str
    checkpoint_dir: str
    accelerator: str = "auto"
    devices: Union[int, str, List[int]] = "auto"

    def _select_gpu_device(self) -> Optional[torch.device]:
        """
        Resolves a single CUDA device to preload the dataset onto, or None if training
        won't run on a single GPU (CPU-only, or multi-device where each process needs its
        own shard and pre-pinning to one device would be wrong).
        """
        if self.accelerator == "cpu" or not torch.cuda.is_available():
            return None
        if self.accelerator not in ("auto", "gpu", "cuda"):
            return None
        if isinstance(self.devices, list) and len(self.devices) > 1:
            return None
        if isinstance(self.devices, int) and self.devices > 1:
            return None
        return torch.device("cuda")

    def _stage_dataset(self, X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        If the whole (X, Y) tensor pair comfortably fits in free GPU memory, moves it there
        once so TensorBatchLoader slices are already GPU-resident and never need a per-step
        host-to-device copy - the CPU stops being the bottleneck for small/medium datasets.
        Falls back to CPU tensors (transferred per batch by Lightning) otherwise.
        """
        device = self._select_gpu_device()
        if device is None:
            return X, Y

        dataset_bytes = X.element_size() * X.nelement() + Y.element_size() * Y.nelement()
        free_bytes, _ = torch.cuda.mem_get_info()

        # Leave headroom for the CUDA context, model weights, activations and gradients
        if dataset_bytes > free_bytes * 0.5:
            logger.info(
                f"↔️ Dataset ({dataset_bytes / 1e6:.0f}MB) too large to keep GPU-resident "
                f"({free_bytes / 1e6:.0f}MB free) — batches will transfer per-step instead."
            )
            return X, Y

        logger.info(f"🚀 Staging full dataset ({dataset_bytes / 1e6:.0f}MB) on {device} — no per-batch host-to-device copy.")
        return X.to(device), Y.to(device)

    def _build_trainer(self, name: str) -> Tuple[pl.Trainer, LossHistoryCallback]:
        """
        A pl.Trainer with the standard EarlyStopping/ModelCheckpoint/loss-history callbacks.

        The best checkpoint is written to the fixed path `<checkpoint_dir>/<name>.ckpt` rather
        than one carrying the epoch and the metric value: save_top_k=1 only prunes within a
        single run, so metric-in-the-name files from earlier runs used to pile up in the same
        directory with no way for a later evaluation step to tell which one was current.
        """
        log_dir = os.path.join(self.log_dir, name)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        loss_callback = LossHistoryCallback()
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[
                EarlyStopping(monitor=MONITOR, patience=self.patience, mode="max", verbose=True),
                ModelCheckpoint(
                    dirpath=self.checkpoint_dir,
                    monitor=MONITOR,
                    save_top_k=1,
                    mode="max",
                    filename=name,
                    auto_insert_metric_name=False
                ),
                loss_callback
            ],
            accelerator=self.accelerator,
            devices=self.devices,
            default_root_dir=log_dir,
            enable_progress_bar=_INTERACTIVE,
            enable_model_summary=_INTERACTIVE
        )
        return trainer, loss_callback

    def fit_kfold(
        self,
        model_class: Type[pl.LightningModule],
        model_kwargs: Dict[str, Any],
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor],
        n_splits: int = 5,
        target_fold: Optional[int] = None,
        seed: int = 42,
        n_inits: int = 1,
        target_init: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Trains every requested (fold, initialisation) pair - a fresh model each, with pos_weight
        recomputed strictly from the fold's training split - and returns one record per model.
        Keeping each fold's best initialisation is BasePipeline.select_best_inits' job, so it
        works the same whether the initialisations ran here or in separate scheduler jobs.

        Splits are stratified on the label so that every fold keeps the dataset's
        signal/background proportion - with the strong class imbalance of this dataset a plain
        KFold can hand a fold a wildly different pos_weight than its siblings, which shows up
        as spurious spread in the cross-validation table. `seed` fixes the partition, so it must
        match across parallel per-fold jobs. n_splits=1 opts out of cross-validation: one model
        on a single stratified train/validation split, through the same machinery.

        `target_fold` / `target_init` (1-indexed) restrict the run to one fold / one
        initialisation, for one training per scheduler job. Several initialisations per fold
        mitigate the influence of local minima.
        """
        X = torch.as_tensor(X, dtype=torch.float32)
        Y = torch.as_tensor(Y, dtype=torch.float32)
        X, Y = self._stage_dataset(X, Y)

        labels = Y.detach().cpu().numpy().flatten()

        if n_splits < 2:
            logger.info(f"🔂 n_splits={n_splits}: training a single model on one stratified "
                        f"{VALIDATION_SPLIT:.0%} validation split instead of cross-validating.")
            train_ids, val_ids = train_test_split(
                np.arange(len(labels)), test_size=VALIDATION_SPLIT,
                random_state=seed, shuffle=True, stratify=labels
            )
            splits = [(np.sort(train_ids), np.sort(val_ids))]
        else:
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = kfold.split(np.zeros(len(labels)), labels)
            logger.info(f"🔁 Starting Cross-Validation with {n_splits} stratified folds (seed={seed})...")

        records: List[Dict[str, Any]] = []

        for fold, (train_ids, val_ids) in enumerate(splits, start=1):
            if target_fold is not None and fold != target_fold:
                continue

            logger.info(f"📌 ==================== Fold {fold}/{n_splits} ====================")

            # Compute pos_weight exclusively on this fold's training indices
            pos_weight = compute_pos_weight(Y[train_ids])

            train_loader = TensorBatchLoader(
                X, Y, torch.as_tensor(train_ids, dtype=torch.long),
                batch_size=self.batch_size, shuffle=True
            )
            val_loader = TensorBatchLoader(
                X, Y, torch.as_tensor(val_ids, dtype=torch.long),
                batch_size=self.batch_size, shuffle=False
            )

            # A job killed mid-fold (a SLURM timeout, a Ctrl-C) leaves its per-init checkpoints
            # and sidecars behind. Clear the ones this run is about to produce, so a rerun
            # starts from a clean slate and never picks a winner among stale leftovers.
            for stale in glob.glob(os.path.join(self.checkpoint_dir, f"fold_{fold}_init_{target_init or '*'}.*")):
                logger.info(f"🧹 Removing a file left by an interrupted run: {stale}")
                os.remove(stale)

            for init in [target_init] if target_init is not None else range(1, n_inits + 1):
                # Distinct but reproducible weights per (seed, fold, init). The data partition
                # is untouched by this - it was already fixed by `seed` above.
                pl.seed_everything(seed * 100_000 + fold * 1_000 + init, workers=True)

                model = model_class(**model_kwargs, pos_weight=pos_weight)
                trainer, loss_callback = self._build_trainer(f"fold_{fold}_init_{init}")

                if n_inits > 1:
                    logger.info(f"🎲 Fold {fold}: initialisation {init}/{n_inits}")

                trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

                best_score = trainer.checkpoint_callback.best_model_score
                records.append({
                    "fold": fold,
                    "init": init,
                    "loss_callback": loss_callback,
                    "pos_weight": pos_weight,
                    "checkpoint": trainer.checkpoint_callback.best_model_path,
                    "best_score": float(best_score) if best_score is not None else None,
                    "epochs": int(trainer.current_epoch),
                    "n_train": int(len(train_ids)),
                    "n_val": int(len(val_ids)),
                    "val_ids": np.asarray(val_ids, dtype=np.int64),
                })
                logger.info(f"✅ Fold {fold}, init {init}: {MONITOR}={records[-1]['best_score']} "
                            f"-> {records[-1]['checkpoint']}")

        logger.info(f"🎉 Training of {len(records)} model(s) completed!")
        return records
