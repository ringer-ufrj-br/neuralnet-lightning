import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
import glob
import numpy as np
import os
import logging
from typing import Iterator, Tuple, List, Dict, Any, Type, Optional, Union
from sklearn.model_selection import StratifiedKFold, train_test_split

logger = logging.getLogger(__name__)


class TensorBatchLoader:
    """
    Batch iterator over in-memory tensors that yields whole batches by fancy-indexing the
    underlying tensors, instead of torch's DataLoader-over-TensorDataset path which fetches
    every row individually and re-stacks 128 one-row tensors per batch in Python. For the
    small models in this project that per-item overhead - not the forward pass - was the
    training bottleneck. Works equally for CPU tensors and GPU-staged tensors (indexing
    happens on whatever device the tensors live on, so nothing is copied per batch).
    """

    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        batch_size: int = 128,
        shuffle: bool = False
    ) -> None:
        """
        Initializes the loader.

        Args:
            X (torch.Tensor): Feature tensor, full dataset.
            Y (torch.Tensor): Label tensor, full dataset.
            indices (Optional[torch.Tensor]): Row subset this loader draws from (e.g. one
                fold's train or validation split). None uses every row.
            batch_size (int): Rows per batch. Defaults to 128.
            shuffle (bool): Re-shuffle the row order on every epoch. Defaults to False.
        """
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

def _discard_checkpoint(path: Optional[str]) -> None:
    """
    Deletes a losing initialisation's checkpoint. Only the winning initialisation of each fold
    is kept, otherwise n_inits would multiply the checkpoints on disk by n_inits.

    Args:
        path (Optional[str]): Checkpoint path, possibly empty or already gone.
    """
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError as exc:
            logger.warning(f"⚠️ Could not remove checkpoint '{path}': {exc}")


def compute_pos_weight(y: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Computes positive class weight for binary classification using:
    pos_weight = n_negatives / n_positives

    Args:
        y (Union[np.ndarray, torch.Tensor]): Target labels (0 and 1).

    Returns:
        torch.Tensor: 1D FloatTensor containing pos_weight.
    """
    if isinstance(y, torch.Tensor):
        y_arr = y.detach().cpu().numpy().flatten()
    else:
        y_arr = np.asarray(y).flatten()

    n_pos = int((y_arr == 1).sum())
    n_neg = int((y_arr == 0).sum())

    if n_pos == 0:
        logger.warning("⚠️ compute_pos_weight: No positive samples found in training split. Defaulting pos_weight to 1.0.")
        return torch.tensor([1.0], dtype=torch.float32)

    pos_weight_val = float(n_neg) / float(n_pos)
    logger.info(f"⚖️ Class Weight Calculation (Train Split Only): Negatives={n_neg}, Positives={n_pos} -> pos_weight={pos_weight_val:.4f}")
    return torch.tensor([pos_weight_val], dtype=torch.float32)


class LossHistoryCallback(Callback):
    """Callback to store training and validation loss history."""
    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.val_loss = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        loss = metrics.get('train_loss_epoch')
        if loss is None:
            loss = metrics.get('train_loss')
        if loss is not None:
            self.train_loss.append(loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        loss = metrics.get('val_loss')
        if loss is not None:
            self.val_loss.append(loss.item())


class ModelTrainer:
    """
    Generic PyTorch Lightning Trainer manager supporting single-holdout and K-Fold cross-validation
    with dynamic positive class weighting (weighted loss).
    """

    def __init__(
        self, 
        max_epochs: int = 20, 
        batch_size: int = 32, 
        validation_split: float = 0.2, 
        patience: int = 5, 
        log_dir: str = "lightning_logs", 
        num_workers: int = 0,
        gradient_clip_val: Optional[float] = 1.0,
        accelerator: str = "auto",
        devices: Union[int, str, List[int]] = "auto",
        monitor_metric: str = "val_loss",
        monitor_mode: str = "min",
        checkpoint_dir: Optional[str] = None
    ) -> None:
        """
        Initializes ModelTrainer instance.

        Args:
            max_epochs (int): Maximum number of training epochs. Defaults to 20.
            batch_size (int): Training batch size. Defaults to 32.
            validation_split (float): Fraction of dataset reserved for validation in holdout. Defaults to 0.2.
            patience (int): Number of epochs with no improvement on the monitored metric before stopping. Defaults to 5.
            log_dir (str): Output directory for model checkpoints and logs. Defaults to 'lightning_logs'.
            num_workers (int): Accepted for config compatibility; unused now that batches are
                sliced from in-memory tensors instead of assembled by worker processes. Defaults to 0.
            gradient_clip_val (Optional[float]): Value for gradient clipping. Defaults to 1.0.
            accelerator (str): PyTorch Lightning accelerator ('auto', 'cpu', 'cuda', etc.). Defaults to 'auto'.
            devices (Union[int, str, List[int]]): Devices to use ('auto', 1, [0], etc.). Defaults to 'auto'.
            monitor_metric (str): Logged metric name used by EarlyStopping/ModelCheckpoint. Defaults to 'val_loss'.
            monitor_mode (str): 'min' or 'max', matching the monitored metric's improvement direction. Defaults to 'min'.
            checkpoint_dir (Optional[str]): Directory where the best checkpoint of each fold is
                written under a fixed, predictable name. Defaults to log_dir.
        """
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.log_dir = log_dir
        self.num_workers = num_workers
        self.gradient_clip_val = gradient_clip_val
        self.accelerator = accelerator
        self.devices = devices
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.checkpoint_dir = checkpoint_dir or log_dir

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

        Args:
            X (torch.Tensor): Feature tensor.
            Y (torch.Tensor): Label tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (X, Y), possibly GPU-resident.
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

    def _build_trainer(
        self,
        model: pl.LightningModule,
        log_dir: str,
        checkpoint_dir: Optional[str] = None,
        checkpoint_name: Optional[str] = None,
        extra_callbacks: Optional[List[Callback]] = None
    ) -> Tuple[pl.Trainer, LossHistoryCallback]:
        """
        Builds a pl.Trainer with the standard EarlyStopping/ModelCheckpoint/loss-history
        callbacks, monitoring self.monitor_metric. Shared by all fit*/fit_kfold* variants.

        When `checkpoint_name` is given, the best checkpoint is written to a **fixed** path
        (`<checkpoint_dir>/<checkpoint_name>.ckpt`) instead of one carrying the epoch and the
        metric value. save_top_k=1 only prunes within a single run, so metric-in-the-name files
        from earlier runs used to pile up in the same directory with no way for a later
        evaluation step to tell which one was current.

        Args:
            model (pl.LightningModule): Model instance (used for checkpoint filename prefix).
            log_dir (str): Directory for logs.
            checkpoint_dir (Optional[str]): Directory for the checkpoint. Defaults to log_dir.
            checkpoint_name (Optional[str]): Stem of the checkpoint file, without extension.
            extra_callbacks (Optional[List[Callback]]): Additional callbacks to append (e.g. SetEpochCallback).

        Returns:
            Tuple[pl.Trainer, LossHistoryCallback]: The configured trainer and its loss-history callback.
        """
        os.makedirs(log_dir, exist_ok=True)
        checkpoint_dir = checkpoint_dir or log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        if checkpoint_name is not None:
            filename = checkpoint_name
            stale = os.path.join(checkpoint_dir, f"{checkpoint_name}.ckpt")
            if os.path.exists(stale):
                os.remove(stale)
                logger.info(f"🧹 Removed stale checkpoint from a previous run: {stale}")
        else:
            filename = f"{model.__class__.__name__}-{{epoch:02d}}-{{{self.monitor_metric}:.4f}}"

        loss_callback = LossHistoryCallback()
        callbacks = [
            EarlyStopping(monitor=self.monitor_metric, patience=self.patience, mode=self.monitor_mode, verbose=True),
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                monitor=self.monitor_metric,
                save_top_k=1,
                mode=self.monitor_mode,
                filename=filename,
                auto_insert_metric_name=checkpoint_name is None
            ),
            loss_callback
        ]
        callbacks.extend(extra_callbacks or [])

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=callbacks,
            accelerator=self.accelerator,
            devices=self.devices,
            default_root_dir=log_dir,
            gradient_clip_val=self.gradient_clip_val
        )
        return trainer, loss_callback

    def prepare_data(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[TensorBatchLoader, TensorBatchLoader, torch.Tensor]:
        """
        Converts feature/label inputs into batch loaders split into train and validation sets
        (same seeded permutation split random_split used to produce), calculating positive
        class weight strictly from the training split.

        Args:
            X (Union[np.ndarray, torch.Tensor]): Input features.
            Y (Union[np.ndarray, torch.Tensor]): Target labels.

        Returns:
            Tuple[TensorBatchLoader, TensorBatchLoader, torch.Tensor]: (train_loader, val_loader, pos_weight).
        """
        if isinstance(X, np.ndarray):
            X = torch.as_tensor(X, dtype=torch.float32)
        if isinstance(Y, np.ndarray):
            Y = torch.as_tensor(Y, dtype=torch.float32)

        X, Y = self._stage_dataset(X, Y)

        val_size = int(len(X) * self.validation_split)
        train_size = len(X) - val_size
        permutation = torch.randperm(len(X), generator=torch.Generator().manual_seed(42))
        train_indices = permutation[:train_size]
        val_indices = permutation[train_size:]

        # Calculate pos_weight strictly on the training subset
        pos_weight = compute_pos_weight(Y[train_indices.to(Y.device)])

        train_loader = TensorBatchLoader(X, Y, train_indices, batch_size=self.batch_size, shuffle=True)
        val_loader = TensorBatchLoader(X, Y, val_indices, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, pos_weight

    def fit(
        self, 
        model: pl.LightningModule, 
        X: Union[np.ndarray, torch.Tensor], 
        Y: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[pl.Trainer, LossHistoryCallback]:
        """
        Trains a PyTorch Lightning module using simple train/validation holdout.

        Args:
            model (pl.LightningModule): The PyTorch Lightning model instance.
            X (Union[np.ndarray, torch.Tensor]): Input features.
            Y (Union[np.ndarray, torch.Tensor]): Target labels.

        Returns:
            Tuple[pl.Trainer, LossHistoryCallback]: The trained PyTorch Lightning Trainer instance and the loss history callback.
        """
        logger.info("📦 Preparing DataLoaders for Holdout...")
        train_loader, val_loader, pos_weight = self.prepare_data(X, Y)
        
        if hasattr(model, 'set_pos_weight'):
            model.set_pos_weight(pos_weight)

        trainer, loss_callback = self._build_trainer(
            model, self.log_dir,
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_name="holdout"
        )

        logger.info(f"🚀 Starting training for model {model.__class__.__name__}...")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        logger.info(f"✅ Training completed! Best model saved at: {trainer.checkpoint_callback.best_model_path}")
        return trainer, loss_callback

    def _is_better(self, score: Optional[float], reference: Optional[float]) -> bool:
        """
        Compares two monitored scores according to monitor_mode.

        Args:
            score (Optional[float]): Candidate score.
            reference (Optional[float]): Incumbent score.

        Returns:
            bool: True when `score` should replace `reference`. A None candidate never wins;
                any real score beats a None incumbent.
        """
        if score is None:
            return False
        if reference is None:
            return True
        return score > reference if self.monitor_mode == "max" else score < reference

    def fit_kfold(
        self, 
        model_class: Type[pl.LightningModule], 
        model_kwargs: Dict[str, Any], 
        X: Union[np.ndarray, torch.Tensor], 
        Y: Union[np.ndarray, torch.Tensor], 
        n_splits: int = 5, 
        target_fold: Optional[int] = None,
        seed: int = 42,
        n_inits: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Executes K-Fold cross-validation, creating a fresh model instance per fold with
        pos_weight dynamically recomputed strictly from each fold's training split.

        Splits are stratified on the label so that every fold keeps the dataset's
        signal/background proportion - with the strong class imbalance of this dataset a plain
        KFold can hand a fold a wildly different pos_weight than its siblings, which shows up
        as spurious spread in the cross-validation table.

        Args:
            model_class (Type[pl.LightningModule]): Model class to instantiate.
            model_kwargs (Dict[str, Any]): Keyword arguments for model constructor.
            X (Union[np.ndarray, torch.Tensor]): Input features.
            Y (Union[np.ndarray, torch.Tensor]): Target labels.
            n_splits (int): Number of K-Fold splits. Defaults to 5. Pass 1 to skip
                cross-validation and train a single model on one stratified holdout split.
            target_fold (Optional[int]): Target fold number (1-indexed) to train individually. Defaults to None.
            seed (int): Seed for the fold partition. Must match across parallel per-fold jobs
                so they all see the same partition. Defaults to 42.
            n_inits (int): Independent weight initialisations to train per fold, keeping the one
                with the best monitored score. More than one mitigates the influence of local
                minima. Defaults to 1.

        Returns:
            List[Dict[str, Any]]: One record per trained fold with keys 'fold', 'model',
            'trainer', 'loss_callback', 'pos_weight', 'checkpoint', 'best_score' and 'epochs'.
        """
        if isinstance(X, np.ndarray):
            X = torch.as_tensor(X, dtype=torch.float32)
        if isinstance(Y, np.ndarray):
            Y = torch.as_tensor(Y, dtype=torch.float32)

        X, Y = self._stage_dataset(X, Y)

        labels = Y.detach().cpu().numpy().flatten()

        if n_splits < 2:
            # n_splits=1 is the documented way to opt out of cross-validation: one model on a
            # single stratified train/validation split. It still flows through the same fold
            # machinery, so the artefacts, the metrics and the table are produced identically -
            # just with a single fold and therefore a zero spread.
            logger.info(f"🔂 n_splits={n_splits}: training a single model on one stratified "
                        f"{self.validation_split:.0%} validation split instead of cross-validating.")
            train_ids, val_ids = train_test_split(
                np.arange(len(labels)), test_size=self.validation_split,
                random_state=seed, shuffle=True, stratify=labels
            )
            splits = [(np.sort(train_ids), np.sort(val_ids))]
        else:
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = kfold.split(np.zeros(len(labels)), labels)
            logger.info(f"🔁 Starting Cross-Validation with {n_splits} stratified folds (seed={seed})...")

        fold_records: List[Dict[str, Any]] = []

        for fold, (train_ids, val_ids) in enumerate(splits):
            fold_number = fold + 1
            if target_fold is not None and fold_number != target_fold:
                continue

            logger.info(f"📌 ==================== Fold {fold_number}/{n_splits} ====================")

            # Compute pos_weight exclusively on this fold's training indices
            train_labels_fold = Y[train_ids]
            pos_weight_fold = compute_pos_weight(train_labels_fold)

            train_loader = TensorBatchLoader(
                X, Y, torch.as_tensor(train_ids, dtype=torch.long),
                batch_size=self.batch_size, shuffle=True
            )
            val_loader = TensorBatchLoader(
                X, Y, torch.as_tensor(val_ids, dtype=torch.long),
                batch_size=self.batch_size, shuffle=False
            )

            # Inject pos_weight into model constructor kwargs
            current_model_kwargs = dict(model_kwargs)
            current_model_kwargs['pos_weight'] = pos_weight_fold

            # A job killed mid-fold (a SLURM timeout, a Ctrl-C) leaves its per-init checkpoints
            # behind, and those would otherwise sit next to the fold's real checkpoint for ever.
            # Clear them before starting, so a rerun always begins from a clean slate.
            for stale in glob.glob(os.path.join(self.checkpoint_dir, f"fold_{fold_number}_init_*.ckpt")):
                logger.info(f"🧹 Removing checkpoint left by an interrupted run: {stale}")
                _discard_checkpoint(stale)

            # Each initialisation is a full training run from different random weights; the
            # fold keeps only the one that scored best on the monitored metric, which is how
            # the reference method mitigates the influence of local minima.
            best_init: Optional[Dict[str, Any]] = None
            for init in range(1, n_inits + 1):
                # Distinct but reproducible weights per (seed, fold, init). The data partition
                # is untouched by this - it was already fixed by `seed` above.
                pl.seed_everything(seed * 100_000 + fold_number * 1_000 + init, workers=True)

                model = model_class(**current_model_kwargs)

                init_name = f"fold_{fold_number}" if n_inits == 1 else f"fold_{fold_number}_init_{init}"
                init_log_dir = os.path.join(self.log_dir, init_name)
                trainer, loss_callback = self._build_trainer(
                    model, init_log_dir,
                    checkpoint_dir=self.checkpoint_dir,
                    checkpoint_name=init_name
                )

                if n_inits > 1:
                    logger.info(f"🎲 Fold {fold_number}: initialisation {init}/{n_inits}")

                trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

                checkpoint_callback = trainer.checkpoint_callback
                best_score = checkpoint_callback.best_model_score
                score = float(best_score) if best_score is not None else None

                candidate = {
                    "init": init,
                    "model": model,
                    "trainer": trainer,
                    "loss_callback": loss_callback,
                    "checkpoint": checkpoint_callback.best_model_path,
                    "best_score": score,
                    "epochs": int(trainer.current_epoch),
                }

                if best_init is None or self._is_better(score, best_init["best_score"]):
                    if best_init is not None:
                        _discard_checkpoint(best_init["checkpoint"])
                    best_init = candidate
                else:
                    _discard_checkpoint(candidate["checkpoint"])

                if n_inits > 1 and score is not None:
                    logger.info(f"   init {init}: {self.monitor_metric}={score:.6f}")

            # Settle the winner under the plain fold_N name the rest of the pipeline expects.
            final_path = os.path.join(self.checkpoint_dir, f"fold_{fold_number}.ckpt")
            if best_init["checkpoint"] and best_init["checkpoint"] != final_path:
                os.replace(best_init["checkpoint"], final_path)
                best_init["checkpoint"] = final_path

            fold_records.append({
                "fold": fold_number,
                "model": best_init["model"],
                "trainer": best_init["trainer"],
                "loss_callback": best_init["loss_callback"],
                "pos_weight": float(pos_weight_fold.item()),
                "checkpoint": best_init["checkpoint"],
                "best_score": best_init["best_score"],
                "epochs": best_init["epochs"],
                "n_train": int(len(train_ids)),
                "n_val": int(len(val_ids)),
                "val_ids": np.asarray(val_ids, dtype=np.int64),
                "n_inits": n_inits,
                "best_init": best_init["init"],
            })

            if n_inits > 1:
                logger.info(f"🏆 Fold {fold_number}: kept initialisation {best_init['init']}/{n_inits} "
                            f"({self.monitor_metric}={best_init['best_score']})")
            logger.info(f"✅ Fold {fold_number} best model saved at: {best_init['checkpoint']}")

        logger.info(f"🎉 Training of {len(fold_records)} fold(s) completed!")
        return fold_records
