import torch
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
import numpy as np
import os
import logging
from typing import Tuple, List, Dict, Any, Type, Optional, Union
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

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
    Generic PyTorch Lightning Trainer manager supporting single-holdout and K-Fold cross-validation.
    """

    def __init__(
        self, 
        max_epochs: int = 20, 
        batch_size: int = 32, 
        validation_split: float = 0.2, 
        patience: int = 5, 
        log_dir: str = "lightning_logs", 
        num_workers: int = 0
    ) -> None:
        """
        Initializes ModelTrainer instance.

        Args:
            max_epochs (int): Maximum number of training epochs. Defaults to 20.
            batch_size (int): Training batch size. Defaults to 32.
            validation_split (float): Fraction of dataset reserved for validation in holdout. Defaults to 0.2.
            patience (int): Number of epochs with no validation loss improvement before stopping. Defaults to 5.
            log_dir (str): Output directory for model checkpoints and logs. Defaults to 'lightning_logs'.
            num_workers (int): Number of subprocesses for data loading. Defaults to 0.
        """
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.log_dir = log_dir
        self.num_workers = num_workers

    def prepare_data(
        self, 
        X: Union[np.ndarray, torch.Tensor], 
        Y: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Converts feature/label inputs into PyTorch DataLoaders split into train and validation sets.

        Args:
            X (Union[np.ndarray, torch.Tensor]): Input features.
            Y (Union[np.ndarray, torch.Tensor]): Target labels.

        Returns:
            Tuple[DataLoader, DataLoader]: Pair of DataLoaders (train_loader, val_loader).
        """
        if isinstance(X, np.ndarray):
            X = torch.as_tensor(X, dtype=torch.float32)
        if isinstance(Y, np.ndarray):
            Y = torch.as_tensor(Y, dtype=torch.float32)

        dataset = TensorDataset(X, Y)
        
        val_size = int(len(dataset) * self.validation_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        return train_loader, val_loader

    def fit(
        self, 
        model: pl.LightningModule, 
        X: Union[np.ndarray, torch.Tensor], 
        Y: Union[np.ndarray, torch.Tensor]
    ) -> pl.Trainer:
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
        train_loader, val_loader = self.prepare_data(X, Y)
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        loss_callback = LossHistoryCallback()
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=self.patience, mode="min", verbose=True),
            ModelCheckpoint(
                dirpath=self.log_dir, 
                monitor="val_loss", 
                save_top_k=1, 
                mode="min",
                filename=f"{model.__class__.__name__}-{{epoch:02d}}-{{val_loss:.4f}}"
            ),
            loss_callback
        ]
        
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=callbacks,
            accelerator="auto",
            devices="auto",
            default_root_dir=self.log_dir
        )
        
        logger.info(f"🚀 Starting training for model {model.__class__.__name__}...")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        logger.info(f"✅ Training completed! Best model saved at: {trainer.checkpoint_callback.best_model_path}")
        return trainer, loss_callback

    def fit_kfold(
        self, 
        model_class: Type[pl.LightningModule], 
        model_kwargs: Dict[str, Any], 
        X: Union[np.ndarray, torch.Tensor], 
        Y: Union[np.ndarray, torch.Tensor], 
        n_splits: int = 5, 
        target_fold: Optional[int] = None
    ) -> Tuple[List[pl.Trainer], List[pl.LightningModule], List[LossHistoryCallback]]:
        """
        Executes K-Fold cross-validation, creating a fresh model instance per fold.

        Args:
            model_class (Type[pl.LightningModule]): Model class to instantiate.
            model_kwargs (Dict[str, Any]): Keyword arguments for model constructor.
            X (Union[np.ndarray, torch.Tensor]): Input features.
            Y (Union[np.ndarray, torch.Tensor]): Target labels.
            n_splits (int): Number of K-Fold splits. Defaults to 5.
            target_fold (Optional[int]): Target fold number (1-indexed) to train individually. Defaults to None.

        Returns:
            Tuple[List[pl.Trainer], List[pl.LightningModule], List[LossHistoryCallback]]: Lists of (fold_trainers, fold_models, fold_loss_callbacks).
        """
        if isinstance(X, np.ndarray):
            X = torch.as_tensor(X, dtype=torch.float32)
        if isinstance(Y, np.ndarray):
            Y = torch.as_tensor(Y, dtype=torch.float32)

        dataset = TensorDataset(X, Y)
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_trainers = []
        fold_models = []
        fold_loss_callbacks = []
        
        logger.info(f"🔁 Starting Cross-Validation with {n_splits} folds...")
        
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            if target_fold is not None and (fold + 1) != target_fold:
                continue
                
            logger.info(f"📌 ==================== Fold {fold + 1}/{n_splits} ====================")
            
            train_sub = Subset(dataset, train_ids)
            val_sub = Subset(dataset, val_ids)
            
            train_loader = DataLoader(
                train_sub, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=self.num_workers
            )
            val_loader = DataLoader(
                val_sub, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=self.num_workers
            )
            
            model = model_class(**model_kwargs)
            
            fold_log_dir = os.path.join(self.log_dir, f"fold_{fold+1}")
            os.makedirs(fold_log_dir, exist_ok=True)
            
            loss_callback = LossHistoryCallback()
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=self.patience, mode="min", verbose=True),
                ModelCheckpoint(
                    dirpath=fold_log_dir, 
                    monitor="val_loss", 
                    save_top_k=1, 
                    mode="min",
                    filename=f"{model.__class__.__name__}-fold{fold+1}-{{epoch:02d}}-{{val_loss:.4f}}"
                ),
                loss_callback
            ]
            
            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                callbacks=callbacks,
                accelerator="auto",
                devices="auto",
                default_root_dir=fold_log_dir
            )
            
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            fold_trainers.append(trainer)
            fold_models.append(model)
            fold_loss_callbacks.append(loss_callback)
            
            logger.info(f"✅ Fold {fold + 1} best model saved at: {trainer.checkpoint_callback.best_model_path}")
            
        logger.info(f"🎉 Cross-Validation of {n_splits} Folds completed!")
        return fold_trainers, fold_models, fold_loss_callbacks
