import os
import sys
import torch
import numpy as np
import logging
from typing import Optional, Union, Tuple, List, Any
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Ensure root directory is in path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from ai.loader.loader import DataLoader
from ai.label.label_generator import LabelGenerator
from ai.preprocess.mlp import PreprocessMLP
from ai.models.mlp import ModelMLP
from ai.trainer.trainer import ModelTrainer
from ai.evaluation.monitor import ModelMonitor
from ai.evaluation.summary import ModelSummary

class PipelineMLP:
    """
    End-to-end training and evaluation pipeline for MLP models using weighted loss.
    """

    def __init__(
        self, 
        data_path: Optional[str] = None, 
        max_files: Optional[int] = None, 
        label_col: str = 'label', 
        model_name: str = "MLP", 
        max_epochs: int = 20, 
        batch_size: int = 32, 
        patience: int = 5,
        num_workers: int = 0,
        accelerator: str = "auto",
        devices: Union[int, str, List[int]] = "auto"
    ) -> None:
        """
        Initializes PipelineMLP instance.

        Args:
            data_path (Optional[str]): Data folder or pattern path.
            max_files (Optional[int]): Maximum number of files to process per folder.
            label_col (str): Column name containing labels. Defaults to 'label'.
            model_name (str): Model name for logging and results folder. Defaults to 'MLP'.
            max_epochs (int): Maximum training epochs. Defaults to 20.
            batch_size (int): Training batch size. Defaults to 32.
            patience (int): Early stopping patience. Defaults to 5.
            num_workers (int): Parallel worker subprocesses. Defaults to 0.
            accelerator (str): PyTorch Lightning accelerator ('auto', 'cpu', 'cuda'). Defaults to 'auto'.
            devices (Union[int, str, List[int]]): Devices specification. Defaults to 'auto'.
        """
        self.model_name = model_name
        self.label_col = label_col

        self.results_dir = os.path.join("results", self.model_name)

        self.loader = DataLoader(data_path=data_path, max_files=max_files)
        self.preprocessor = PreprocessMLP()
        
        self.trainer = ModelTrainer(
            max_epochs=max_epochs,
            batch_size=batch_size,
            patience=patience,
            num_workers=num_workers,
            log_dir=os.path.join(self.results_dir, "lightning_logs"),
            accelerator=accelerator,
            devices=devices,
            monitor_metric="val_sp",
            monitor_mode="max"
        )
        
        self.monitor = ModelMonitor(output_dir=os.path.join(self.results_dir, "plots"))
        self.summary = ModelSummary(output_dir=os.path.join(self.results_dir, "metrics"))

    def evaluate_model(
        self, 
        model: torch.nn.Module, 
        X_test: np.ndarray, 
        Y_test: np.ndarray, 
        threshold: float = 0.5, 
        suffix: str = "",
        loss_callback: Optional[Any] = None
    ) -> None:
        """
        Evaluates trained model on unseen test dataset and generates metric reports/plots.

        Args:
            model (torch.nn.Module): Trained model module.
            X_test (np.ndarray): Test feature matrix.
            Y_test (np.ndarray): Test true labels array.
            threshold (float): Classification decision threshold. Defaults to 0.5.
            suffix (str): Filename suffix for evaluation reports. Defaults to ''.
            loss_callback (Optional[Any]): Loss history callback. Defaults to None.

        Returns:
            None
        """
        suffix_print = f" ({suffix})" if suffix else ""
        logger.info(f"📊 Step 4: Evaluating model {self.model_name} (Threshold={threshold}){suffix_print}...")
        
        model.eval()
        with torch.no_grad():
            X_tensor = torch.as_tensor(X_test, dtype=torch.float32)
            logits = model(X_tensor)
            y_prob = torch.sigmoid(logits).cpu().numpy().flatten()
            
        y_true = Y_test.flatten()
        y_pred = (y_prob >= threshold).astype(int)
        
        file_suffix = f"_{suffix}" if suffix else ""
        
        pos_weight_val = None
        if hasattr(model, 'pos_weight') and model.pos_weight is not None:
            pos_weight_val = float(model.pos_weight.item())
        
        logger.info(f"📝 Saving CSV metrics to {self.summary.output_dir}...")
        self.summary.save_metrics(
            y_true, y_prob,
            threshold=threshold,
            pos_weight=pos_weight_val,
            filename=f"test_metrics{file_suffix}.csv"
        )

        logger.info("🎯 Computing operating points (Tight/Medium/Loose)...")
        operating_points = self.summary.save_operating_points(
            y_true, y_prob,
            filename=f"operating_points{file_suffix}.csv"
        )
        for point in operating_points:
            logger.info(
                f"   {point['Operating_Point']:<7} PD={point['PD']:.4f} (target {point['Target_PD']:.2f}) "
                f"-> FA={point['FA']:.4f}, SP={point['SP_Index']:.4f}, threshold={point['Threshold']:.4f}"
            )

        logger.info(f"🖼️ Saving evaluation plots to {self.monitor.output_dir}...")
        self.monitor.plot_roc_curve(y_true, y_prob, filename=f"roc_curve{file_suffix}.pdf", operating_points=operating_points)
        self.monitor.plot_pr_curve(y_true, y_prob, filename=f"pr_curve{file_suffix}.pdf")
        self.monitor.plot_confusion_matrix(y_true, y_pred, filename=f"confusion_matrix{file_suffix}.pdf")
        
        if loss_callback is not None:
            self.monitor.plot_loss(loss_callback.train_loss, loss_callback.val_loss, filename=f"loss_curve{file_suffix}.pdf")
            
        logger.info(f"✅ Evaluation complete{suffix_print}!")

    def run(
        self,
        use_kfold: bool = False,
        n_splits: int = 5,
        test_size: float = 0.15,
        learning_rate: float = 0.001,
        threshold: float = 0.5,
        target_fold: Optional[int] = None
    ) -> Optional[Union[Tuple[ModelMLP, Any], Tuple[List[Any], List[ModelMLP]]]]:
        """
        Executes end-to-end pipeline (loading, preprocessing, split, training with weighted loss, evaluation).

        Args:
            use_kfold (bool): Whether to use K-Fold cross-validation. Defaults to False.
            n_splits (int): Number of K-Fold splits. Defaults to 5.
            test_size (float): Holdout test dataset ratio. Defaults to 0.15.
            learning_rate (float): Model learning rate. Defaults to 0.001.
            threshold (float): Decision threshold. Defaults to 0.5.
            target_fold (Optional[int]): Target fold index for isolated fold run. Defaults to None.

        Returns:
            Optional[Union[Tuple[ModelMLP, Any], Tuple[List[Any], List[ModelMLP]]]]: Model and trainer objects or None if data loading fails.
        """
        logger.info(f"🚀 Starting Pipeline: {self.model_name}")

        logger.info("📂 Step 1: Loading dataset...")
        df = self.loader.execute()

        if df is None or df.empty:
            logger.error("❌ No data was loaded.")
            return None

        logger.info("🏷️ Step 1.5: Generating labels...")
        df = LabelGenerator.apply_label(df, file_path_col='file_path', label_col=self.label_col)
        df.drop(columns=['file_path'], inplace=True)

        logger.info("⚙️ Step 2: Preprocessing features...")
        X = self.preprocessor.transform(df)
        Y = self.preprocessor.get_labels(df, label_col=self.label_col)

        if Y is None:
            logger.error("❌ Labels column not found.")
            return None

        logger.info(f"✂️ Splitting {test_size * 100}% data for holdout testing...")
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42, shuffle=True)

        input_dim = X.shape[1]
        logger.info(f"🏋️ Step 3: Training (input_dim={input_dim}, K-Fold={use_kfold}, Weighted Loss Enabled)...")

        if use_kfold:
            model_kwargs = {'learning_rate': learning_rate, 'input_dim': input_dim}
            fold_trainers, fold_models, fold_loss_callbacks = self.trainer.fit_kfold(
                ModelMLP, model_kwargs, X_train, Y_train,
                n_splits=n_splits, target_fold=target_fold
            )
            logger.info("📢 Generating visual evaluation per trained fold...")
            for i, (model, loss_callback) in enumerate(zip(fold_models, fold_loss_callbacks)):
                fold_idx = target_fold if target_fold is not None else (i + 1)
                self.evaluate_model(model, X_test, Y_test, threshold=threshold, suffix=f"fold_{fold_idx}", loss_callback=loss_callback)
            return fold_trainers, fold_models
        else:
            model = ModelMLP(learning_rate=learning_rate, input_dim=input_dim)
            trained_trainer, loss_callback = self.trainer.fit(model, X_train, Y_train)

            self.evaluate_model(model, X_test, Y_test, threshold=threshold, loss_callback=loss_callback)
            return model, trained_trainer

