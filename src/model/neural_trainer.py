"""Training loop for the FCN salary regression model."""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from src.model.constants import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    NN_BATCH_SIZE,
    NN_DROPOUT_RATE,
    NN_EPOCHS,
    NN_HIDDEN_DIMS,
    NN_LEARNING_RATE,
    NN_LR_FACTOR,
    NN_LR_PATIENCE,
    NN_PATIENCE,
    NN_WEIGHT_DECAY,
    SALARY_MAX,
    SALARY_MIN,
)
from src.model.fcn_model import FCNModel

logger = logging.getLogger(__name__)


class NeuralTrainer:
    """Manages the full training lifecycle of the FCN salary model.

    Handles outlier filtering, log-transform of the target, train/test
    split, mini-batch training with early stopping, and evaluation on
    the original salary scale.

    Attributes:
        epochs: Maximum number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Initial learning rate for Adam.
        weight_decay: L2 regularisation coefficient.
        test_size: Fraction of data reserved for evaluation.
        patience: Early-stopping patience in epochs.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        epochs: int = NN_EPOCHS,
        batch_size: int = NN_BATCH_SIZE,
        learning_rate: float = NN_LEARNING_RATE,
        weight_decay: float = NN_WEIGHT_DECAY,
        test_size: float = DEFAULT_TEST_SIZE,
        patience: int = NN_PATIENCE,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        """Configure the trainer.

        Args:
            epochs: Maximum number of training epochs.
            batch_size: Number of samples per gradient update.
            learning_rate: Adam initial learning rate.
            weight_decay: L2 penalty coefficient.
            test_size: Proportion of samples held out for evaluation.
            patience: Stop training after this many epochs without improvement.
            random_state: Seed for dataset splits and reproducibility.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_size = test_size
        self.patience = patience
        self.random_state = random_state
        self._model: FCNModel | None = None

    def train(
        self, x_data: np.ndarray, y_data: np.ndarray
    ) -> Dict[str, Any]:
        """Prepare data, train the model, and return evaluation metrics.

        Args:
            x_data: Feature array of shape (n_samples, n_features).
            y_data: Raw salary array of shape (n_samples,).

        Returns:
            Dictionary with keys: mae, r2, best_val_loss, epochs_trained,
            and hyperparams (sub-dict for logging).

        Raises:
            ValueError: If x_data and y_data have incompatible shapes.
        """
        if x_data.shape[0] != y_data.shape[0]:
            raise ValueError(
                f"Shape mismatch: x={x_data.shape[0]}, y={y_data.shape[0]}"
            )

        x_clean, y_clean = self._filter_outliers(x_data, y_data)
        y_log = np.log1p(y_clean).astype(np.float32)
        x_clean = x_clean.astype(np.float32)

        x_train, x_test, y_train, y_test, y_test_raw = self._split(
            x_clean, y_log, y_clean
        )

        input_dim = x_train.shape[1]
        self._model = FCNModel(input_dim, NN_HIDDEN_DIMS, NN_DROPOUT_RATE)
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer, factor=NN_LR_FACTOR, patience=NN_LR_PATIENCE
        )
        criterion = nn.MSELoss()
        loader = self._make_loader(x_train, y_train)

        epochs_trained, best_val_loss = self._run_epochs(
            loader, x_test, y_test, criterion, optimizer, scheduler
        )

        metrics = self._evaluate(x_test, y_test_raw)
        metrics["best_val_loss"] = best_val_loss
        metrics["epochs_trained"] = epochs_trained
        metrics["hyperparams"] = self._hyperparams(input_dim)

        logger.info(
            "Training done — MAE: %.2f, R2: %.4f, epochs: %d",
            metrics["mae"],
            metrics["r2"],
            epochs_trained,
        )
        return metrics

    def save(self, output_path: Path) -> None:
        """Save model weights to disk.

        Args:
            output_path: Destination .pt file path.

        Raises:
            RuntimeError: If called before training.
        """
        if self._model is None:
            raise RuntimeError("Model has not been trained yet.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), output_path)
        logger.info("Model weights saved to %s", output_path)

    def get_model(self) -> FCNModel:
        """Return the trained PyTorch model.

        Returns:
            Trained FCNModel instance.

        Raises:
            RuntimeError: If called before training.
        """
        if self._model is None:
            raise RuntimeError("Model has not been trained yet.")
        return self._model

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _filter_outliers(
        self, x_data: np.ndarray, y_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove rows whose salary is outside the valid range.

        Args:
            x_data: Full feature array.
            y_data: Full salary array.

        Returns:
            Filtered (x_data, y_data) tuple.
        """
        mask = (y_data >= SALARY_MIN) & (y_data <= SALARY_MAX)
        removed = int((~mask).sum())
        if removed > 0:
            logger.info(
                "Removed %d outlier rows (salary outside [%d, %d])",
                removed,
                SALARY_MIN,
                SALARY_MAX,
            )
        return x_data[mask], y_data[mask]

    def _split(
        self,
        x: np.ndarray,
        y_log: np.ndarray,
        y_raw: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split into train and test, returning log and raw test targets.

        Args:
            x: Feature array.
            y_log: Log-transformed salary array.
            y_raw: Original salary array (for final metric computation).

        Returns:
            Tuple of (x_train, x_test, y_train, y_test_log, y_test_raw).
        """
        indices = np.arange(len(x))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        return (
            x[train_idx],
            x[test_idx],
            y_log[train_idx],
            y_log[test_idx],
            y_raw[test_idx],
        )

    def _make_loader(
        self, x_train: np.ndarray, y_train: np.ndarray
    ) -> DataLoader:
        """Wrap training arrays in a shuffled DataLoader.

        Args:
            x_train: Training feature array.
            y_train: Training log-salary array.

        Returns:
            Configured DataLoader.
        """
        dataset = TensorDataset(
            torch.tensor(x_train),
            torch.tensor(y_train),
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _run_epochs(
        self,
        loader: DataLoader,
        x_val: np.ndarray,
        y_val: np.ndarray,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: ReduceLROnPlateau,
    ) -> Tuple[int, float]:
        """Run the training loop with early stopping.

        Args:
            loader: DataLoader for training batches.
            x_val: Validation feature array.
            y_val: Validation log-salary array.
            criterion: Loss function.
            optimizer: Gradient optimiser.
            scheduler: Learning-rate scheduler.

        Returns:
            Tuple of (epochs_trained, best_validation_loss).
        """
        best_val_loss = float("inf")
        patience_counter = 0

        x_val_t = torch.tensor(x_val)
        y_val_t = torch.tensor(y_val)

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(loader, criterion, optimizer)
            val_loss = self._val_loss(x_val_t, y_val_t, criterion)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                logger.info(
                    "Epoch %d/%d — train_loss: %.4f, val_loss: %.4f",
                    epoch,
                    self.epochs,
                    train_loss,
                    val_loss,
                )

            if patience_counter >= self.patience:
                logger.info("Early stopping at epoch %d", epoch)
                return epoch, best_val_loss

        return self.epochs, best_val_loss

    def _train_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Run one training epoch and return mean loss.

        Args:
            loader: DataLoader supplying mini-batches.
            criterion: Loss function.
            optimizer: Gradient optimiser.

        Returns:
            Mean training loss over all batches.
        """
        self._model.train()
        total_loss = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            predictions = self._model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _val_loss(
        self,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        criterion: nn.Module,
    ) -> float:
        """Compute validation loss without gradient tracking.

        Args:
            x_val: Validation feature tensor.
            y_val: Validation log-salary tensor.
            criterion: Loss function.

        Returns:
            Scalar validation loss.
        """
        self._model.eval()
        with torch.no_grad():
            predictions = self._model(x_val)
            return criterion(predictions, y_val).item()

    def _evaluate(
        self, x_test: np.ndarray, y_test_raw: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate the model on original salary scale.

        Args:
            x_test: Test feature array.
            y_test_raw: True salary values (not log-transformed).

        Returns:
            Dictionary with mae and r2 scores.
        """
        self._model.eval()
        with torch.no_grad():
            preds_log = self._model(torch.tensor(x_test)).numpy()
        preds = np.expm1(preds_log)
        return {
            "mae": float(mean_absolute_error(y_test_raw, preds)),
            "r2": float(r2_score(y_test_raw, preds)),
        }

    def _hyperparams(self, input_dim: int) -> Dict[str, Any]:
        """Collect hyperparameters for MLflow logging.

        Args:
            input_dim: Number of input features.

        Returns:
            Dictionary of hyperparameter names and values.
        """
        return {
            "model_type": "FCN",
            "input_dim": input_dim,
            "hidden_dims": str(NN_HIDDEN_DIMS),
            "dropout_rate": NN_DROPOUT_RATE,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs_max": self.epochs,
            "patience": self.patience,
            "log_target": True,
            "salary_min": SALARY_MIN,
            "salary_max": SALARY_MAX,
        }
