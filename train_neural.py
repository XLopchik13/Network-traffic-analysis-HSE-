"""Training script for the FCN salary model with MLflow experiment tracking.

Trains a fully-connected neural network on the preprocessed hh.ru feature
arrays, logs all hyperparameters and metrics to a remote MLflow server, and
saves the model weights locally.

Usage:
    python train_neural.py data/x_data.npy data/y_data.npy
"""

import logging
import sys
from pathlib import Path
from typing import Tuple

# Ensure UTF-8 output so MLflow progress indicators (which contain emoji)
# do not raise encoding errors on Windows terminals.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import mlflow
import mlflow.pytorch
import numpy as np

from src.model.constants import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    PYTORCH_MODEL_FILENAME,
    RESOURCES_DIR_NAME,
)
from src.model.neural_trainer import NeuralTrainer


def setup_logging() -> logging.Logger:
    """Configure INFO-level logging to stdout.

    Returns:
        Configured logger instance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def load_arrays(
    x_path: Path, y_path: Path
) -> Tuple[np.ndarray, np.ndarray]:
    """Load feature and target arrays from .npy files.

    Args:
        x_path: Path to the feature array (.npy) file.
        y_path: Path to the target salary array (.npy) file.

    Returns:
        Tuple of (x_data, y_data) numpy arrays.

    Raises:
        FileNotFoundError: If either file does not exist.
        ValueError: If the loaded arrays have unexpected dimensionality.
    """
    if not x_path.exists():
        raise FileNotFoundError(f"Feature file not found: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Target file not found: {y_path}")

    x_data: np.ndarray = np.load(x_path)
    y_data: np.ndarray = np.load(y_path)

    if x_data.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got {x_data.shape}")
    if y_data.ndim != 1:
        raise ValueError(f"Expected 1D target array, got {y_data.shape}")

    return x_data, y_data


def configure_mlflow() -> None:
    """Point MLflow at the remote tracking server and set the experiment.

    Raises:
        mlflow.exceptions.MlflowException: If the server is unreachable.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def run_experiment(
    x_data: np.ndarray,
    y_data: np.ndarray,
    output_path: Path,
    logger: logging.Logger,
) -> str:
    """Train the model inside an MLflow run and return the run ID.

    Args:
        x_data: Feature array of shape (n_samples, n_features).
        y_data: Raw salary array of shape (n_samples,).
        output_path: Local destination for saved model weights.
        logger: Logger instance for progress messages.

    Returns:
        MLflow run ID of the completed run.
    """
    trainer = NeuralTrainer()

    with mlflow.start_run(run_name=MLFLOW_MODEL_NAME) as run:
        logger.info("MLflow run started: %s", run.info.run_id)

        metrics = trainer.train(x_data, y_data)

        mlflow.log_params(metrics["hyperparams"])
        mlflow.log_metric("mae_test", metrics["mae"])
        mlflow.log_metric("r2_score_test", metrics["r2"])
        mlflow.log_metric("best_val_loss", metrics["best_val_loss"])
        mlflow.log_metric("epochs_trained", metrics["epochs_trained"])

        mlflow.pytorch.log_model(
            trainer.get_model(),
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME,
        )

        logger.info(
            "Logged to MLflow — MAE: %.2f, R2: %.4f",
            metrics["mae"],
            metrics["r2"],
        )

        trainer.save(output_path)
        return run.info.run_id


def main() -> None:
    """Parse arguments, train the model, and log results to MLflow.

    Raises:
        SystemExit: If arguments are invalid or training fails.
    """
    logger = setup_logging()

    if len(sys.argv) != 3:
        logger.error("Invalid arguments")
        print("Usage: python train_neural.py path/to/x_data.npy path/to/y_data.npy")
        sys.exit(1)

    x_path = Path(sys.argv[1])
    y_path = Path(sys.argv[2])
    output_path = Path(RESOURCES_DIR_NAME) / PYTORCH_MODEL_FILENAME

    try:
        logger.info("Loading training data")
        x_data, y_data = load_arrays(x_path, y_path)
        logger.info("Loaded x=%s, y=%s", x_data.shape, y_data.shape)

        configure_mlflow()

        run_id = run_experiment(x_data, y_data, output_path, logger)
        logger.info("Done. RUN_ID: %s", run_id)
        print(f"\nRUN_ID: {run_id}")

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Invalid data: %s", e)
        sys.exit(1)
    except Exception as e:  # noqa: BLE001 — catch-all for MLflow/network errors
        logger.error("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
