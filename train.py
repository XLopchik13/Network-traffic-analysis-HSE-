"""Training script for the salary prediction model.

Loads pre-built feature and target arrays, trains a GradientBoostingRegressor,
and saves the serialized model to resources/.

Usage:
    python train.py path/to/x_data.npy path/to/y_data.npy
"""

import sys
import logging
from pathlib import Path
from typing import Tuple

import numpy as np

from src.model.model_trainer import ModelTrainer
from src.model.constants import MODEL_FILENAME, RESOURCES_DIR_NAME


def setup_logging() -> logging.Logger:
    """Configure and return the root logger.

    Returns:
        Configured logger instance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def load_arrays(x_path: Path, y_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load feature and target arrays from .npy files.

    Args:
        x_path: Path to the feature array (.npy) file.
        y_path: Path to the target array (.npy) file.

    Returns:
        Tuple of (x_data, y_data) numpy arrays.

    Raises:
        FileNotFoundError: If either file does not exist.
        ValueError: If the arrays have unexpected dimensionality.
    """
    if not x_path.exists():
        raise FileNotFoundError(f"Feature file not found: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Target file not found: {y_path}")

    x_data: np.ndarray = np.load(x_path)
    y_data: np.ndarray = np.load(y_path)

    if x_data.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape {x_data.shape}")
    if y_data.ndim != 1:
        raise ValueError(f"Expected 1D target array, got shape {y_data.shape}")

    return x_data, y_data


def main() -> None:
    """Train the salary model and save weights to resources/.

    Raises:
        SystemExit: If arguments are invalid or training fails.
    """
    logger = setup_logging()

    if len(sys.argv) != 3:
        logger.error("Invalid arguments")
        print("Usage: python train.py path/to/x_data.npy path/to/y_data.npy")
        sys.exit(1)

    x_path = Path(sys.argv[1])
    y_path = Path(sys.argv[2])
    output_path = Path(RESOURCES_DIR_NAME) / MODEL_FILENAME

    try:
        logger.info("Loading training data")
        x_data, y_data = load_arrays(x_path, y_path)
        logger.info("Loaded x=%s, y=%s", x_data.shape, y_data.shape)

        trainer = ModelTrainer()
        metrics = trainer.train(x_data, y_data)

        logger.info(
            "Training complete — MAE: %.2f, R2: %.4f",
            metrics["mae"],
            metrics["r2"],
        )
        trainer.save(output_path)
        logger.info("Model saved to %s", output_path)

    except (FileNotFoundError, ValueError) as e:
        logger.error("Training failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
