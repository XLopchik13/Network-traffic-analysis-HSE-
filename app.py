"""Application entry point for salary prediction.

Loads a feature array produced by the preprocessing pipeline and outputs
predicted salaries using the trained model stored in resources/.

Usage:
    python app.py path/to/x_data.npy

Output:
    [float, float, ...]  — predicted salaries in rubles printed to stdout.
"""

import sys
import logging
from pathlib import Path

import numpy as np

from src.model.salary_predictor import SalaryPredictor
from src.model.constants import MODEL_FILENAME, RESOURCES_DIR_NAME


def setup_logging() -> logging.Logger:
    """Configure and return the application logger.

    Logs are written to stderr so they do not interfere with prediction output.

    Returns:
        Configured logger instance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    return logging.getLogger(__name__)


def load_features(x_path: Path) -> np.ndarray:
    """Load a feature array from a .npy file.

    Args:
        x_path: Path to the .npy feature file produced by the pipeline.

    Returns:
        2D numpy array of shape (n_samples, n_features).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not contain a 2D array.
    """
    if not x_path.exists():
        raise FileNotFoundError(f"Feature file not found: {x_path}")

    if x_path.suffix.lower() != ".npy":
        raise ValueError(f"Expected a .npy file, got: {x_path}")

    x_data: np.ndarray = np.load(x_path)

    if x_data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x_data.shape}")

    return x_data


def main() -> None:
    """Load features, run the salary predictor, and print results.

    Raises:
        SystemExit: If arguments are invalid or prediction fails.
    """
    logger = setup_logging()

    if len(sys.argv) != 2:
        logger.error("Invalid arguments")
        print("Usage: python app.py path/to/x_data.npy", file=sys.stderr)
        sys.exit(1)

    x_path = Path(sys.argv[1])
    model_path = Path(RESOURCES_DIR_NAME) / MODEL_FILENAME

    try:
        x_data = load_features(x_path)
        logger.info("Loaded feature array: %s", x_data.shape)

        predictor = SalaryPredictor(model_path)
        predictions = predictor.predict(x_data)

        print([round(float(p), 2) for p in predictions])

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Invalid data: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
