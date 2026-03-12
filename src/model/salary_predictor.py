"""Salary prediction module."""

import logging
from pathlib import Path

import numpy as np
import joblib

logger = logging.getLogger(__name__)


class SalaryPredictor:
    """Loads a trained model and generates salary predictions.

    Attributes:
        model_path: Path to the saved model file.
    """

    def __init__(self, model_path: Path) -> None:
        """Initialize the predictor by loading a saved model.

        Args:
            model_path: Filesystem path to the serialized model file.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info("Loading model from %s", model_path)
        self._model = joblib.load(model_path)
        logger.info("Model loaded successfully")

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        """Generate salary predictions for the given feature array.

        Args:
            x_data: 2D feature array of shape (n_samples, n_features).

        Returns:
            1D array of predicted salaries in rubles.

        Raises:
            ValueError: If x_data is not a 2D array.
        """
        if x_data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {x_data.shape}")

        logger.info("Predicting salaries for %d samples", x_data.shape[0])
        predictions = self._model.predict(x_data)
        logger.info("Prediction complete")
        return predictions
