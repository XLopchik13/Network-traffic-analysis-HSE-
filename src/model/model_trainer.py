"""Model training module for salary prediction."""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import joblib
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from src.model.constants import (
    DEFAULT_N_ESTIMATORS,
    DEFAULT_MAX_DEPTH,
    DEFAULT_LEARNING_RATE,
    DEFAULT_TEST_SIZE,
    DEFAULT_RANDOM_STATE,
    SALARY_MIN,
    SALARY_MAX,
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and persists a salary regression model.

    Salary targets are log-transformed before fitting so that the model
    learns relative differences rather than absolute ones.  Outlier rows
    (salaries outside [SALARY_MIN, SALARY_MAX]) are removed before training.

    Attributes:
        n_estimators: Number of boosting stages.
        max_depth: Maximum tree depth.
        learning_rate: Shrinkage rate applied to each estimator.
        test_size: Fraction of data reserved for evaluation.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        max_depth: int = DEFAULT_MAX_DEPTH,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        test_size: float = DEFAULT_TEST_SIZE,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        """Configure the trainer.

        Args:
            n_estimators: Number of boosting stages to fit.
            max_depth: Maximum depth of individual regression estimators.
            learning_rate: Contribution of each tree to the final prediction.
            test_size: Proportion of samples used for evaluation.
            random_state: Seed value for reproducible splits and training.
        """
        base = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            random_state=random_state,
        )
        # TransformedTargetRegressor applies log1p before fit and expm1 after
        # predict, so predictions are always in the original ruble scale.
        self._model = TransformedTargetRegressor(
            regressor=base,
            func=np.log1p,
            inverse_func=np.expm1,
        )
        self._test_size = test_size
        self._random_state = random_state

    def train(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, Any]:
        """Filter outliers, train the model, and return evaluation metrics.

        Args:
            x_data: Feature array of shape (n_samples, n_features).
            y_data: Target salary array of shape (n_samples,).

        Returns:
            Dictionary with mae, r2, and n_samples_used after filtering.

        Raises:
            ValueError: If x_data and y_data have incompatible sample counts.
        """
        if x_data.shape[0] != y_data.shape[0]:
            raise ValueError(
                f"Sample count mismatch: x={x_data.shape[0]}, y={y_data.shape[0]}"
            )

        x_clean, y_clean = self._filter_outliers(x_data, y_data)

        x_train, x_test, y_train, y_test = train_test_split(
            x_clean,
            y_clean,
            test_size=self._test_size,
            random_state=self._random_state,
        )

        logger.info(
            "Training on %d samples, evaluating on %d",
            len(x_train),
            len(x_test),
        )
        self._model.fit(x_train, y_train)

        metrics = self._evaluate(x_test, y_test)
        metrics["n_samples_used"] = len(x_clean)
        logger.info(
            "Training complete — MAE: %.2f, R2: %.4f",
            metrics["mae"],
            metrics["r2"],
        )
        return metrics

    def _filter_outliers(
        self, x_data: np.ndarray, y_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove rows whose salary falls outside the valid range.

        Args:
            x_data: Full feature array.
            y_data: Full salary array.

        Returns:
            Filtered (x_data, y_data) tuple with outlier rows removed.
        """
        mask = (y_data >= SALARY_MIN) & (y_data <= SALARY_MAX)
        removed = int((~mask).sum())
        if removed > 0:
            logger.info("Removed %d outlier rows (salary outside [%d, %d])", removed, SALARY_MIN, SALARY_MAX)
        return x_data[mask], y_data[mask]

    def _evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Compute evaluation metrics on a held-out test set.

        Args:
            x_test: Feature array for evaluation.
            y_test: True salary values for evaluation.

        Returns:
            Dictionary with mae (mean absolute error) and r2 (R-squared) scores.
        """
        predictions = self._model.predict(x_test)
        return {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "r2": float(r2_score(y_test, predictions)),
        }

    def save(self, output_path: Path) -> None:
        """Serialize the trained model to disk.

        Args:
            output_path: Destination file path for the serialized model.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, output_path)
        logger.info("Model saved to %s", output_path)
