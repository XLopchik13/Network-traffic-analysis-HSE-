"""Classifier for predicting IT developer seniority level."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DEFAULT_RANDOM_STATE = 42
DEFAULT_N_ESTIMATORS = 200
DEFAULT_TEST_SIZE = 0.2

logger = logging.getLogger(__name__)


class DeveloperClassifier:
    """Random Forest classifier for junior / middle / senior prediction.

    Class weights are balanced automatically to handle the skewed
    distribution across seniority levels.

    Attributes:
        n_estimators: Number of trees in the forest.
        test_size: Fraction of data held out for evaluation.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        test_size: float = DEFAULT_TEST_SIZE,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        """Configure the classifier.

        Args:
            n_estimators: Number of decision trees to build.
            test_size: Proportion of samples reserved for evaluation.
            random_state: Seed value for reproducible results.
        """
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        self._encoder = LabelEncoder()
        self._test_size = test_size
        self._random_state = random_state
        self._feature_names: List[str] = []

    def train(
        self, x_data: np.ndarray, y_labels: "list[str]", feature_names: List[str]
    ) -> Dict[str, Any]:
        """Encode labels, split data, fit the model, and return metrics.

        Args:
            x_data: Feature matrix of shape (n_samples, n_features).
            y_labels: List of string seniority labels for each sample.
            feature_names: Names of the feature columns (for importance reporting).

        Returns:
            Dictionary with 'report' (classification report string) and
            'feature_importances' (array aligned with feature_names).
        """
        self._feature_names = feature_names
        y_encoded = self._encoder.fit_transform(y_labels)

        x_train, x_test, y_train, y_test = train_test_split(
            x_data,
            y_encoded,
            test_size=self._test_size,
            random_state=self._random_state,
            stratify=y_encoded,
        )

        logger.info("Training on %d samples, evaluating on %d", len(x_train), len(x_test))
        self._model.fit(x_train, y_train)

        y_pred = self._model.predict(x_test)
        report = classification_report(
            y_test,
            y_pred,
            target_names=self._encoder.classes_,
        )
        logger.info("Classification report:\n%s", report)

        return {
            "report": report,
            "feature_importances": self._model.feature_importances_,
        }

    def get_feature_names(self) -> List[str]:
        """Return the feature names recorded during training.

        Returns:
            List of feature column names.
        """
        return self._feature_names

    def save(self, output_path: Path) -> None:
        """Serialize the trained model and label encoder to disk.

        Args:
            output_path: Destination file path for the serialized bundle.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "encoder": self._encoder}, output_path)
        logger.info("Classifier saved to %s", output_path)
