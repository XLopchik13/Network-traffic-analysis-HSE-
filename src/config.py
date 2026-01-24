"""Configuration module for pipeline settings."""

from pathlib import Path
from typing import Dict, Any


PIPELINE_CONFIG: Dict[str, Any] = {
    "target_column": "Label",
    "drop_duplicates": True,
    "handle_missing": "drop",
    "fill_value": 0.0,
    "normalization_method": "standard",
}

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

X_DATA_FILENAME = "x_data.npy"
Y_DATA_FILENAME = "y_data.npy"
