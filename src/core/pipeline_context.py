"""Pipeline context for data processing chain."""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
import logging


class PipelineContext:
    """Context object that passes through the chain of responsibility.
    
    This class holds the data and metadata as it moves through the processing pipeline.
    Each handler in the chain can read from and write to this context.
    
    Attributes:
        data: DataFrame containing the raw or processed data.
        x_data: NumPy array for features (populated during processing).
        y_data: NumPy array for target (populated during processing).
        metadata: Dictionary for storing additional processing information.
        logger: Logger instance for tracking processing steps.
    """
    
    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """Initialize pipeline context.
        
        Args:
            data: Initial DataFrame to process. Defaults to None.
            logger: Logger instance for logging. If None, creates a new logger.
        """
        self.data: Optional[pd.DataFrame] = data
        self.x_data: Optional[np.ndarray] = None
        self.y_data: Optional[np.ndarray] = None
        self.metadata: Dict[str, Any] = {}
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        
    def update_data(self, data: pd.DataFrame) -> None:
        """Update the data in the context.
        
        Args:
            data: New DataFrame to set as current data.
        """
        self.data = data
        self.logger.debug(f"Data updated. Shape: {data.shape}")
        
    def set_x_data(self, x_data: np.ndarray) -> None:
        """Set the feature data.
        
        Args:
            x_data: NumPy array containing features.
        """
        self.x_data = x_data
        self.logger.debug(f"X data set. Shape: {x_data.shape}")
        
    def set_y_data(self, y_data: np.ndarray) -> None:
        """Set the target data.
        
        Args:
            y_data: NumPy array containing target values.
        """
        self.y_data = y_data
        self.logger.debug(f"Y data set. Shape: {y_data.shape}")
        
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the context.
        
        Args:
            key: Metadata key.
            value: Metadata value.
        """
        self.metadata[key] = value
        self.logger.debug(f"Metadata added: {key} = {value}")
        
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata from the context.
        
        Args:
            key: Metadata key.
            default: Default value if key not found. Defaults to None.
            
        Returns:
            Metadata value or default.
        """
        return self.metadata.get(key, default)
