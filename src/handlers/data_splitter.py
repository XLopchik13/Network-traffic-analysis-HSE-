"""Data splitting handler."""

from typing import Optional
import pandas as pd
import numpy as np
import logging

from ..core.handler import Handler
from ..core.pipeline_context import PipelineContext


class DataSplitterHandler(Handler):
    """Handler for splitting data into features and target.
    
    This handler separates the dataset into X (features) and y (target) arrays.
    
    Attributes:
        target_column: Name of the target column.
        feature_columns: List of feature column names. If None, uses all except target.
    """
    
    def __init__(
        self,
        target_column: str,
        feature_columns: Optional[list[str]] = None,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """Initialize the data splitter handler.
        
        Args:
            target_column: Name of the target column.
            feature_columns: Feature column names. If None, uses all except target.
            logger: Logger instance. If None, creates a new logger.
        """
        super().__init__(logger)
        self.target_column: str = target_column
        self.feature_columns: Optional[list[str]] = feature_columns
        
    def process(self, context: PipelineContext) -> PipelineContext:
        """Split data into features and target.
        
        Args:
            context: The pipeline context with data to split.
            
        Returns:
            Context with X and y data set.
            
        Raises:
            ValueError: If data is None or target column not found.
        """
        if context.data is None:
            error_msg = "No data to split"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        data = context.data

        if self.target_column not in data.columns:
            error_msg = f"Target column '{self.target_column}' not found in data"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if self.feature_columns is None:
            self.feature_columns = [
                col for col in data.columns if col != self.target_column
            ]
            
        self.logger.info(f"Target column: {self.target_column}")
        self.logger.info(f"Feature columns ({len(self.feature_columns)}): {self.feature_columns}")

        x_data = data[self.feature_columns].values
        y_data = data[self.target_column].values
        
        self.logger.info(f"X shape: {x_data.shape}, y shape: {y_data.shape}")
        
        context.set_x_data(x_data)
        context.set_y_data(y_data)
        context.add_metadata("target_column", self.target_column)
        context.add_metadata("feature_columns", self.feature_columns)
        
        return context
