"""Data cleaning handler."""

from typing import Optional, List
import pandas as pd
import logging

from ..core.handler import Handler
from ..core.pipeline_context import PipelineContext


class DataCleanerHandler(Handler):
    """Handler for cleaning data.
    
    This handler performs data cleaning operations such as:
    - Removing duplicates
    - Handling missing values
    - Filtering invalid data
    
    Attributes:
        drop_duplicates: Whether to drop duplicate rows.
        handle_missing: Strategy for handling missing values ('drop', 'fill', 'none').
        fill_value: Value to use when filling missing values.
    """
    
    def __init__(
        self,
        drop_duplicates: bool = True,
        handle_missing: str = 'drop',
        fill_value: float = 0.0,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """Initialize the data cleaner handler.
        
        Args:
            drop_duplicates: Whether to drop duplicate rows. Defaults to True.
            handle_missing: Strategy for missing values ('drop', 'fill', 'none'). 
                Defaults to 'drop'.
            fill_value: Value for filling missing data. Defaults to 0.0.
            logger: Logger instance. If None, creates a new logger.
        """
        super().__init__(logger)
        self.drop_duplicates: bool = drop_duplicates
        self.handle_missing: str = handle_missing
        self.fill_value: float = fill_value
        
    def process(self, context: PipelineContext) -> PipelineContext:
        """Clean the data.
        
        Args:
            context: The pipeline context with data to clean.
            
        Returns:
            Context with cleaned data.
            
        Raises:
            ValueError: If data is None or handle_missing has invalid value.
        """
        if context.data is None:
            error_msg = "No data to clean"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        data = context.data.copy()
        original_shape = data.shape

        if self.drop_duplicates:
            duplicates_count = data.duplicated().sum()
            if duplicates_count > 0:
                self.logger.info(f"Dropping {duplicates_count} duplicate rows")
                data = data.drop_duplicates()

        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            self.logger.info(f"Found {missing_count} missing values")
            
            if self.handle_missing == 'drop':
                self.logger.info("Dropping rows with missing values")
                data = data.dropna()
            elif self.handle_missing == 'fill':
                self.logger.info(f"Filling missing values with {self.fill_value}")
                data = data.fillna(self.fill_value)
            elif self.handle_missing == 'none':
                self.logger.info("Keeping missing values as is")
            else:
                error_msg = f"Invalid handle_missing value: {self.handle_missing}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
                
        self.logger.info(
            f"Data cleaning complete. Shape: {original_shape} -> {data.shape}"
        )
        
        context.update_data(data)
        context.add_metadata("cleaned_shape", data.shape)
        context.add_metadata("rows_removed", original_shape[0] - data.shape[0])
        
        return context
