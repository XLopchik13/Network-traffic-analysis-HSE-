"""Feature engineering handler."""

from typing import Optional, List
import pandas as pd
import numpy as np
import logging

from ..core.handler import Handler
from ..core.pipeline_context import PipelineContext


class FeatureEngineeringHandler(Handler):
    """Handler for feature engineering.
    
    This handler performs feature engineering operations such as:
    - Encoding categorical variables
    - Creating new features
    - Transforming existing features
    
    Attributes:
        categorical_columns: List of categorical column names to encode.
        numeric_columns: List of numeric column names to keep.
    """
    
    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """Initialize the feature engineering handler.
        
        Args:
            categorical_columns: Categorical columns to encode. Defaults to None.
            numeric_columns: Numeric columns to keep. Defaults to None.
            logger: Logger instance. If None, creates a new logger.
        """
        super().__init__(logger)
        self.categorical_columns: Optional[List[str]] = categorical_columns
        self.numeric_columns: Optional[List[str]] = numeric_columns
        
    def process(self, context: PipelineContext) -> PipelineContext:
        """Perform feature engineering.
        
        Args:
            context: The pipeline context with data to engineer.
            
        Returns:
            Context with engineered features.
            
        Raises:
            ValueError: If data is None.
        """
        if context.data is None:
            error_msg = "No data for feature engineering"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        data = context.data.copy()

        if self.categorical_columns is None and self.numeric_columns is None:
            self.logger.info("Auto-detecting column types")
            self.numeric_columns = list(data.select_dtypes(
                include=[np.number]
            ).columns)
            self.categorical_columns = list(data.select_dtypes(
                include=['object', 'category']
            ).columns)
            
        self.logger.info(
            f"Numeric columns ({len(self.numeric_columns or [])}): "
            f"{self.numeric_columns}"
        )
        self.logger.info(
            f"Categorical columns ({len(self.categorical_columns or [])}): "
            f"{self.categorical_columns}"
        )

        if self.categorical_columns:
            for col in self.categorical_columns:
                if col in data.columns:
                    self.logger.debug(f"Encoding categorical column: {col}")
                    data[col] = pd.Categorical(data[col]).codes
                    
        context.update_data(data)
        context.add_metadata("numeric_columns", self.numeric_columns)
        context.add_metadata("categorical_columns", self.categorical_columns)
        
        return context
