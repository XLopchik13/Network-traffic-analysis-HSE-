"""Data loading handler."""

from pathlib import Path
from typing import Optional
import pandas as pd
import logging

from ..core.handler import Handler
from ..core.pipeline_context import PipelineContext


class DataLoaderHandler(Handler):
    """Handler for loading data from CSV files.
    
    This handler reads CSV files and loads them into the pipeline context.
    
    Attributes:
        file_path: Path to the CSV file to load.
    """
    
    def __init__(
        self,
        file_path: Path,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """Initialize the data loader handler.
        
        Args:
            file_path: Path to the CSV file.
            logger: Logger instance. If None, creates a new logger.
        """
        super().__init__(logger)
        self.file_path: Path = file_path
        
    def process(self, context: PipelineContext) -> PipelineContext:
        """Load data from CSV file.
        
        Args:
            context: The pipeline context.
            
        Returns:
            Context with loaded data.
            
        Raises:
            FileNotFoundError: If the CSV file doesn't exist.
            pd.errors.ParserError: If the CSV file is malformed.
        """
        if not self.file_path.exists():
            error_msg = f"File not found: {self.file_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        self.logger.info(f"Loading data from {self.file_path}")
        
        try:
            data = pd.read_csv(self.file_path)
            self.logger.info(f"Loaded data with shape: {data.shape}")
            self.logger.debug(f"Columns: {list(data.columns)}")
            
            context.update_data(data)
            context.add_metadata("original_shape", data.shape)
            context.add_metadata("original_columns", list(data.columns))
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
        return context
