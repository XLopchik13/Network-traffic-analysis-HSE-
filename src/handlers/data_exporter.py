"""Data exporter handler."""

from pathlib import Path
from typing import Optional
import numpy as np
import logging

from ..core.handler import Handler
from ..core.pipeline_context import PipelineContext


class DataExporterHandler(Handler):
    """Handler for exporting processed data to files.
    
    This handler saves the processed X and y data as NumPy .npy files.
    
    Attributes:
        output_dir: Directory where output files will be saved.
        x_filename: Filename for X data. Defaults to 'x_data.npy'.
        y_filename: Filename for y data. Defaults to 'y_data.npy'.
    """
    
    def __init__(
        self,
        output_dir: Path,
        x_filename: str = 'x_data.npy',
        y_filename: str = 'y_data.npy',
        logger: Optional[logging.Logger] = None
    ) -> None:
        """Initialize the data exporter handler.
        
        Args:
            output_dir: Directory for output files.
            x_filename: Filename for X data. Defaults to 'x_data.npy'.
            y_filename: Filename for y data. Defaults to 'y_data.npy'.
            logger: Logger instance. If None, creates a new logger.
        """
        super().__init__(logger)
        self.output_dir: Path = output_dir
        self.x_filename: str = x_filename
        self.y_filename: str = y_filename
        
    def process(self, context: PipelineContext) -> PipelineContext:
        """Export X and y data to .npy files.
        
        Args:
            context: The pipeline context with data to export.
            
        Returns:
            Context with export paths added to metadata.
            
        Raises:
            ValueError: If X or y data is None.
        """
        if context.x_data is None:
            error_msg = "No X data to export"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        if context.y_data is None:
            error_msg = "No y data to export"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        x_path = self.output_dir / self.x_filename
        np.save(x_path, context.x_data)
        self.logger.info(f"Saved X data to {x_path}")

        y_path = self.output_dir / self.y_filename
        np.save(y_path, context.y_data)
        self.logger.info(f"Saved y data to {y_path}")
        
        context.add_metadata("x_data_path", str(x_path))
        context.add_metadata("y_data_path", str(y_path))
        
        return context
