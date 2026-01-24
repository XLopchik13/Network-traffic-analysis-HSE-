"""Data normalization handler."""

from typing import Optional
import numpy as np
import logging

from ..core.handler import Handler
from ..core.pipeline_context import PipelineContext


class DataNormalizerHandler(Handler):
    """Handler for normalizing data.
    
    This handler normalizes feature data using various strategies.
    
    Attributes:
        method: Normalization method ('standard', 'minmax', 'none').
    """
    
    def __init__(
        self,
        method: str = 'standard',
        logger: Optional[logging.Logger] = None
    ) -> None:
        """Initialize the data normalizer handler.
        
        Args:
            method: Normalization method ('standard', 'minmax', 'none'). 
                Defaults to 'standard'.
            logger: Logger instance. If None, creates a new logger.
        """
        super().__init__(logger)
        self.method: str = method
        
    def process(self, context: PipelineContext) -> PipelineContext:
        """Normalize the feature data.
        
        Args:
            context: The pipeline context with data to normalize.
            
        Returns:
            Context with normalized X data.
            
        Raises:
            ValueError: If X data is None or method is invalid.
        """
        if context.x_data is None:
            error_msg = "No X data to normalize"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        x_data = context.x_data.astype(np.float64)
        
        if self.method == 'standard':
            self.logger.info("Applying standard normalization (z-score)")
            mean = np.mean(x_data, axis=0)
            std = np.std(x_data, axis=0)
            std[std == 0] = 1.0
            x_data = (x_data - mean) / std
            context.add_metadata("normalization_mean", mean)
            context.add_metadata("normalization_std", std)
            
        elif self.method == 'minmax':
            self.logger.info("Applying min-max normalization")
            min_val = np.min(x_data, axis=0)
            max_val = np.max(x_data, axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1.0
            x_data = (x_data - min_val) / range_val
            context.add_metadata("normalization_min", min_val)
            context.add_metadata("normalization_max", max_val)
            
        elif self.method == 'none':
            self.logger.info("No normalization applied")
            
        else:
            error_msg = f"Invalid normalization method: {self.method}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        context.set_x_data(x_data)
        context.add_metadata("normalization_method", self.method)
        
        return context
