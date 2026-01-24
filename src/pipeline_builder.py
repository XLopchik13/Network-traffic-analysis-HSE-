"""Pipeline builder for creating data processing chains."""

from pathlib import Path
from typing import Optional
import logging

from .core.handler import Handler
from .core.pipeline_context import PipelineContext
from .handlers.data_loader import DataLoaderHandler
from .handlers.data_cleaner import DataCleanerHandler
from .handlers.feature_engineering import FeatureEngineeringHandler
from .handlers.advanced_feature_extractor import AdvancedFeatureExtractorHandler
from .handlers.data_splitter import DataSplitterHandler
from .handlers.data_normalizer import DataNormalizerHandler
from .handlers.data_exporter import DataExporterHandler


class PipelineBuilder:
    """Builder for creating data processing pipelines.
    
    This class provides a fluent interface for building processing chains
    using the Chain of Responsibility pattern.
    
    Attributes:
        logger: Logger instance for the pipeline.
        first_handler: The first handler in the chain.
        last_handler: The last handler in the chain (for adding new handlers).
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """Initialize the pipeline builder.
        
        Args:
            logger: Logger instance. If None, creates a new logger.
        """
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        self.first_handler: Optional[Handler] = None
        self.last_handler: Optional[Handler] = None
        
    def add_handler(self, handler: Handler) -> 'PipelineBuilder':
        """Add a handler to the pipeline chain.
        
        Args:
            handler: The handler to add.
            
        Returns:
            Self for method chaining.
        """
        if self.first_handler is None:
            self.first_handler = handler
            self.last_handler = handler
        else:
            if self.last_handler is not None:
                self.last_handler.set_next(handler)
            self.last_handler = handler
            
        return self
        
    def build(self) -> Handler:
        """Build and return the pipeline.
        
        Returns:
            The first handler in the chain.
            
        Raises:
            ValueError: If no handlers were added.
        """
        if self.first_handler is None:
            error_msg = "Cannot build pipeline: no handlers added"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        return self.first_handler
        
    @staticmethod
    def create_default_pipeline(
        input_file: Path,
        output_dir: Path,
        target_column: str = 'Label',
        logger: Optional[logging.Logger] = None
    ) -> Handler:
        """Create a default data processing pipeline.
        
        Args:
            input_file: Path to input CSV file.
            output_dir: Directory for output files.
            target_column: Name of the target column. Defaults to 'Label'.
            logger: Logger instance. If None, creates a new logger.
            
        Returns:
            The first handler of the configured pipeline.
        """
        builder = PipelineBuilder(logger)
        
        pipeline = (
            builder
            .add_handler(DataLoaderHandler(input_file, logger))
            .add_handler(AdvancedFeatureExtractorHandler(logger=logger))
            .add_handler(DataCleanerHandler(
                drop_duplicates=True,
                handle_missing='drop',
                logger=logger
            ))
            .add_handler(DataSplitterHandler(target_column, logger=logger))
            .add_handler(DataNormalizerHandler(method='standard', logger=logger))
            .add_handler(DataExporterHandler(output_dir, logger=logger))
            .build()
        )
        
        return pipeline
