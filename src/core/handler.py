"""Base handler class for Chain of Responsibility pattern."""

from abc import ABC, abstractmethod
from typing import Optional
import logging

from .pipeline_context import PipelineContext


class Handler(ABC):
    """Abstract base class for pipeline handlers.
    
    Implements the Chain of Responsibility pattern. Each handler processes
    the data and passes it to the next handler in the chain.
    
    Attributes:
        _next_handler: The next handler in the chain.
        logger: Logger instance for this handler.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """Initialize the handler.
        
        Args:
            logger: Logger instance. If None, creates a new logger.
        """
        self._next_handler: Optional[Handler] = None
        self.logger: logging.Logger = logger or logging.getLogger(self.__class__.__name__)
        
    def set_next(self, handler: 'Handler') -> 'Handler':
        """Set the next handler in the chain.
        
        Args:
            handler: The next handler to process the context.
            
        Returns:
            The handler that was set as next (for chaining).
        """
        self._next_handler = handler
        return handler
        
    def handle(self, context: PipelineContext) -> PipelineContext:
        """Handle the processing of the context.
        
        This method calls the process method and then passes the context
        to the next handler if one exists.
        
        Args:
            context: The pipeline context to process.
            
        Returns:
            The processed pipeline context.
        """
        self.logger.info(f"Starting {self.__class__.__name__}")
        context = self.process(context)
        self.logger.info(f"Completed {self.__class__.__name__}")
        
        if self._next_handler:
            return self._next_handler.handle(context)
        return context
        
    @abstractmethod
    def process(self, context: PipelineContext) -> PipelineContext:
        """Process the context.
        
        This method must be implemented by concrete handlers to define
        the specific processing logic.
        
        Args:
            context: The pipeline context to process.
            
        Returns:
            The processed pipeline context.
        """
        pass
