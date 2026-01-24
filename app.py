"""Main application entry point."""

import sys
import logging
from pathlib import Path
from typing import NoReturn

from src.pipeline_builder import PipelineBuilder
from src.core.pipeline_context import PipelineContext


def setup_logging() -> logging.Logger:
    """Configure and return the application logger.
    
    Returns:
        Configured logger instance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def main() -> None:
    """Main application function.
    
    Processes command line arguments and runs the data processing pipeline.
    Creates x_data.npy and y_data.npy in the same directory as the input file.
    
    Usage:
        python app.py path/to/hh.csv
    
    Raises:
        SystemExit: If arguments are invalid or processing fails.
    """
    logger = setup_logging()

    if len(sys.argv) != 2:
        logger.error("Invalid arguments")
        print("Usage: python app.py path/to/hh.csv")
        sys.exit(1)
        
    input_path = Path(sys.argv[1])

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
        
    if not input_path.suffix.lower() == '.csv':
        logger.error(f"Input file must be a CSV file: {input_path}")
        sys.exit(1)

    output_dir = input_path.parent

    logger.info("Starting Network Traffic Analysis Pipeline")
    logger.info(f"Input file: {input_path}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        pipeline = PipelineBuilder.create_default_pipeline(
            input_file=input_path,
            output_dir=output_dir,
            target_column='salary',
            logger=logger
        )

        context = PipelineContext(logger=logger)
        result_context = pipeline.handle(context)

        logger.info("Pipeline completed successfully")
        logger.info(f"X data saved to: {result_context.get_metadata('x_data_path')}")
        logger.info(f"Y data saved to: {result_context.get_metadata('y_data_path')}")
        logger.info(f"X shape: {result_context.x_data.shape if result_context.x_data is not None else 'N/A'}")
        logger.info(f"Y shape: {result_context.y_data.shape if result_context.y_data is not None else 'N/A'}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
