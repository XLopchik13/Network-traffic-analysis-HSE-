"""Pipeline entry point for preprocessing hh.ru resume data.

Reads a CSV file, runs the full processing pipeline, and writes
x_data.npy and y_data.npy to the same directory as the input file.

Usage:
    python run_pipeline.py path/to/hh.csv
"""

import sys
import logging
from pathlib import Path

from src.pipeline_builder import PipelineBuilder
from src.core.pipeline_context import PipelineContext


def setup_logging() -> logging.Logger:
    """Configure and return the application logger.

    Returns:
        Configured logger instance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def main() -> None:
    """Run the data processing pipeline on a CSV input file.

    Raises:
        SystemExit: If arguments are invalid or processing fails.
    """
    logger = setup_logging()

    if len(sys.argv) != 2:
        logger.error("Invalid arguments")
        print("Usage: python run_pipeline.py path/to/hh.csv")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    if input_path.suffix.lower() != ".csv":
        logger.error("Input file must be a CSV file: %s", input_path)
        sys.exit(1)

    output_dir = input_path.parent

    logger.info("Starting data processing pipeline")
    logger.info("Input file: %s", input_path)
    logger.info("Output directory: %s", output_dir)

    try:
        pipeline = PipelineBuilder.create_default_pipeline(
            input_file=input_path,
            output_dir=output_dir,
            target_column="salary",
            logger=logger,
        )

        context = PipelineContext(logger=logger)
        result_context = pipeline.handle(context)

        logger.info("Pipeline completed successfully")
        logger.info("X data saved to: %s", result_context.get_metadata("x_data_path"))
        logger.info("Y data saved to: %s", result_context.get_metadata("y_data_path"))
        logger.info(
            "X shape: %s",
            result_context.x_data.shape if result_context.x_data is not None else "N/A",
        )
        logger.info(
            "Y shape: %s",
            result_context.y_data.shape if result_context.y_data is not None else "N/A",
        )

    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
