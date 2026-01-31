"""
Command-line interface for StrepSuis-GenPhenNet
"""

import argparse
import logging
import sys

from . import __version__
from .analyzer import NetworkAnalyzer
from .config import Config


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="StrepSuis-GenPhenNet: Network-Based Integration of Genome-Phenome Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    parser.add_argument(
        "--data-dir", type=str, required=True, help="Directory containing input CSV files"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output directory for results (default: ./output)",
    )

    parser.add_argument(
        "--bootstrap", type=int, default=500, help="Number of bootstrap iterations (default: 500)"
    )

    parser.add_argument(
        "--fdr-alpha", type=float, default=0.05, help="FDR correction alpha level (default: 0.05)"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        config = Config(
            data_dir=args.data_dir,
            output_dir=args.output,
            bootstrap_iterations=args.bootstrap,
            fdr_alpha=args.fdr_alpha,
        )

        logger.info(f"StrepSuis-GenPhenNet v{__version__}")
        logger.info(f"Data directory: {config.data_dir}")
        logger.info(f"Output directory: {config.output_dir}")

        analyzer = NetworkAnalyzer(config)
        logger.info("Starting analysis...")

        _ = analyzer.run()

        logger.info("Analysis completed successfully!")
        logger.info(f"Results saved to: {config.output_dir}")

        return 0

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
