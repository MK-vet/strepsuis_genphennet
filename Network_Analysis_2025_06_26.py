#!/usr/bin/env python3
"""
Network_Analysis_2025_06_26.py - Canonical Entry Point
=======================================================

StrepSuis-GenPhenNet: Network-based genomic-phenotypic integration

This is the canonical entry point for workflow orchestration (Nextflow/Snakemake).
It wraps the strepsuis_genphennet package with standardized I/O following the
4-layer architecture.

Usage:
    python Network_Analysis_2025_06_26.py --config config.yaml
    python Network_Analysis_2025_06_26.py --data-dir input/raw_data --output out/run_20260131

Architecture Compliance:
    Input:  input/raw_data/*.csv + config.yaml
    Output: out/run_<ID>/
            ├── manifest.json
            ├── summary.json
            ├── results/*.parquet
            ├── figures/*.png
            ├── exports/*.csv
            ├── report.pdf
            └── site/

Module ID: StrepSuis-GenPhenNet
Canonical Name: Network_Analysis_2025_06_26.py
Date: 2025-06-26
"""

import sys
import argparse
from pathlib import Path

# Import the actual implementation from the package
from strepsuis_genphennet.cli import main as cli_main

def main():
    """
    Canonical entry point with architecture-compliant defaults.
    """
    parser = argparse.ArgumentParser(
        description='StrepSuis-GenPhenNet: Network-based genomic-phenotypic integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--config',
        type=Path,
        help='Path to config.yaml (default: input/config.yaml)'
    )

    parser.add_argument(
        '--data-dir',
        type=Path,
        help='Input data directory (default: input/raw_data)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        help='Output directory (default: out/run_<timestamp>)'
    )

    parser.add_argument(
        '--run-id',
        type=str,
        help='Run identifier for output directory (default: timestamp)'
    )

    # Pass through to underlying CLI
    sys.exit(cli_main())


if __name__ == '__main__':
    main()
