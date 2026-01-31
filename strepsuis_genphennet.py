#!/usr/bin/env python3
"""
strepsuis_genphennet.py - Canonical Entry Point
================================================

Network-based genomic-phenotypic integration

Usage:
    python strepsuis_genphennet.py --data-dir input/raw_data --output out/run_20260131
    python strepsuis_genphennet.py --config config.yaml

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
"""

import sys
from strepsuis_genphennet.cli import main

if __name__ == '__main__':
    sys.exit(main())
