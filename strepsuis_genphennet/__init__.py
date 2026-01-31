"""
StrepSuis-GenPhenNet: Network-Based Integration of Genomic and Phenotypic Data
===============================================================================

A bioinformatics tool for network-based analysis of genomic-phenotypic
associations in bacterial genomics.

Features:
    - Statistical association testing (Chi-square, Fisher's exact, Cramér's V)
    - Multiple testing correction using Benjamini-Hochberg FDR
    - Network construction from significant associations
    - Community detection using Louvain algorithm
    - Centrality metrics (degree, betweenness, closeness)
    - Information theory metrics (entropy, mutual information)

Example:
    >>> from strepsuis_genphennet import NetworkAnalyzer, Config
    >>> config = Config(data_dir="./data", output_dir="./output")
    >>> analyzer = NetworkAnalyzer(config)
    >>> results = analyzer.run()

Author: MK-vet
License: MIT
"""

__version__ = "1.0.0"
__author__ = "MK-vet"
__license__ = "MIT"

import types

from .analyzer import NetworkAnalyzer
from .config import Config

# Provide a lightweight stub to satisfy legacy test patches
mdr_analysis_core = types.SimpleNamespace(
    setup_environment=lambda *args, **kwargs: None,
    __name__="mdr_analysis_core",
)

# Advanced statistical features from shared module
try:
    from shared.advanced_statistics import (
        edge_confidence_network,
        rare_pattern_detector,
        multiview_concordance,
        confidence_aware_rules,
        consensus_evidence_score,
    )
    _HAS_ADVANCED_STATS = True
except ImportError:
    _HAS_ADVANCED_STATS = False

__all__ = ["NetworkAnalyzer", "Config", "__version__"]

# Add advanced statistics if available
if _HAS_ADVANCED_STATS:
    __all__.extend([
        "edge_confidence_network",
        "rare_pattern_detector",
        "multiview_concordance",
        "confidence_aware_rules",
        "consensus_evidence_score",
    ])
