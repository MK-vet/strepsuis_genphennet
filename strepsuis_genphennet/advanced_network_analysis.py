#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Network Analysis Module for StrepSuis-GenPhenNet
========================================================

This module integrates advanced statistical methods for robust network
construction and association analysis.

Features:
    - Edge Confidence: Bootstrap-based network edge stability
    - Rare Pattern Detection: Find rare but consistent genomic-phenotypic associations
    - Multi-view Concordance: Compare network communities with other groupings
    - Confidence-aware Rules: Association rules with bootstrap confidence intervals
    - Consensus Evidence Score: Multi-method feature ranking

Author: MK-vet
Version: 1.0.0
License: MIT
"""

from typing import Dict, List, Optional, Callable, Tuple
import pandas as pd
import numpy as np
import networkx as nx
import logging

try:
    from shared.advanced_statistics import (
        edge_confidence_network,
        rare_pattern_detector,
        multiview_concordance,
        confidence_aware_rules,
        consensus_evidence_score,
    )
    HAS_ADVANCED_STATS = True
except ImportError:
    HAS_ADVANCED_STATS = False
    logging.warning(
        "Advanced statistics module not available. "
        "Install with: pip install -e ../shared"
    )

logger = logging.getLogger(__name__)


def build_robust_network(
    data: pd.DataFrame,
    edge_function: Callable,
    n_bootstrap: int = 500,
    confidence_threshold: float = 0.8,
    n_jobs: int = -1
) -> Dict:
    """
    Build a network with bootstrap-validated edges.

    Only edges that appear in at least `confidence_threshold` fraction of
    bootstrap iterations are retained in the final network.

    Parameters
    ----------
    data : pd.DataFrame
        Input data (samples × features)
    edge_function : callable
        Function that takes DataFrame and returns list of (node1, node2, weight) tuples
    n_bootstrap : int, default=500
        Number of bootstrap iterations
    confidence_threshold : float, default=0.8
        Minimum confidence to retain edge (0.8 = edge must appear in 80% of bootstraps)
    n_jobs : int, default=-1
        Number of parallel jobs

    Returns
    -------
    dict
        Dictionary with:
        - 'network': NetworkX graph with high-confidence edges
        - 'edge_stats': Statistics for all edges
        - 'confidence_distribution': Distribution of edge confidences

    Example
    -------
    >>> def my_edge_func(df):
    ...     # Calculate associations and return edges
    ...     edges = []
    ...     for col1 in df.columns:
    ...         for col2 in df.columns:
    ...             if col1 < col2:
    ...                 corr = df[col1].corr(df[col2])
    ...                 if abs(corr) > 0.3:
    ...                     edges.append((col1, col2, abs(corr)))
    ...     return edges
    >>>
    >>> network_results = build_robust_network(
    ...     data=genomic_data,
    ...     edge_function=my_edge_func,
    ...     confidence_threshold=0.8
    ... )
    >>> G = network_results['network']
    >>> print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    """
    if not HAS_ADVANCED_STATS:
        raise ImportError(
            "Advanced statistics module is required. "
            "Install shared module: pip install -e ../shared"
        )

    logger.info("Building robust network with bootstrap edge confidence...")

    # Get edge confidence statistics
    edge_results = edge_confidence_network(
        data=data,
        edge_func=edge_function,
        n_bootstrap=n_bootstrap,
        confidence_threshold=confidence_threshold,
        n_jobs=n_jobs
    )

    # Build NetworkX graph from high-confidence edges
    G = nx.Graph()

    for edge, stats in edge_results['high_confidence_edges'].items():
        node1, node2 = edge
        G.add_edge(
            node1,
            node2,
            weight=stats['mean_weight'],
            confidence=stats['confidence'],
            weight_std=stats['weight_std']
        )

    # Confidence distribution
    all_confidences = [s['confidence'] for s in edge_results['all_edges'].values()]
    confidence_dist = {
        'mean': np.mean(all_confidences),
        'median': np.median(all_confidences),
        'min': np.min(all_confidences),
        'max': np.max(all_confidences),
        'std': np.std(all_confidences)
    }

    return {
        'network': G,
        'edge_stats': edge_results,
        'confidence_distribution': confidence_dist,
        'n_total_edges': edge_results['n_total_edges'],
        'n_high_confidence': edge_results['n_high_confidence'],
        'retention_rate': edge_results['n_high_confidence'] / max(edge_results['n_total_edges'], 1)
    }


def find_rare_genomic_phenotypic_associations(
    data: pd.DataFrame,
    min_support: float = 0.02,
    max_support: float = 0.05,
    min_confidence: float = 0.8,
    min_samples: int = 3
) -> List[Dict]:
    """
    Detect rare but consistent genomic-phenotypic associations.

    These are patterns with low support (few samples) but high confidence,
    representing potential strain-specific or rare phenotypic combinations.

    Parameters
    ----------
    data : pd.DataFrame
        Binary feature matrix (genomic + phenotypic features)
    min_support : float, default=0.02
        Minimum support threshold (2%)
    max_support : float, default=0.05
        Maximum support for rare patterns (5%)
    min_confidence : float, default=0.8
        Minimum confidence threshold (80%)
    min_samples : int, default=3
        Minimum absolute sample count

    Returns
    -------
    list of dict
        List of rare association patterns

    Example
    -------
    >>> rare_assocs = find_rare_genomic_phenotypic_associations(
    ...     data=combined_data,
    ...     min_support=0.02,
    ...     max_support=0.05,
    ...     min_confidence=0.8
    ... )
    >>> for assoc in rare_assocs[:5]:
    ...     print(assoc['interpretation'])
    """
    if not HAS_ADVANCED_STATS:
        raise ImportError(
            "Advanced statistics module is required. "
            "Install shared module: pip install -e ../shared"
        )

    logger.info("Searching for rare genomic-phenotypic associations...")

    patterns = rare_pattern_detector(
        data=data,
        min_support=min_support,
        max_support=max_support,
        min_confidence=min_confidence,
        min_samples=min_samples
    )

    logger.info(f"Found {len(patterns)} rare association patterns")
    return patterns


def compare_network_with_other_analyses(
    network_communities: np.ndarray,
    clustering_labels: np.ndarray,
    phylo_groups: Optional[np.ndarray] = None
) -> Dict:
    """
    Compare network community assignments with other analysis methods.

    Uses NMI and ARI to quantify agreement between different grouping methods.

    Parameters
    ----------
    network_communities : np.ndarray
        Community assignments from network analysis
    clustering_labels : np.ndarray
        Cluster assignments from clustering (e.g., K-Modes)
    phylo_groups : np.ndarray, optional
        Phylogenetic groupings

    Returns
    -------
    dict
        Concordance metrics and interpretation

    Example
    -------
    >>> import community as community_louvain
    >>> communities = community_louvain.best_partition(network)
    >>> community_array = np.array([communities[node] for node in sorted(communities.keys())])
    >>>
    >>> concordance = compare_network_with_other_analyses(
    ...     network_communities=community_array,
    ...     clustering_labels=kmodes_labels
    ... )
    >>> print(concordance['overall']['interpretation'])
    """
    if not HAS_ADVANCED_STATS:
        raise ImportError(
            "Advanced statistics module is required. "
            "Install shared module: pip install -e ../shared"
        )

    logger.info("Comparing network communities with other analyses...")

    results = multiview_concordance(
        clustering_labels=clustering_labels,
        network_communities=network_communities,
        association_groups=phylo_groups
    )

    return results


def association_rules_with_confidence_intervals(
    data: pd.DataFrame,
    min_support: float = 0.05,
    min_confidence: float = 0.7,
    n_bootstrap: int = 500,
    ci_level: float = 0.95,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Find association rules with bootstrap confidence intervals for lift.

    This provides a more robust assessment of rule strength by quantifying
    uncertainty in the lift metric.

    Parameters
    ----------
    data : pd.DataFrame
        Binary feature data
    min_support : float, default=0.05
        Minimum support threshold
    min_confidence : float, default=0.7
        Minimum confidence threshold
    n_bootstrap : int, default=500
        Number of bootstrap iterations
    ci_level : float, default=0.95
        Confidence interval level
    n_jobs : int, default=-1
        Number of parallel jobs

    Returns
    -------
    pd.DataFrame
        Association rules with confidence intervals and significance flags

    Example
    -------
    >>> rules_with_ci = association_rules_with_confidence_intervals(
    ...     data=binary_data,
    ...     min_support=0.05,
    ...     min_confidence=0.7,
    ...     n_bootstrap=500
    ... )
    >>>
    >>> # Filter for significant rules
    >>> significant_rules = rules_with_ci[rules_with_ci['is_significant']]
    >>> print(f"Found {len(significant_rules)} significant association rules")
    """
    if not HAS_ADVANCED_STATS:
        raise ImportError(
            "Advanced statistics module is required. "
            "Install shared module: pip install -e ../shared"
        )

    logger.info("Computing association rules with bootstrap confidence intervals...")

    rules = confidence_aware_rules(
        data=data,
        min_support=min_support,
        min_confidence=min_confidence,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        n_jobs=n_jobs
    )

    if not rules.empty:
        logger.info(f"Found {len(rules)} total rules, "
                   f"{rules['is_significant'].sum()} significant")

    return rules


def multi_method_feature_ranking(
    data: pd.DataFrame,
    target: pd.Series,
    features: List[str],
    n_bootstrap: int = 500,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Rank features using consensus evidence from multiple statistical methods.

    Parameters
    ----------
    data : pd.DataFrame
        Feature matrix
    target : pd.Series
        Target variable (e.g., phenotype, cluster label)
    features : list of str
        List of feature names to rank
    n_bootstrap : int, default=500
        Number of bootstrap iterations for each method
    n_jobs : int, default=-1
        Number of parallel jobs

    Returns
    -------
    pd.DataFrame
        Features ranked by consensus evidence score

    Example
    -------
    >>> feature_ranking = multi_method_feature_ranking(
    ...     data=genomic_features,
    ...     target=resistance_phenotype,
    ...     features=genomic_features.columns.tolist(),
    ...     n_bootstrap=500
    ... )
    >>> print(feature_ranking.head(10))
    """
    if not HAS_ADVANCED_STATS:
        raise ImportError(
            "Advanced statistics module is required. "
            "Install shared module: pip install -e ../shared"
        )

    logger.info("Computing consensus evidence scores for feature ranking...")

    results = []
    for feature in features:
        try:
            ces = consensus_evidence_score(
                data=data,
                target=target,
                feature=feature,
                n_bootstrap=n_bootstrap,
                n_jobs=n_jobs
            )
            results.append({
                'feature': feature,
                'consensus_score': ces['consensus_score'],
                'chi2_stat': ces['chi2_stat'],
                'log_odds': ces['log_odds'],
                'rf_importance': ces['rf_importance']
            })
        except Exception as e:
            logger.warning(f"Failed to compute CES for {feature}: {e}")

    ranking_df = pd.DataFrame(results).sort_values('consensus_score', ascending=False)
    return ranking_df


def generate_network_advanced_report(
    robust_network: Optional[Dict] = None,
    rare_patterns: Optional[List[Dict]] = None,
    concordance: Optional[Dict] = None,
    rules_with_ci: Optional[pd.DataFrame] = None,
    feature_ranking: Optional[pd.DataFrame] = None,
    output_file: str = "network_advanced_report.html"
) -> None:
    """
    Generate HTML report for advanced network analysis.

    Parameters
    ----------
    robust_network : dict, optional
        Results from build_robust_network()
    rare_patterns : list, optional
        Results from find_rare_genomic_phenotypic_associations()
    concordance : dict, optional
        Results from compare_network_with_other_analyses()
    rules_with_ci : pd.DataFrame, optional
        Results from association_rules_with_confidence_intervals()
    feature_ranking : pd.DataFrame, optional
        Results from multi_method_feature_ranking()
    output_file : str
        Output HTML file path
    """
    html = ["<html><head><title>Advanced Network Analysis</title>"]
    html.append("<style>")
    html.append("body { font-family: Arial, sans-serif; margin: 20px; }")
    html.append("table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
    html.append("th, td { border: 1px solid #ddd; padding: 8px; }")
    html.append("th { background-color: #1565C0; color: white; }")
    html.append("h1 { color: #0D47A1; }")
    html.append("h2 { color: #1976D2; }")
    html.append(".metric { background-color: #E3F2FD; padding: 10px; margin: 10px 0; border-radius: 5px; }")
    html.append("</style></head><body>")

    html.append("<h1>Advanced Network Analysis Report</h1>")

    # Robust network
    if robust_network:
        html.append("<h2>Bootstrap-Validated Network</h2>")
        html.append(f"<div class='metric'>")
        html.append(f"<strong>Total possible edges:</strong> {robust_network['n_total_edges']}<br>")
        html.append(f"<strong>High-confidence edges:</strong> {robust_network['n_high_confidence']}<br>")
        html.append(f"<strong>Edge retention rate:</strong> {robust_network['retention_rate']:.1%}<br>")
        html.append(f"<strong>Mean edge confidence:</strong> {robust_network['confidence_distribution']['mean']:.3f}")
        html.append(f"</div>")

    # Rare patterns
    if rare_patterns:
        html.append("<h2>Rare Genomic-Phenotypic Associations</h2>")
        html.append(f"<p>Found <strong>{len(rare_patterns)}</strong> rare but consistent patterns:</p>")
        html.append("<ul>")
        for pattern in rare_patterns[:10]:
            html.append(f"<li>{pattern['interpretation']}</li>")
        html.append("</ul>")

    # Concordance
    if concordance:
        html.append("<h2>Multi-View Concordance</h2>")
        overall = concordance['overall']
        html.append(f"<div class='metric'>")
        html.append(f"<strong>Mean NMI:</strong> {overall['mean_nmi']:.3f}<br>")
        html.append(f"<strong>Interpretation:</strong> {overall['interpretation']}")
        html.append(f"</div>")

    # Rules with CI
    if rules_with_ci is not None and not rules_with_ci.empty:
        html.append("<h2>Significant Association Rules (CI excludes 1)</h2>")
        sig_rules = rules_with_ci[rules_with_ci['is_significant']]
        html.append(f"<p>Found <strong>{len(sig_rules)}</strong> significant rules</p>")
        if len(sig_rules) > 0:
            display_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift', 'lift_ci_lower', 'lift_ci_upper']
            html.append(sig_rules[display_cols].head(15).to_html(index=False))

    # Feature ranking
    if feature_ranking is not None:
        html.append("<h2>Top Features by Consensus Evidence</h2>")
        html.append(feature_ranking.head(15).to_html(index=False))

    html.append("</body></html>")

    with open(output_file, 'w') as f:
        f.write('\n'.join(html))

    logger.info(f"Advanced network analysis report saved to: {output_file}")
