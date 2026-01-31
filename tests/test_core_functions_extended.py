#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extended tests for core functions in network_analysis_core.py
"""

import os
import pytest
import pandas as pd
import numpy as np
import tempfile

from strepsuis_genphennet.network_analysis_core import (
    chi2_phi,
    cramers_v,
    calculate_entropy,
    conditional_entropy,
    information_gain,
    normalized_mutual_info,
    find_mutually_exclusive,
    get_cluster_hubs,
    adaptive_phi_threshold,
    create_interactive_table,
    create_interactive_table_with_empty,
    create_section_summary,
    summarize_by_category,
    summarize_by_feature,
    summarize_by_category_excl,
    summarize_by_feature_excl,
    summarize_by_category_network,
    summarize_by_feature_network,
    get_centrality,
    expand_categories,
    find_matching_files,
    setup_logging,
)


class TestChi2PhiExtended:
    """Extended tests for chi2_phi function."""
    
    def test_chi2_phi_perfect_correlation(self):
        """Test with perfectly correlated variables."""
        x = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
        y = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
        
        p, phi, contingency, lo, hi = chi2_phi(x, y)
        
        assert phi > 0.9  # High correlation
    
    def test_chi2_phi_no_correlation(self):
        """Test with uncorrelated variables."""
        np.random.seed(42)
        x = pd.Series(np.random.randint(0, 2, 100))
        y = pd.Series(np.random.randint(0, 2, 100))
        
        p, phi, contingency, lo, hi = chi2_phi(x, y)
        
        assert phi < 0.3  # Low correlation
    
    def test_chi2_phi_small_sample(self):
        """Test with small sample (uses Fisher exact)."""
        x = pd.Series([0, 0, 1, 1])
        y = pd.Series([0, 1, 0, 1])
        
        p, phi, contingency, lo, hi = chi2_phi(x, y)
        
        assert isinstance(p, float)


class TestEntropyExtended:
    """Extended tests for entropy functions."""
    
    def test_calculate_entropy_high_variance(self):
        """Test entropy with high variance data."""
        x = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        
        H, Hn = calculate_entropy(x)
        
        assert H > 0.5  # High entropy (binary uniform is ~0.69)
    
    def test_calculate_entropy_low_variance(self):
        """Test entropy with low variance data."""
        x = pd.Series([0, 0, 0, 0, 0, 0, 0, 1])
        
        H, Hn = calculate_entropy(x)
        
        assert H < 0.6  # Lower entropy
    
    def test_conditional_entropy_independent(self):
        """Test conditional entropy for independent variables."""
        np.random.seed(42)
        x = pd.Series(np.random.randint(0, 2, 100))
        y = pd.Series(np.random.randint(0, 2, 100))
        
        ce = conditional_entropy(x, y)
        
        assert ce > 0
    
    def test_information_gain_dependent(self):
        """Test information gain for dependent variables."""
        x = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
        y = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
        
        ig = information_gain(x, y)
        
        assert ig > 0.5  # High information gain


class TestNetworkBuildingExtended:
    """Extended tests for network building."""
    
    def test_get_centrality_basic(self):
        """Test centrality calculation."""
        import networkx as nx
        
        G = nx.Graph()
        G.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'), ('b', 'd')])
        
        # get_centrality expects a dict, not a graph
        centrality_dict = nx.degree_centrality(G)
        centrality = get_centrality(centrality_dict)
        
        assert 'b' in centrality
    
    def test_get_centrality_empty(self):
        """Test centrality with empty dict."""
        centrality = get_centrality({})
        
        assert isinstance(centrality, dict)
        assert len(centrality) == 0


class TestStatisticalTestsExtended:
    """Extended tests for statistical tests."""
    
    def test_chi2_phi_various_sizes(self):
        """Test chi2_phi with various sample sizes."""
        for n in [10, 50, 100]:
            np.random.seed(42)
            x = pd.Series(np.random.randint(0, 2, n))
            y = pd.Series(np.random.randint(0, 2, n))
            
            p, phi, contingency, lo, hi = chi2_phi(x, y)
            
            assert isinstance(p, float)
            assert 0 <= p <= 1


class TestExpandCategoriesExtended:
    """Extended tests for category expansion."""
    
    def test_expand_categories_basic(self):
        """Test basic category expansion."""
        df = pd.DataFrame({
            'Strain_ID': [1, 2, 3, 4],
            'Category': ['A', 'B', 'A', 'C']
        })
        
        expanded = expand_categories(df, 'Category')
        
        assert 'Strain_ID' in expanded.columns
        assert len(expanded.columns) > 2  # Should have expanded columns


class TestSummaryFunctionsExtended:
    """Extended tests for summary functions."""
    
    def test_summarize_by_category_basic(self):
        """Test basic category summarization."""
        df = pd.DataFrame({
            'Category1': ['AMR', 'AMR', 'Vir', 'Vir'],
            'Category2': ['MIC', 'MIC', 'MIC', 'MIC'],
            'Value': [0.5, 0.6, 0.7, 0.8],
        })
        
        result = summarize_by_category(df, 'Value', ['Category1', 'Category2'])
        
        assert isinstance(result, dict)
        assert 'AMR' in result or 'MIC' in result
    
    def test_summarize_by_feature_basic(self):
        """Test basic feature summarization."""
        df = pd.DataFrame({
            'Feature1': ['gene1', 'gene1', 'gene2', 'gene2'],
            'Feature2': ['pheno1', 'pheno2', 'pheno1', 'pheno2'],
            'Value': [0.5, 0.6, 0.7, 0.8],
        })
        
        result = summarize_by_feature(df, 'Value', ['Feature1', 'Feature2'])
        
        assert isinstance(result, dict)
    
    def test_summarize_by_category_network(self):
        """Test network category summarization."""
        df = pd.DataFrame({
            'Feature': ['gene1', 'gene2', 'pheno1'],
            'Category': ['AMR', 'AMR', 'MIC'],
            'Degree_Centrality': [0.5, 0.6, 0.7],
        })
        
        result = summarize_by_category_network(df)
        
        assert isinstance(result, dict)
    
    def test_summarize_by_feature_network(self):
        """Test network feature summarization."""
        df = pd.DataFrame({
            'Feature': ['gene1', 'gene2', 'pheno1'],
            'Category': ['AMR', 'AMR', 'MIC'],
            'Degree_Centrality': [0.5, 0.6, 0.7],
        })
        
        result = summarize_by_feature_network(df)
        
        assert isinstance(result, dict)


class TestTableCreationExtended:
    """Extended tests for table creation."""
    
    def test_create_interactive_table_basic(self):
        """Test basic interactive table creation."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z'],
        })
        
        result = create_interactive_table(df, 'test_table')
        
        assert isinstance(result, str)
        assert 'table' in result.lower()
    
    def test_create_interactive_table_with_empty_nonempty(self):
        """Test interactive table with non-empty data."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z'],
        })
        
        result = create_interactive_table_with_empty(df, 'test_table')
        
        assert isinstance(result, str)
        assert 'table' in result.lower()
    
    def test_create_section_summary_basic(self):
        """Test section summary creation."""
        summary = {
            'Total': 100,
            'Significant': 50,
            'Mean': 0.5,
        }
        
        # create_section_summary takes (title, stats, per_category, per_feature)
        result = create_section_summary("Test Section", summary)
        
        assert isinstance(result, str)


class TestSetupLoggingExtended:
    """Extended tests for logging setup."""
    
    def test_setup_logging_creates_logger(self):
        """Test that setup_logging creates a logger."""
        setup_logging()
        
        import logging
        logger = logging.getLogger()
        
        assert logger is not None


class TestAdaptiveThresholdExtended:
    """Extended tests for adaptive threshold."""
    
    def test_adaptive_phi_threshold_percentile(self):
        """Test percentile method."""
        phi_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        
        threshold = adaptive_phi_threshold(phi_vals, method='percentile', percentile=75)
        
        assert 0.6 <= threshold <= 0.8
    
    def test_adaptive_phi_threshold_iqr(self):
        """Test IQR method."""
        phi_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        
        threshold = adaptive_phi_threshold(phi_vals, method='iqr')
        
        assert threshold > 0
    
    def test_adaptive_phi_threshold_statistical(self):
        """Test statistical method."""
        phi_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        
        threshold = adaptive_phi_threshold(phi_vals, method='statistical')
        
        assert threshold > 0
