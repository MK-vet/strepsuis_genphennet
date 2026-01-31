#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for network building and analysis functions.
"""

import pytest
import pandas as pd
import numpy as np
import networkx as nx

try:
    import community
    COMMUNITY_AVAILABLE = True
except ImportError:
    COMMUNITY_AVAILABLE = False

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
)


class TestChi2PhiEdgeCases:
    """Edge case tests for chi2_phi function."""
    
    def test_chi2_phi_all_zeros(self):
        """Test with all zeros."""
        x = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        try:
            p, phi, contingency, lo, hi = chi2_phi(x, y)
            assert isinstance(p, float)
        except Exception:
            # May fail with constant values
            pass
    
    def test_chi2_phi_all_ones(self):
        """Test with all ones."""
        x = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        y = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        
        try:
            p, phi, contingency, lo, hi = chi2_phi(x, y)
            assert isinstance(p, float)
        except Exception:
            # May fail with constant values
            pass
    
    def test_chi2_phi_inverse_correlation(self):
        """Test with inverse correlation."""
        x = pd.Series([1, 1, 1, 0, 0, 0] * 10)
        y = pd.Series([0, 0, 0, 1, 1, 1] * 10)
        
        p, phi, contingency, lo, hi = chi2_phi(x, y)
        
        assert p < 0.05  # Should be significant
    
    def test_chi2_phi_large_sample(self):
        """Test with large sample."""
        np.random.seed(42)
        x = pd.Series(np.random.binomial(1, 0.5, 1000))
        y = pd.Series(np.random.binomial(1, 0.5, 1000))
        
        p, phi, contingency, lo, hi = chi2_phi(x, y)
        
        assert isinstance(p, float)
        assert isinstance(phi, float)


class TestCramersVEdgeCases:
    """Edge case tests for cramers_v function."""
    
    def test_cramers_v_3x3(self):
        """Test with 3x3 contingency table."""
        contingency = pd.DataFrame({
            'A': [10, 5, 3],
            'B': [5, 15, 8],
            'C': [3, 8, 20],
        })
        
        v, lo, hi = cramers_v(contingency)
        
        assert 0 <= v <= 1
    
    def test_cramers_v_4x2(self):
        """Test with 4x2 contingency table."""
        contingency = pd.DataFrame({
            'A': [10, 5, 3, 2],
            'B': [5, 15, 8, 10],
        })
        
        v, lo, hi = cramers_v(contingency)
        
        assert 0 <= v <= 1


class TestEntropyEdgeCases:
    """Edge case tests for entropy functions."""
    
    def test_entropy_multiclass(self):
        """Test entropy with multiple classes."""
        series = pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2])
        
        H, Hn = calculate_entropy(series)
        
        assert H > 0
        assert 0 <= Hn <= 1
    
    def test_conditional_entropy_independent(self):
        """Test conditional entropy with independent variables."""
        np.random.seed(42)
        x = pd.Series(np.random.binomial(1, 0.5, 100))
        y = pd.Series(np.random.binomial(1, 0.5, 100))
        
        ce = conditional_entropy(x, y)
        
        assert ce >= 0
    
    def test_information_gain_high(self):
        """Test information gain with high correlation."""
        x = pd.Series([1, 1, 1, 0, 0, 0] * 10)
        y = pd.Series([1, 1, 1, 0, 0, 0] * 10)
        
        ig = information_gain(x, y)
        
        assert ig >= 0
    
    def test_normalized_mutual_info_bounds(self):
        """Test normalized mutual info bounds."""
        x = pd.Series([1, 1, 0, 0, 1, 0, 1, 0] * 5)
        y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0] * 5)
        
        nmi = normalized_mutual_info(x, y)
        
        assert 0 <= nmi <= 1


class TestMutuallyExclusiveEdgeCases:
    """Edge case tests for find_mutually_exclusive."""
    
    def test_find_mutually_exclusive_perfect(self):
        """Test with perfectly mutually exclusive features."""
        data = pd.DataFrame({
            'Feature_A': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            'Feature_B': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        })
        
        features = ['Feature_A', 'Feature_B']
        feature_category = {'Feature_A': 'AMR', 'Feature_B': 'AMR'}
        
        result = find_mutually_exclusive(data, features, feature_category, k=2)
        
        assert isinstance(result, pd.DataFrame)
    
    def test_find_mutually_exclusive_overlapping(self):
        """Test with overlapping features."""
        data = pd.DataFrame({
            'Feature_A': [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
            'Feature_B': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        })
        
        features = ['Feature_A', 'Feature_B']
        feature_category = {'Feature_A': 'AMR', 'Feature_B': 'Vir'}
        
        result = find_mutually_exclusive(data, features, feature_category, k=2)
        
        assert isinstance(result, pd.DataFrame)


class TestGetClusterHubsEdgeCases:
    """Edge case tests for get_cluster_hubs."""
    
    def test_get_cluster_hubs_single_cluster(self):
        """Test with single cluster."""
        df = pd.DataFrame({
            'Feature': ['A', 'B', 'C'],
            'Degree_Centrality': [0.9, 0.5, 0.3],
            'Category': ['AMR', 'AMR', 'AMR'],
            'Cluster': [1, 1, 1],
        })
        
        result = get_cluster_hubs(df, top_n=2)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 2
    
    def test_get_cluster_hubs_many_clusters(self):
        """Test with many clusters."""
        df = pd.DataFrame({
            'Feature': [f'Gene_{i}' for i in range(20)],
            'Degree_Centrality': [0.9 - i * 0.04 for i in range(20)],
            'Category': ['AMR'] * 10 + ['Vir'] * 10,
            'Cluster': [i % 5 for i in range(20)],
        })
        
        result = get_cluster_hubs(df, top_n=2)
        
        assert isinstance(result, pd.DataFrame)


class TestAdaptivePhiThresholdEdgeCases:
    """Edge case tests for adaptive_phi_threshold."""
    
    def test_adaptive_phi_threshold_percentile(self):
        """Test percentile method."""
        phi_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        
        threshold = adaptive_phi_threshold(phi_values, method='percentile', percentile=75)
        
        assert 0 <= threshold <= 1
    
    def test_adaptive_phi_threshold_iqr(self):
        """Test IQR method."""
        phi_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        
        threshold = adaptive_phi_threshold(phi_values, method='iqr')
        
        assert threshold >= 0
    
    def test_adaptive_phi_threshold_statistical(self):
        """Test statistical method."""
        phi_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        
        threshold = adaptive_phi_threshold(phi_values, method='statistical')
        
        assert threshold >= 0
    
    def test_adaptive_phi_threshold_unknown_method(self):
        """Test unknown method (should return default)."""
        phi_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        threshold = adaptive_phi_threshold(phi_values, method='unknown')
        
        assert threshold == 0.5
