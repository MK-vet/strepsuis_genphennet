#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for statistical functions in network_analysis_core.py.

Tests chi2_phi, cramers_v, and other statistical functions.
"""

import pytest
import pandas as pd
import numpy as np

try:
    from strepsuis_genphennet.network_analysis_core import (
        chi2_phi,
        cramers_v,
        adaptive_phi_threshold,
        get_cluster_hubs,
    )
    FUNCTIONS_AVAILABLE = True
except ImportError:
    FUNCTIONS_AVAILABLE = False


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
class TestChi2Phi:
    """Test chi2_phi function."""
    
    def test_chi2_phi_basic(self):
        """Test basic chi2_phi calculation."""
        x = pd.Series([1, 1, 0, 0, 1, 0])
        y = pd.Series([1, 0, 1, 0, 1, 0])
        
        p, phi, contingency, lo, hi = chi2_phi(x, y)
        
        assert isinstance(p, (float, np.floating))
        assert isinstance(phi, (float, np.floating))
        assert isinstance(contingency, pd.DataFrame)
        assert 0 <= phi <= 1
    
    def test_chi2_phi_small_sample(self):
        """Test chi2_phi with small sample (should use Fisher exact)."""
        x = pd.Series([1, 0, 1])
        y = pd.Series([1, 0, 0])
        
        p, phi, contingency, lo, hi = chi2_phi(x, y)
        
        assert isinstance(p, (float, np.floating))
        assert isinstance(phi, (float, np.floating))
    
    def test_chi2_phi_identical(self):
        """Test chi2_phi with identical series."""
        x = pd.Series([1, 1, 0, 0])
        y = pd.Series([1, 1, 0, 0])
        
        p, phi, contingency, lo, hi = chi2_phi(x, y)
        
        # Should have high phi (strong association)
        assert phi > 0.5


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
class TestCramersV:
    """Test cramers_v function."""
    
    def test_cramers_v_basic(self):
        """Test basic cramers_v calculation."""
        contingency = pd.DataFrame({
            'A': [10, 5],
            'B': [5, 10],
        })
        
        cv, chi2, p = cramers_v(contingency)
        
        assert isinstance(cv, (float, np.floating))
        assert isinstance(chi2, (float, np.floating))
        assert isinstance(p, (float, np.floating))
        assert 0 <= cv <= 1
    
    def test_cramers_v_small_table(self):
        """Test cramers_v with small table."""
        contingency = pd.DataFrame({
            'A': [1, 1],
        })
        
        cv, chi2, p = cramers_v(contingency)
        
        # May return 0.0 for invalid table
        assert isinstance(cv, (float, np.floating))


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
class TestAdaptivePhiThreshold:
    """Test adaptive_phi_threshold function."""
    
    def test_adaptive_phi_threshold_percentile(self):
        """Test percentile method."""
        phi_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        threshold = adaptive_phi_threshold(phi_vals, method='percentile', percentile=90)
        
        assert isinstance(threshold, (float, np.floating))
        assert threshold >= 0
    
    def test_adaptive_phi_threshold_iqr(self):
        """Test IQR method."""
        phi_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        threshold = adaptive_phi_threshold(phi_vals, method='iqr')
        
        assert isinstance(threshold, (float, np.floating))
        assert threshold >= 0
    
    def test_adaptive_phi_threshold_statistical(self):
        """Test statistical method."""
        phi_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        threshold = adaptive_phi_threshold(phi_vals, method='statistical')
        
        assert isinstance(threshold, (float, np.floating))
        assert threshold >= 0


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
class TestGetClusterHubs:
    """Test get_cluster_hubs function."""
    
    def test_get_cluster_hubs_basic(self):
        """Test basic cluster hubs extraction."""
        df = pd.DataFrame({
            'Cluster': [1, 1, 1, 2, 2, 2],
            'Feature': ['A', 'B', 'C', 'D', 'E', 'F'],
            'Category': ['AMR', 'AMR', 'Vir', 'AMR', 'Vir', 'Vir'],
            'Degree_Centrality': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        })
        
        hubs = get_cluster_hubs(df, top_n=2)
        
        assert isinstance(hubs, pd.DataFrame)
        if not hubs.empty:
            assert 'Cluster' in hubs.columns
            assert 'Feature' in hubs.columns
            assert 'Degree_Centrality' in hubs.columns
    
    def test_get_cluster_hubs_missing_columns(self):
        """Test with missing columns."""
        df = pd.DataFrame({
            'Feature': ['A', 'B'],
            'Value': [1, 2],
        })
        
        hubs = get_cluster_hubs(df)
        
        # Should return empty DataFrame
        assert isinstance(hubs, pd.DataFrame)
        assert hubs.empty
