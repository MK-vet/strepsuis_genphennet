#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for network analysis functions.

Tests build_network, fdr_correction, get_centrality, and other network functions.
"""

import pytest
import pandas as pd
import numpy as np
import networkx as nx

try:
    from strepsuis_genphennet.network_analysis_core import (
        build_network,
        fdr_correction,
        get_centrality,
        summarize_by_category,
        summarize_by_feature,
    )
    FUNCTIONS_AVAILABLE = True
except ImportError:
    FUNCTIONS_AVAILABLE = False


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
class TestBuildNetwork:
    """Test network building."""
    
    @pytest.fixture
    def sample_associations(self):
        """Create sample associations DataFrame."""
        return pd.DataFrame({
            'Feature1': ['Gene_A', 'Gene_B', 'Gene_C'],
            'Feature2': ['Pheno_X', 'Pheno_Y', 'Pheno_Z'],
            'Phi': [0.5, 0.6, 0.7],
            'P_Value': [0.01, 0.02, 0.03],
            'FDR_corrected_p': [0.05, 0.06, 0.07],
        })
    
    def test_build_network_basic(self, sample_associations):
        """Test basic network building."""
        network = build_network(sample_associations, threshold=0.5)
        
        assert isinstance(network, nx.Graph)
        assert len(network.nodes()) > 0
    
    def test_build_network_empty(self):
        """Test with empty associations."""
        empty_df = pd.DataFrame()
        network = build_network(empty_df, threshold=0.5)
        
        assert isinstance(network, nx.Graph)
        assert len(network.nodes()) == 0


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
class TestFdrCorrection:
    """Test FDR correction."""
    
    def test_fdr_correction_basic(self):
        """Test basic FDR correction."""
        p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        
        rejected, corrected = fdr_correction(p_values, alpha=0.05)
        
        assert isinstance(rejected, np.ndarray)
        assert isinstance(corrected, np.ndarray)
        assert len(rejected) == len(p_values)
        assert len(corrected) == len(p_values)
    
    def test_fdr_correction_empty(self):
        """Test with empty p-values."""
        p_values = np.array([])
        
        rejected, corrected = fdr_correction(p_values, alpha=0.05)
        
        assert len(rejected) == 0
        assert len(corrected) == 0


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
class TestGetCentrality:
    """Test centrality calculation."""
    
    @pytest.fixture
    def sample_network(self):
        """Create sample network."""
        G = nx.Graph()
        G.add_edge('A', 'B')
        G.add_edge('B', 'C')
        G.add_edge('C', 'D')
        return G
    
    def test_get_centrality_basic(self, sample_network):
        """Test basic centrality calculation."""
        centrality = get_centrality(sample_network, 'degree')
        
        assert isinstance(centrality, dict)
        assert len(centrality) == len(sample_network.nodes())
    
    def test_get_centrality_betweenness(self, sample_network):
        """Test betweenness centrality."""
        centrality = get_centrality(sample_network, 'betweenness')
        
        assert isinstance(centrality, dict)
        assert len(centrality) == len(sample_network.nodes())


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
class TestSummarizeFunctions:
    """Test summary functions."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            'Category1': ['AMR', 'AMR', 'Vir'],
            'Category2': ['Vir', 'AMR', 'AMR'],
            'Value': [0.5, 0.6, 0.7],
        })
    
    def test_summarize_by_category(self, sample_df):
        """Test summarize by category."""
        result = summarize_by_category(sample_df, 'Value', ['Category1', 'Category2'])
        
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_summarize_by_feature(self, sample_df):
        """Test summarize by feature."""
        result = summarize_by_feature(sample_df, 'Value', ['Category1', 'Category2'])
        
        assert isinstance(result, dict)
        assert len(result) > 0
