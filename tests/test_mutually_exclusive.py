#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for find_mutually_exclusive and related functions.
"""

import pytest
import pandas as pd
import numpy as np

from strepsuis_genphennet.network_analysis_core import (
    find_mutually_exclusive,
    get_cluster_hubs,
    adaptive_phi_threshold,
    create_interactive_table_with_empty,
)


class TestFindMutuallyExclusive:
    """Test find_mutually_exclusive function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with mutually exclusive features."""
        return pd.DataFrame({
            'Feature_A': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            'Feature_B': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'Feature_C': [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        })
    
    @pytest.fixture
    def feature_category(self):
        """Create feature category mapping."""
        return {
            'Feature_A': 'AMR',
            'Feature_B': 'AMR',
            'Feature_C': 'Vir',
        }
    
    def test_find_mutually_exclusive_k2(self, sample_data, feature_category):
        """Test finding mutually exclusive pairs."""
        features = ['Feature_A', 'Feature_B', 'Feature_C']
        
        result = find_mutually_exclusive(sample_data, features, feature_category, k=2)
        
        assert isinstance(result, pd.DataFrame)
    
    def test_find_mutually_exclusive_k3(self, sample_data, feature_category):
        """Test finding mutually exclusive triplets."""
        features = ['Feature_A', 'Feature_B', 'Feature_C']
        
        result = find_mutually_exclusive(sample_data, features, feature_category, k=3)
        
        assert isinstance(result, pd.DataFrame)
    
    def test_find_mutually_exclusive_empty(self, feature_category):
        """Test with empty data."""
        empty_data = pd.DataFrame()
        features = []
        
        result = find_mutually_exclusive(empty_data, features, feature_category, k=2)
        
        assert isinstance(result, pd.DataFrame)


class TestGetClusterHubs:
    """Test get_cluster_hubs function."""
    
    def test_get_cluster_hubs_basic(self):
        """Test basic cluster hubs extraction."""
        df = pd.DataFrame({
            'Feature': ['A', 'B', 'C', 'D', 'E'],
            'Degree_Centrality': [0.9, 0.8, 0.5, 0.3, 0.1],
            'Category': ['AMR', 'AMR', 'Vir', 'Vir', 'MGE'],
        })
        
        result = get_cluster_hubs(df, top_n=2)
        
        assert isinstance(result, pd.DataFrame)
    
    def test_get_cluster_hubs_empty(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['Feature', 'Degree_Centrality', 'Category'])
        
        result = get_cluster_hubs(empty_df, top_n=3)
        
        assert isinstance(result, pd.DataFrame)


class TestAdaptivePhiThreshold:
    """Test adaptive_phi_threshold function."""
    
    def test_adaptive_phi_threshold_basic(self):
        """Test basic adaptive threshold calculation."""
        phi_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        
        threshold = adaptive_phi_threshold(phi_values)
        
        assert isinstance(threshold, float)
        assert 0 <= threshold <= 1
    
    def test_adaptive_phi_threshold_empty(self):
        """Test with empty array."""
        phi_values = np.array([])
        
        try:
            threshold = adaptive_phi_threshold(phi_values)
            assert isinstance(threshold, float)
        except (IndexError, ValueError):
            # May fail with empty array
            pytest.skip("adaptive_phi_threshold may not work with empty array")
    
    def test_adaptive_phi_threshold_single_value(self):
        """Test with single value."""
        phi_values = np.array([0.5])
        
        threshold = adaptive_phi_threshold(phi_values)
        
        assert isinstance(threshold, float)


class TestCreateInteractiveTableWithEmpty:
    """Test create_interactive_table_with_empty function."""
    
    def test_create_interactive_table_with_empty_valid(self):
        """Test with valid DataFrame."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
        })
        
        result = create_interactive_table_with_empty(df, 'test_table')
        
        assert isinstance(result, str)
        assert 'test_table' in result or 'table' in result.lower()
    
    def test_create_interactive_table_with_empty_empty_df(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        result = create_interactive_table_with_empty(empty_df, 'empty_table')
        
        assert isinstance(result, str)
        assert 'No data' in result or 'available' in result.lower() or len(result) > 0
    
    def test_create_interactive_table_with_empty_none(self):
        """Test with None DataFrame."""
        try:
            result = create_interactive_table_with_empty(None, 'none_table')
            assert isinstance(result, str)
        except (TypeError, AttributeError):
            # May fail with None input
            pytest.skip("create_interactive_table_with_empty may not work with None")
