#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detailed tests for summarize functions.
"""

import pytest
import pandas as pd
import numpy as np

from strepsuis_genphennet.network_analysis_core import (
    summarize_by_category,
    summarize_by_feature,
    summarize_by_category_excl,
    summarize_by_feature_excl,
    summarize_by_category_network,
    summarize_by_feature_network,
)


class TestSummarizeByCategoryDetailed:
    """Detailed tests for summarize_by_category."""
    
    def test_summarize_by_category_with_data(self):
        """Test with actual data."""
        df = pd.DataFrame({
            'Category1': ['AMR', 'AMR', 'Vir', 'Vir', 'MGE'],
            'Category2': ['Vir', 'AMR', 'AMR', 'Vir', 'AMR'],
            'Value': [0.5, 0.6, 0.7, 0.8, 0.9],
        })
        
        result = summarize_by_category(df, 'Value', ['Category1', 'Category2'])
        
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_summarize_by_category_single_category(self):
        """Test with single category."""
        df = pd.DataFrame({
            'Category1': ['AMR', 'AMR', 'AMR'],
            'Category2': ['AMR', 'AMR', 'AMR'],
            'Value': [0.5, 0.6, 0.7],
        })
        
        result = summarize_by_category(df, 'Value', ['Category1', 'Category2'])
        
        assert isinstance(result, dict)


class TestSummarizeByFeatureDetailed:
    """Detailed tests for summarize_by_feature."""
    
    def test_summarize_by_feature_with_data(self):
        """Test with actual data."""
        df = pd.DataFrame({
            'Feature1': ['Gene_A', 'Gene_B', 'Gene_C', 'Gene_A', 'Gene_B'],
            'Feature2': ['Gene_X', 'Gene_Y', 'Gene_Z', 'Gene_Y', 'Gene_X'],
            'Value': [0.5, 0.6, 0.7, 0.8, 0.9],
        })
        
        result = summarize_by_feature(df, 'Value', ['Feature1', 'Feature2'])
        
        assert isinstance(result, dict)
        assert len(result) > 0


class TestSummarizeByCategoryExclDetailed:
    """Detailed tests for summarize_by_category_excl."""
    
    def test_summarize_by_category_excl_with_data(self):
        """Test with actual data for k=2."""
        df = pd.DataFrame({
            'Category1': ['AMR', 'Vir', 'MGE'],
            'Category2': ['Vir', 'AMR', 'AMR'],
            'P_Value': [0.01, 0.02, 0.03],
        })
        
        result = summarize_by_category_excl(df, k=2)
        
        assert isinstance(result, dict)
    
    def test_summarize_by_category_excl_k3(self):
        """Test with k=3."""
        df = pd.DataFrame({
            'Category1': ['AMR', 'Vir'],
            'Category2': ['Vir', 'AMR'],
            'Category3': ['MGE', 'MGE'],
            'P_Value': [0.01, 0.02],
        })
        
        result = summarize_by_category_excl(df, k=3)
        
        assert isinstance(result, dict)


class TestSummarizeByFeatureExclDetailed:
    """Detailed tests for summarize_by_feature_excl."""
    
    def test_summarize_by_feature_excl_with_data(self):
        """Test with actual data for k=2."""
        df = pd.DataFrame({
            'Feature1': ['Gene_A', 'Gene_B', 'Gene_C'],
            'Feature2': ['Gene_X', 'Gene_Y', 'Gene_Z'],
            'P_Value': [0.01, 0.02, 0.03],
        })
        
        result = summarize_by_feature_excl(df, k=2)
        
        assert isinstance(result, dict)


class TestSummarizeByCategoryNetworkDetailed:
    """Detailed tests for summarize_by_category_network."""
    
    def test_summarize_by_category_network_with_data(self):
        """Test with actual data."""
        df = pd.DataFrame({
            'Feature': ['Gene_A', 'Gene_B', 'Gene_C', 'Gene_D'],
            'Category': ['AMR', 'AMR', 'Vir', 'MGE'],
            'Degree_Centrality': [0.5, 0.6, 0.7, 0.8],
        })
        
        result = summarize_by_category_network(df, 'Degree_Centrality')
        
        assert isinstance(result, dict)


class TestSummarizeByFeatureNetworkDetailed:
    """Detailed tests for summarize_by_feature_network."""
    
    def test_summarize_by_feature_network_with_data(self):
        """Test with actual data."""
        df = pd.DataFrame({
            'Feature': ['Gene_A', 'Gene_B', 'Gene_C'],
            'Degree_Centrality': [0.5, 0.6, 0.7],
        })
        
        result = summarize_by_feature_network(df, 'Degree_Centrality')
        
        assert isinstance(result, dict)
        assert len(result) == 3
