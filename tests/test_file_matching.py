#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for find_matching_files and expand_categories functions.
"""

import pytest
import pandas as pd
import numpy as np

from strepsuis_genphennet.network_analysis_core import (
    find_matching_files,
    expand_categories,
    get_centrality,
)


class TestFindMatchingFiles:
    """Test find_matching_files function."""
    
    def test_find_matching_files_exact_match(self):
        """Test with exact file name matches."""
        uploaded_files = {
            'MGE.csv': 'content1',
            'MIC.csv': 'content2',
            'MLST.csv': 'content3',
        }
        expected_files = ['MGE.csv', 'MIC.csv', 'MLST.csv']
        
        result = find_matching_files(uploaded_files, expected_files)
        
        assert isinstance(result, dict)
        assert len(result) == 3
        assert 'MGE.csv' in result
    
    def test_find_matching_files_partial_match(self):
        """Test with partial file name matches."""
        uploaded_files = {
            'MGE (1).csv': 'content1',
            'MIC_data.csv': 'content2',
        }
        expected_files = ['MGE.csv', 'MIC.csv']
        
        result = find_matching_files(uploaded_files, expected_files)
        
        assert isinstance(result, dict)
    
    def test_find_matching_files_no_match(self):
        """Test with no matching files."""
        uploaded_files = {
            'other_file.csv': 'content1',
        }
        expected_files = ['MGE.csv', 'MIC.csv']
        
        result = find_matching_files(uploaded_files, expected_files)
        
        assert isinstance(result, dict)
    
    def test_find_matching_files_empty_uploaded(self):
        """Test with empty uploaded files."""
        uploaded_files = {}
        expected_files = ['MGE.csv', 'MIC.csv']
        
        result = find_matching_files(uploaded_files, expected_files)
        
        assert isinstance(result, dict)
        assert len(result) == 0
    
    def test_find_matching_files_empty_expected(self):
        """Test with empty expected files."""
        uploaded_files = {'MGE.csv': 'content'}
        expected_files = []
        
        result = find_matching_files(uploaded_files, expected_files)
        
        assert isinstance(result, dict)
        assert len(result) == 0


class TestExpandCategories:
    """Test expand_categories function."""
    
    def test_expand_categories_mlst(self):
        """Test category expansion for MLST."""
        df = pd.DataFrame({
            'Strain_ID': ['S1', 'S2', 'S3', 'S4'],
            'MLST': ['1', '2', '1', '3'],
        })
        
        result = expand_categories(df, 'MLST')
        
        assert isinstance(result, pd.DataFrame)
        assert 'Strain_ID' in result.columns
        assert len(result) == 4
    
    def test_expand_categories_serotype(self):
        """Test category expansion for Serotype."""
        df = pd.DataFrame({
            'Strain_ID': ['S1', 'S2', 'S3'],
            'Serotype': ['A', 'B', 'A'],
        })
        
        result = expand_categories(df, 'Serotype')
        
        assert isinstance(result, pd.DataFrame)
        assert 'Strain_ID' in result.columns
    
    def test_expand_categories_numeric_mlst(self):
        """Test category expansion with numeric MLST values."""
        df = pd.DataFrame({
            'Strain_ID': ['S1', 'S2', 'S3'],
            'MLST': [1.0, 2.0, 1.0],
        })
        
        result = expand_categories(df, 'MLST')
        
        assert isinstance(result, pd.DataFrame)
        # Should remove trailing .0 from MLST values


class TestGetCentrality:
    """Test get_centrality function."""
    
    def test_get_centrality_basic(self):
        """Test basic centrality retrieval."""
        centrality_dict = {
            'A': 0.5,
            'B': 0.3,
            'C': 0.8,
        }
        
        result = get_centrality(centrality_dict)
        
        assert isinstance(result, dict)
        assert result == centrality_dict
    
    def test_get_centrality_empty(self):
        """Test with empty dictionary."""
        centrality_dict = {}
        
        result = get_centrality(centrality_dict)
        
        assert isinstance(result, dict)
        assert len(result) == 0
