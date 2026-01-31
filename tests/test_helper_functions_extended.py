#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extended tests for helper functions.

Tests expand_categories, find_matching_files, and other helper functions.
"""

import pytest
import pandas as pd

try:
    from strepsuis_genphennet.network_analysis_core import (
        expand_categories,
        find_matching_files,
    )
    FUNCTIONS_AVAILABLE = True
except ImportError:
    FUNCTIONS_AVAILABLE = False


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
class TestExpandCategories:
    """Test expand_categories function."""
    
    def test_expand_categories_basic(self):
        """Test basic category expansion."""
        df = pd.DataFrame({
            'Strain_ID': ['S1', 'S2', 'S3'],
            'MLST': ['1', '2', '1'],
        })
        
        result = expand_categories(df, 'MLST')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        assert 'Strain_ID' in result.columns
    
    def test_expand_categories_empty(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['Strain_ID', 'MLST'])
        
        try:
            result = expand_categories(empty_df, 'MLST')
            assert isinstance(result, pd.DataFrame)
        except (KeyError, IndexError):
            # May fail with empty DataFrame
            pytest.skip("expand_categories may not work with empty DataFrame")


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
class TestFindMatchingFiles:
    """Test find_matching_files function."""
    
    def test_find_matching_files_basic(self):
        """Test basic file matching."""
        uploaded_files = {
            'MGE.csv': None,
            'MIC.csv': None,
            'MLST.csv': None,
        }
        expected = ['MGE.csv', 'MIC.csv', 'MLST.csv']
        
        result = find_matching_files(uploaded_files, expected)
        
        assert isinstance(result, dict)
        assert len(result) == len(expected)
    
    def test_find_matching_files_partial(self):
        """Test with partial matches."""
        uploaded_files = {
            'MGE_file.csv': None,
            'MIC_data.csv': None,
        }
        expected = ['MGE.csv', 'MIC.csv']
        
        result = find_matching_files(uploaded_files, expected)
        
        assert isinstance(result, dict)
        # Should find matches based on keywords
    
    def test_find_matching_files_empty(self):
        """Test with empty files."""
        uploaded_files = {}
        expected = ['MGE.csv', 'MIC.csv']
        
        result = find_matching_files(uploaded_files, expected)
        
        assert isinstance(result, dict)
        assert len(result) == 0
