#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for create_interactive_table and create_section_summary functions.
"""

import pytest
import pandas as pd
import numpy as np

from strepsuis_genphennet.network_analysis_core import (
    create_interactive_table,
    create_section_summary,
)


class TestCreateInteractiveTable:
    """Test create_interactive_table function."""
    
    def test_create_interactive_table_basic(self):
        """Test basic table creation."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4.5, 5.5, 6.5],
            'C': ['x', 'y', 'z'],
        })
        
        result = create_interactive_table(df, 'test_table')
        
        assert isinstance(result, str)
        assert 'table' in result.lower()
        assert 'test_table' in result
        assert '<thead>' in result
        assert '<tbody>' in result
    
    def test_create_interactive_table_with_numeric(self):
        """Test table with numeric columns (should be rounded)."""
        df = pd.DataFrame({
            'Value': [1.123456, 2.234567, 3.345678],
        })
        
        result = create_interactive_table(df, 'numeric_table')
        
        assert isinstance(result, str)
        # Values should be rounded to 3 decimal places
        assert '1.123' in result
    
    def test_create_interactive_table_single_row(self):
        """Test table with single row."""
        df = pd.DataFrame({
            'A': [1],
            'B': [2],
        })
        
        result = create_interactive_table(df, 'single_row')
        
        assert isinstance(result, str)
        assert '<tr>' in result
    
    def test_create_interactive_table_many_columns(self):
        """Test table with many columns."""
        df = pd.DataFrame({
            f'Col_{i}': [i] for i in range(10)
        })
        
        result = create_interactive_table(df, 'many_cols')
        
        assert isinstance(result, str)
        assert 'Col_0' in result
        assert 'Col_9' in result


class TestCreateSectionSummary:
    """Test create_section_summary function."""
    
    def test_create_section_summary_basic(self):
        """Test basic section summary."""
        stats = {
            'Total': 100,
            'Significant': 50,
            'Mean': 0.5,
        }
        
        result = create_section_summary('Test Summary', stats)
        
        assert isinstance(result, str)
        assert 'Test Summary' in result
        assert 'Total' in result
        assert '100' in result
    
    def test_create_section_summary_with_categories(self):
        """Test section summary with per_category."""
        stats = {'Total': 100}
        per_category = {'AMR': 30, 'Vir': 20, 'MGE': 10}
        
        result = create_section_summary('Summary', stats, per_category=per_category)
        
        assert isinstance(result, str)
        assert 'AMR' in result or 'category' in result.lower()
    
    def test_create_section_summary_with_features(self):
        """Test section summary with per_feature."""
        stats = {'Total': 100}
        per_feature = {'Gene_A': 10, 'Gene_B': 20}
        
        result = create_section_summary('Summary', stats, per_feature=per_feature)
        
        assert isinstance(result, str)
    
    def test_create_section_summary_with_both(self):
        """Test section summary with both per_category and per_feature."""
        stats = {'Total': 100, 'Significant': 50}
        per_category = {'AMR': 30, 'Vir': 20}
        per_feature = {'Gene_A': 10, 'Gene_B': 20}
        
        result = create_section_summary('Full Summary', stats, per_category, per_feature)
        
        assert isinstance(result, str)
        assert 'Full Summary' in result
    
    def test_create_section_summary_empty_stats(self):
        """Test section summary with empty stats."""
        stats = {}
        
        result = create_section_summary('Empty Summary', stats)
        
        assert isinstance(result, str)
        assert 'Empty Summary' in result
    
    def test_create_section_summary_float_values(self):
        """Test section summary with float values."""
        stats = {
            'Mean': 0.12345,
            'Std': 0.56789,
        }
        
        result = create_section_summary('Float Summary', stats)
        
        assert isinstance(result, str)
