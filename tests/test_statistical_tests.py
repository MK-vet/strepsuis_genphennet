#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for perform_statistical_tests function.

Tests the main statistical testing function.
"""

import pytest
import pandas as pd
import numpy as np

try:
    from strepsuis_genphennet.network_analysis_core import (
        perform_statistical_tests,
    )
    FUNCTIONS_AVAILABLE = True
except ImportError:
    FUNCTIONS_AVAILABLE = False


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
class TestPerformStatisticalTests:
    """Test perform_statistical_tests function."""
    
    @pytest.fixture
    def sample_gene_data(self):
        """Create sample gene data."""
        np.random.seed(42)
        return pd.DataFrame({
            'Gene_A': np.random.binomial(1, 0.5, 30),
            'Gene_B': np.random.binomial(1, 0.3, 30),
            'Gene_C': np.random.binomial(1, 0.7, 30),
        })
    
    @pytest.fixture
    def sample_phenotype_data(self):
        """Create sample phenotype data."""
        np.random.seed(42)
        return pd.DataFrame({
            'Pheno_X': np.random.binomial(1, 0.4, 30),
            'Pheno_Y': np.random.binomial(1, 0.6, 30),
        })
    
    def test_perform_statistical_tests_basic(self, sample_gene_data, sample_phenotype_data):
        """Test basic statistical tests."""
        result = perform_statistical_tests(sample_gene_data, sample_phenotype_data)
        
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert 'Feature1' in result.columns or 'Feature' in result.columns
            assert 'P_Value' in result.columns or 'p_value' in result.columns
    
    def test_perform_statistical_tests_empty(self):
        """Test with empty data."""
        empty_gene = pd.DataFrame()
        empty_pheno = pd.DataFrame()
        
        result = perform_statistical_tests(empty_gene, empty_pheno)
        
        assert isinstance(result, pd.DataFrame)
    
    def test_perform_statistical_tests_single_feature(self, sample_gene_data):
        """Test with single phenotype feature."""
        single_pheno = pd.DataFrame({
            'Pheno_X': np.random.binomial(1, 0.5, 30),
        })
        
        result = perform_statistical_tests(sample_gene_data, single_pheno)
        
        assert isinstance(result, pd.DataFrame)
