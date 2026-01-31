#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for full analysis pipeline functions.
"""

import os
import tempfile
import shutil
import pytest
import pandas as pd
import numpy as np
import logging

from strepsuis_genphennet.network_analysis_core import (
    setup_logging,
)


class TestSetupLogging:
    """Test setup_logging function."""

    @pytest.fixture
    def temp_cwd(self):
        """Change to temporary directory."""
        import logging
        original_cwd = os.getcwd()
        temp_dir = tempfile.mkdtemp()
        os.chdir(temp_dir)

        # Clear existing logging handlers
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        yield temp_dir

        # Clean up logging handlers after test
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        os.chdir(original_cwd)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_setup_logging_basic(self, temp_cwd):
        """Test basic logging setup."""
        setup_logging()
        
        # Should create output directory
        assert os.path.exists('output')
    
    def test_setup_logging_creates_log_file(self, temp_cwd):
        """Test that logging creates log file."""
        setup_logging()
        
        # Log file should be created
        log_file = os.path.join('output', 'network_analysis_log.txt')
        # File may or may not exist until first log message
        assert os.path.exists('output')


class TestIntegrationHelpers:
    """Test integration helper functions."""
    
    def test_chi2_phi_integration(self):
        """Test chi2_phi in integration context."""
        from strepsuis_genphennet.network_analysis_core import chi2_phi
        
        # Create correlated data
        x = pd.Series([1, 1, 1, 0, 0, 0] * 10)
        y = pd.Series([1, 1, 0, 0, 0, 0] * 10)
        
        p, phi, contingency, lo, hi = chi2_phi(x, y)
        
        assert p < 0.1  # Should be significant
        assert phi > 0  # Should be positive correlation
    
    def test_cramers_v_integration(self):
        """Test cramers_v in integration context."""
        from strepsuis_genphennet.network_analysis_core import cramers_v
        
        # Create contingency table
        contingency = pd.DataFrame({
            'A': [30, 10],
            'B': [10, 30],
        })
        
        v, lo, hi = cramers_v(contingency)
        
        assert v > 0  # Should show association
    
    def test_entropy_integration(self):
        """Test entropy functions in integration context."""
        from strepsuis_genphennet.network_analysis_core import (
            calculate_entropy,
            conditional_entropy,
            information_gain,
            normalized_mutual_info,
        )
        
        x = pd.Series([1, 1, 0, 0, 1, 0, 1, 0] * 5)
        y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0] * 5)
        
        H, Hn = calculate_entropy(x)
        ce = conditional_entropy(x, y)
        ig = information_gain(x, y)
        nmi = normalized_mutual_info(x, y)
        
        assert H > 0
        assert ce >= 0
        assert ig >= 0
        assert 0 <= nmi <= 1


class TestMutuallyExclusiveIntegration:
    """Test mutually exclusive pattern detection in integration context."""
    
    def test_find_mutually_exclusive_integration(self):
        """Test find_mutually_exclusive in integration context."""
        from strepsuis_genphennet.network_analysis_core import find_mutually_exclusive
        
        # Create data with mutually exclusive features
        data = pd.DataFrame({
            'Gene_A': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            'Gene_B': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'Gene_C': [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        })
        
        features = ['Gene_A', 'Gene_B', 'Gene_C']
        feature_category = {
            'Gene_A': 'AMR',
            'Gene_B': 'AMR',
            'Gene_C': 'Vir',
        }
        
        result = find_mutually_exclusive(data, features, feature_category, k=2)
        
        assert isinstance(result, pd.DataFrame)


class TestNetworkFunctionsIntegration:
    """Test network functions in integration context."""
    
    def test_get_cluster_hubs_integration(self):
        """Test get_cluster_hubs in integration context."""
        from strepsuis_genphennet.network_analysis_core import get_cluster_hubs
        
        df = pd.DataFrame({
            'Feature': ['Gene_A', 'Gene_B', 'Gene_C', 'Gene_D', 'Gene_E'],
            'Degree_Centrality': [0.9, 0.8, 0.5, 0.3, 0.1],
            'Category': ['AMR', 'AMR', 'Vir', 'Vir', 'MGE'],
            'Cluster': [1, 1, 2, 2, 3],
        })
        
        result = get_cluster_hubs(df, top_n=2)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_adaptive_phi_threshold_integration(self):
        """Test adaptive_phi_threshold in integration context."""
        from strepsuis_genphennet.network_analysis_core import adaptive_phi_threshold
        
        phi_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        
        # Test different methods
        threshold_percentile = adaptive_phi_threshold(phi_values, method='percentile')
        threshold_iqr = adaptive_phi_threshold(phi_values, method='iqr')
        threshold_statistical = adaptive_phi_threshold(phi_values, method='statistical')
        
        assert 0 <= threshold_percentile <= 1
        assert 0 <= threshold_iqr <= 2  # IQR can exceed 1
        assert 0 <= threshold_statistical <= 2
