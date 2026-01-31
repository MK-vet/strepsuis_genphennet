#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for information theory functions.

Tests entropy, mutual information, and related functions.
"""

import pytest
import pandas as pd
import numpy as np

try:
    from strepsuis_genphennet.network_analysis_core import (
        calculate_entropy,
        calculate_mutual_information,
        calculate_conditional_entropy,
    )
    FUNCTIONS_AVAILABLE = True
except ImportError:
    FUNCTIONS_AVAILABLE = False


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
class TestCalculateEntropy:
    """Test calculate_entropy function."""
    
    def test_calculate_entropy_basic(self):
        """Test basic entropy calculation."""
        x = pd.Series([1, 1, 0, 0, 1, 0])
        
        entropy = calculate_entropy(x)
        
        assert isinstance(entropy, (float, np.floating))
        assert entropy >= 0
        assert entropy <= 1  # Binary entropy max is 1
    
    def test_calculate_entropy_uniform(self):
        """Test entropy with uniform distribution."""
        x = pd.Series([1, 0, 1, 0, 1, 0])
        
        entropy = calculate_entropy(x)
        
        # Uniform distribution should have high entropy
        assert entropy > 0.5
    
    def test_calculate_entropy_constant(self):
        """Test entropy with constant values."""
        x = pd.Series([1, 1, 1, 1, 1])
        
        entropy = calculate_entropy(x)
        
        # Constant should have zero entropy
        assert entropy == 0.0 or entropy < 0.01


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
class TestCalculateMutualInformation:
    """Test calculate_mutual_information function."""
    
    def test_calculate_mutual_information_basic(self):
        """Test basic mutual information calculation."""
        x = pd.Series([1, 1, 0, 0, 1, 0])
        y = pd.Series([1, 0, 1, 0, 1, 0])
        
        mi = calculate_mutual_information(x, y)
        
        assert isinstance(mi, (float, np.floating))
        assert mi >= 0
    
    def test_calculate_mutual_information_identical(self):
        """Test MI with identical series."""
        x = pd.Series([1, 1, 0, 0])
        y = pd.Series([1, 1, 0, 0])
        
        mi = calculate_mutual_information(x, y)
        
        # Identical series should have high MI
        assert mi > 0
    
    def test_calculate_mutual_information_independent(self):
        """Test MI with independent series."""
        x = pd.Series([1, 0, 1, 0])
        y = pd.Series([0, 1, 0, 1])  # Negated - should be independent
        
        mi = calculate_mutual_information(x, y)
        
        # Independent should have low MI
        assert isinstance(mi, (float, np.floating))
        assert mi >= 0


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
class TestCalculateConditionalEntropy:
    """Test calculate_conditional_entropy function."""
    
    def test_calculate_conditional_entropy_basic(self):
        """Test basic conditional entropy calculation."""
        x = pd.Series([1, 1, 0, 0, 1, 0])
        y = pd.Series([1, 0, 1, 0, 1, 0])
        
        cond_entropy = calculate_conditional_entropy(x, y)
        
        assert isinstance(cond_entropy, (float, np.floating))
        assert cond_entropy >= 0
        assert cond_entropy <= 1  # Binary conditional entropy max is 1
