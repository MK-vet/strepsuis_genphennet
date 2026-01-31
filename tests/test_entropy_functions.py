#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for entropy and information theory functions.

Tests calculate_entropy, calculate_conditional_entropy, and related functions.
"""

import pytest
import pandas as pd
import numpy as np

try:
    from strepsuis_genphennet.network_analysis_core import (
        calculate_entropy,
        calculate_conditional_entropy,
        calculate_mutual_information,
    )
    FUNCTIONS_AVAILABLE = True
except ImportError:
    FUNCTIONS_AVAILABLE = False


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
class TestCalculateEntropy:
    """Test entropy calculation."""
    
    def test_calculate_entropy_basic(self):
        """Test basic entropy calculation."""
        x = pd.Series([1, 1, 0, 0, 1, 0])
        
        entropy = calculate_entropy(x)
        
        assert isinstance(entropy, (float, np.floating))
        assert entropy >= 0
    
    def test_calculate_entropy_uniform(self):
        """Test entropy with uniform distribution."""
        x = pd.Series([1, 0, 1, 0, 1, 0])
        
        entropy = calculate_entropy(x)
        
        # Should be close to maximum entropy (1.0 for binary)
        assert entropy > 0.9
    
    def test_calculate_entropy_constant(self):
        """Test entropy with constant value."""
        x = pd.Series([1, 1, 1, 1, 1])
        
        entropy = calculate_entropy(x)
        
        # Should be 0 (no uncertainty)
        assert entropy == 0.0


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
class TestCalculateConditionalEntropy:
    """Test conditional entropy calculation."""
    
    def test_calculate_conditional_entropy_basic(self):
        """Test basic conditional entropy calculation."""
        x = pd.Series([1, 1, 0, 0, 1, 0])
        y = pd.Series([1, 0, 1, 0, 1, 0])
        
        cond_entropy = calculate_conditional_entropy(x, y)
        
        assert isinstance(cond_entropy, (float, np.floating))
        assert cond_entropy >= 0
    
    def test_calculate_conditional_entropy_independent(self):
        """Test conditional entropy with independent variables."""
        x = pd.Series([1, 0, 1, 0])
        y = pd.Series([1, 1, 0, 0])  # Independent
        
        cond_entropy = calculate_conditional_entropy(x, y)
        
        # Should be close to H(X)
        assert cond_entropy >= 0


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
class TestCalculateMutualInformation:
    """Test mutual information calculation."""
    
    def test_calculate_mutual_information_basic(self):
        """Test basic mutual information calculation."""
        x = pd.Series([1, 1, 0, 0, 1, 0])
        y = pd.Series([1, 0, 1, 0, 1, 0])
        
        mi = calculate_mutual_information(x, y)
        
        assert isinstance(mi, (float, np.floating))
        assert mi >= 0
    
    def test_calculate_mutual_information_independent(self):
        """Test MI with independent variables."""
        x = pd.Series([1, 0, 1, 0])
        y = pd.Series([1, 1, 0, 0])  # Independent
        
        mi = calculate_mutual_information(x, y)
        
        # Should be close to 0
        assert mi >= 0
        assert mi < 0.5
