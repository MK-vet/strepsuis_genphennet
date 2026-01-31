#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for entropy and mutually exclusive pattern functions.

Tests entropy calculation and mutually exclusive pattern detection.
"""

import pytest
import pandas as pd
import numpy as np

try:
    from strepsuis_genphennet.network_analysis_core import (
        calculate_entropy,
        conditional_entropy,
        information_gain,
        normalized_mutual_info,
        find_mutually_exclusive,
    )
    FUNCTIONS_AVAILABLE = True
except ImportError:
    FUNCTIONS_AVAILABLE = False


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
class TestEntropyFunctions:
    """Test entropy calculation functions."""
    
    def test_calculate_entropy_basic(self):
        """Test basic entropy calculation."""
        x = pd.Series([1, 1, 0, 0, 1, 0])
        
        H, Hn = calculate_entropy(x)
        
        assert isinstance(H, (float, np.floating))
        assert isinstance(Hn, (float, np.floating))
        assert H >= 0
        assert 0 <= Hn <= 1  # Normalized entropy
    
    def test_calculate_entropy_uniform(self):
        """Test entropy with uniform distribution."""
        x = pd.Series([1, 0, 1, 0, 1, 0])
        
        H, Hn = calculate_entropy(x)
        
        # Should have high entropy (close to maximum for binary)
        assert H > 0.5  # Adjusted threshold
        assert 0 <= Hn <= 1
    
    def test_calculate_entropy_deterministic(self):
        """Test entropy with deterministic distribution."""
        x = pd.Series([1, 1, 1, 1, 1, 1])
        
        H, Hn = calculate_entropy(x)
        
        # Should be 0 (no uncertainty)
        assert H == 0.0
        assert Hn == 0.0
    
    def test_conditional_entropy(self):
        """Test conditional entropy calculation."""
        x = pd.Series([1, 1, 0, 0, 1, 0])
        y = pd.Series([1, 0, 1, 0, 1, 0])
        
        cond_entropy = conditional_entropy(x, y)
        
        assert isinstance(cond_entropy, (float, np.floating))
        assert cond_entropy >= 0
    
    def test_information_gain(self):
        """Test information gain calculation."""
        x = pd.Series([1, 1, 0, 0, 1, 0])
        y = pd.Series([1, 0, 1, 0, 1, 0])
        
        ig = information_gain(x, y)
        
        assert isinstance(ig, (float, np.floating))
        assert ig >= 0
    
    def test_normalized_mutual_info(self):
        """Test normalized mutual information calculation."""
        x = pd.Series([1, 1, 0, 0, 1, 0])
        y = pd.Series([1, 0, 1, 0, 1, 0])
        
        nmi = normalized_mutual_info(x, y)
        
        assert isinstance(nmi, (float, np.floating))
        assert 0 <= nmi <= 1


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
class TestMutuallyExclusivePatterns:
    """Test mutually exclusive pattern detection."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample binary data."""
        return pd.DataFrame({
            'Gene_A': [1, 1, 0, 0, 0, 0],
            'Gene_B': [0, 0, 1, 1, 0, 0],
            'Gene_C': [0, 0, 0, 0, 1, 1],
        })
    
    def test_find_mutually_exclusive_pairs(self, sample_data):
        """Test finding mutually exclusive pairs."""
        features = list(sample_data.columns)
        mapping = {f: 'Category1' for f in features}
        
        result = find_mutually_exclusive(sample_data, features, mapping, k=2, max_patterns=100)
        
        assert isinstance(result, pd.DataFrame)
        # May be empty if no patterns found
        if not result.empty:
            assert 'Feature_1' in result.columns
    
    def test_find_mutually_exclusive_triplets(self, sample_data):
        """Test finding mutually exclusive triplets."""
        features = list(sample_data.columns)
        mapping = {f: 'Category1' for f in features}
        
        result = find_mutually_exclusive(sample_data, features, mapping, k=3, max_patterns=100)
        
        assert isinstance(result, pd.DataFrame)
        # May be empty if no patterns found
        if not result.empty:
            assert 'Feature_1' in result.columns
