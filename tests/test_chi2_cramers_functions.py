#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for chi2_phi, cramers_v, and related statistical functions.
"""

import pytest
import pandas as pd
import numpy as np

from strepsuis_genphennet.network_analysis_core import (
    chi2_phi,
    cramers_v,
    calculate_entropy,
    conditional_entropy,
    information_gain,
    normalized_mutual_info,
)


class TestChi2Phi:
    """Test chi2_phi function."""
    
    def test_chi2_phi_basic_2x2(self):
        """Test basic 2x2 contingency table."""
        x = pd.Series([1, 1, 0, 0, 1, 0, 1, 0, 1, 0] * 5)
        y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 0, 1] * 5)
        
        p, phi, contingency, lo, hi = chi2_phi(x, y)
        
        assert isinstance(p, float)
        assert isinstance(phi, float)
        assert isinstance(contingency, pd.DataFrame)
        assert 0 <= p <= 1
        assert -1 <= phi <= 1
    
    def test_chi2_phi_perfect_correlation(self):
        """Test with perfect correlation."""
        x = pd.Series([1, 1, 1, 0, 0, 0] * 10)
        y = pd.Series([1, 1, 1, 0, 0, 0] * 10)
        
        p, phi, contingency, lo, hi = chi2_phi(x, y)
        
        assert phi > 0.5  # Strong positive correlation
    
    def test_chi2_phi_no_correlation(self):
        """Test with no correlation."""
        np.random.seed(42)
        x = pd.Series(np.random.binomial(1, 0.5, 100))
        y = pd.Series(np.random.binomial(1, 0.5, 100))
        
        p, phi, contingency, lo, hi = chi2_phi(x, y)
        
        assert isinstance(p, float)
        assert isinstance(phi, float)
    
    def test_chi2_phi_small_sample(self):
        """Test with small sample (Fisher exact)."""
        x = pd.Series([1, 1, 0, 0, 1, 0, 1, 0, 1, 0])
        y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 0, 1])
        
        p, phi, contingency, lo, hi = chi2_phi(x, y)
        
        assert isinstance(p, float)
        assert 0 <= p <= 1
    
    def test_chi2_phi_very_small_sample(self):
        """Test with very small sample."""
        x = pd.Series([1, 0, 1])
        y = pd.Series([1, 0, 0])
        
        p, phi, contingency, lo, hi = chi2_phi(x, y)
        
        assert isinstance(p, float)


class TestCramersV:
    """Test cramers_v function."""
    
    def test_cramers_v_basic(self):
        """Test basic Cramer's V calculation."""
        contingency = pd.DataFrame({
            'A': [10, 5, 3],
            'B': [5, 15, 8],
            'C': [3, 8, 20],
        })
        
        v, lo, hi = cramers_v(contingency)
        
        assert isinstance(v, float)
        assert 0 <= v <= 1
    
    def test_cramers_v_2x2(self):
        """Test 2x2 contingency table."""
        contingency = pd.DataFrame({
            'A': [20, 10],
            'B': [10, 30],
        })
        
        v, lo, hi = cramers_v(contingency)
        
        assert isinstance(v, float)
        assert 0 <= v <= 1
    
    def test_cramers_v_small_table(self):
        """Test with small table (r < 2 or k < 2)."""
        contingency = pd.DataFrame({
            'A': [10],
        })
        
        v, lo, hi = cramers_v(contingency)
        
        assert v == 0.0
    
    def test_cramers_v_small_n(self):
        """Test with small n."""
        contingency = pd.DataFrame({
            'A': [1, 1],
            'B': [1, 0],
        })
        
        v, lo, hi = cramers_v(contingency)
        
        assert v == 0.0


class TestCalculateEntropy:
    """Test calculate_entropy function."""
    
    def test_calculate_entropy_uniform(self):
        """Test with uniform distribution."""
        series = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        
        H, Hn = calculate_entropy(series)
        
        assert isinstance(H, float)
        assert isinstance(Hn, float)
        assert H > 0
    
    def test_calculate_entropy_deterministic(self):
        """Test with deterministic series."""
        series = pd.Series([1, 1, 1, 1, 1])
        
        H, Hn = calculate_entropy(series)
        
        assert H == 0.0
        assert Hn == 0.0
    
    def test_calculate_entropy_single_value(self):
        """Test with single value."""
        series = pd.Series([1])
        
        H, Hn = calculate_entropy(series)
        
        assert H == 0.0


class TestConditionalEntropy:
    """Test conditional_entropy function."""
    
    def test_conditional_entropy_basic(self):
        """Test basic conditional entropy."""
        x = pd.Series([1, 1, 0, 0, 1, 0, 1, 0] * 5)
        y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0] * 5)
        
        ce = conditional_entropy(x, y)
        
        assert isinstance(ce, float)
        assert ce >= 0
    
    def test_conditional_entropy_perfect_prediction(self):
        """Test with perfect prediction."""
        x = pd.Series([1, 1, 0, 0, 1, 1, 0, 0])
        y = pd.Series([1, 1, 0, 0, 1, 1, 0, 0])
        
        ce = conditional_entropy(x, y)
        
        assert ce == 0.0


class TestInformationGain:
    """Test information_gain function."""
    
    def test_information_gain_basic(self):
        """Test basic information gain."""
        x = pd.Series([1, 1, 0, 0, 1, 0, 1, 0] * 5)
        y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0] * 5)
        
        ig = information_gain(x, y)
        
        assert isinstance(ig, float)
        assert ig >= 0
    
    def test_information_gain_no_gain(self):
        """Test with no information gain."""
        np.random.seed(42)
        x = pd.Series(np.random.binomial(1, 0.5, 100))
        y = pd.Series(np.random.binomial(1, 0.5, 100))
        
        ig = information_gain(x, y)
        
        assert isinstance(ig, float)
        assert ig >= 0


class TestNormalizedMutualInfo:
    """Test normalized_mutual_info function."""
    
    def test_normalized_mutual_info_basic(self):
        """Test basic normalized mutual information."""
        x = pd.Series([1, 1, 0, 0, 1, 0, 1, 0] * 5)
        y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0] * 5)
        
        nmi = normalized_mutual_info(x, y)
        
        assert isinstance(nmi, float)
        assert 0 <= nmi <= 1
    
    def test_normalized_mutual_info_perfect(self):
        """Test with perfect mutual information."""
        x = pd.Series([1, 1, 0, 0, 1, 1, 0, 0])
        y = pd.Series([1, 1, 0, 0, 1, 1, 0, 0])
        
        nmi = normalized_mutual_info(x, y)
        
        assert nmi > 0.5
    
    def test_normalized_mutual_info_deterministic(self):
        """Test with deterministic series."""
        x = pd.Series([1, 1, 1, 1, 1])
        y = pd.Series([0, 0, 0, 0, 0])
        
        nmi = normalized_mutual_info(x, y)
        
        assert nmi == 0.0
