#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for generate_synthetic_data.py to increase coverage.
"""

import os
import pytest
import pandas as pd
import numpy as np
import tempfile

from strepsuis_genphennet.generate_synthetic_data import (
    generate_correlated_binary_features,
    generate_network_synthetic_dataset,
    save_synthetic_network_data,
    validate_synthetic_network_data,
)


class TestGenerateCorrelatedBinaryFeatures:
    """Test correlated binary feature generation."""
    
    def test_generate_basic(self):
        """Test basic generation."""
        n_samples = 100
        
        # Function returns tuple of (feature1, feature2)
        result = generate_correlated_binary_features(
            n_samples=n_samples,
            base_prevalence=0.5,
            target_phi=0.5,
            random_state=42
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert len(result[0]) == n_samples
    
    def test_generate_binary_values(self):
        """Test that values are binary."""
        feature1, feature2 = generate_correlated_binary_features(
            n_samples=50,
            base_prevalence=0.5,
            target_phi=0.3,
            random_state=42
        )
        
        assert set(np.unique(feature1)).issubset({0, 1})
        assert set(np.unique(feature2)).issubset({0, 1})
    
    def test_generate_reproducible(self):
        """Test reproducibility with same seed."""
        result1 = generate_correlated_binary_features(
            n_samples=50, base_prevalence=0.5, target_phi=0.5, random_state=42
        )
        result2 = generate_correlated_binary_features(
            n_samples=50, base_prevalence=0.5, target_phi=0.5, random_state=42
        )
        
        np.testing.assert_array_equal(result1[0], result2[0])
        np.testing.assert_array_equal(result1[1], result2[1])
    
    def test_generate_different_correlations(self):
        """Test with different correlation strengths."""
        for phi in [0.1, 0.5, 0.9]:
            result = generate_correlated_binary_features(
                n_samples=50, base_prevalence=0.5, target_phi=phi, random_state=42
            )
            assert len(result) == 2


class TestGenerateNetworkSyntheticDataset:
    """Test network synthetic dataset generation."""
    
    def test_generate_dataset_basic(self):
        """Test basic dataset generation."""
        # Function returns tuple (data_df, metadata)
        result = generate_network_synthetic_dataset()
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_generate_dataset_returns_dataframe(self):
        """Test that dataset returns DataFrame."""
        data_df, metadata = generate_network_synthetic_dataset()
        
        assert isinstance(data_df, pd.DataFrame)
    
    def test_generate_dataset_reproducible(self):
        """Test reproducibility."""
        from strepsuis_genphennet.generate_synthetic_data import SyntheticNetworkConfig
        
        config1 = SyntheticNetworkConfig(random_state=42)
        config2 = SyntheticNetworkConfig(random_state=42)
        
        data1, _ = generate_network_synthetic_dataset(config1)
        data2, _ = generate_network_synthetic_dataset(config2)
        
        pd.testing.assert_frame_equal(data1, data2)


class TestSaveSyntheticNetworkData:
    """Test saving synthetic network data."""
    
    def test_save_data_basic(self):
        """Test basic data saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_df, metadata = generate_network_synthetic_dataset()
            
            result = save_synthetic_network_data(data_df, metadata, tmpdir)
            
            # Check that result is dict
            assert isinstance(result, dict)
            
            # Check that files were created
            files = os.listdir(tmpdir)
            assert len(files) > 0
    
    def test_save_data_creates_csv(self):
        """Test that CSV files are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_df, metadata = generate_network_synthetic_dataset()
            
            result = save_synthetic_network_data(data_df, metadata, tmpdir)
            
            # Check that files were created
            csv_files = [f for f in os.listdir(tmpdir) if f.endswith('.csv')]
            assert len(csv_files) > 0


class TestValidateSyntheticNetworkData:
    """Test validation of synthetic network data."""
    
    def test_validate_valid_data(self):
        """Test validation of valid data."""
        data_df, metadata = generate_network_synthetic_dataset()
        
        result = validate_synthetic_network_data(data_df, metadata)
        
        # Should return ValidationResult or dict or bool
        assert result is not None
    
    def test_validate_with_valid_generated_data(self):
        """Test validation with valid generated data."""
        from strepsuis_genphennet.generate_synthetic_data import SyntheticNetworkConfig
        
        config = SyntheticNetworkConfig(n_strains=30, n_features=10)
        data_df, metadata = generate_network_synthetic_dataset(config)
        
        result = validate_synthetic_network_data(data_df, metadata)
        
        # Should return validation result
        assert result is not None
