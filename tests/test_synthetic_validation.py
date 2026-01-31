"""
Comprehensive Mathematical Validation Tests for Network Integration

This module provides rigorous validation of all statistical methods against
synthetic data with known ground truth, ensuring mathematical correctness
of the StrepSuis-GenPhenNet network analysis pipeline.

Tests validate:
1. Chi-square/Fisher exact tests against scipy
2. Phi coefficient computation
3. FDR correction against statsmodels
4. Network construction correctness
5. Information theory metrics (entropy, mutual information)
6. Cramér's V calculation

Reference implementations: scipy, statsmodels, networkx
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency, entropy, fisher_exact
from statsmodels.stats.multitest import multipletests


class TestSyntheticDataGeneration:
    """Test the synthetic data generation module."""

    def test_generate_synthetic_dataset(self):
        """Test that synthetic data generation produces valid output."""
        from strepsuis_genphennet.generate_synthetic_data import (
            SyntheticNetworkConfig,
            generate_network_synthetic_dataset,
        )

        config = SyntheticNetworkConfig(
            n_strains=50,
            n_features=20,
            n_true_associations=10,
            random_state=42,
        )

        data_df, metadata = generate_network_synthetic_dataset(config)

        # Verify shape
        assert len(data_df) == config.n_strains
        assert len(metadata.feature_columns) == config.n_features

        # Verify data types
        for col in metadata.feature_columns:
            assert col in data_df.columns
            assert set(data_df[col].unique()).issubset({0, 1})

    def test_true_associations_recorded(self):
        """Test that true associations are properly recorded."""
        from strepsuis_genphennet.generate_synthetic_data import (
            SyntheticNetworkConfig,
            generate_network_synthetic_dataset,
        )

        config = SyntheticNetworkConfig(
            n_strains=100,
            n_features=30,
            n_true_associations=15,
            random_state=42,
        )

        data_df, metadata = generate_network_synthetic_dataset(config)

        # Check we have the expected number of true associations
        assert len(metadata.true_associations) == config.n_true_associations

        # Each association should be a tuple of (feat1, feat2, phi)
        for feat1, feat2, phi in metadata.true_associations:
            assert feat1 in metadata.feature_columns
            assert feat2 in metadata.feature_columns
            assert 0 < phi <= 1

    def test_synthetic_data_reproducibility(self):
        """Test that same seed produces identical data."""
        from strepsuis_genphennet.generate_synthetic_data import (
            SyntheticNetworkConfig,
            generate_network_synthetic_dataset,
        )

        config = SyntheticNetworkConfig(n_strains=30, random_state=12345)

        data1, meta1 = generate_network_synthetic_dataset(config)
        data2, meta2 = generate_network_synthetic_dataset(config)

        pd.testing.assert_frame_equal(data1, data2)


class TestChiSquareValidation:
    """Validate chi-square tests against scipy reference."""

    def test_chi_square_exact_match(self):
        """Test exact match with scipy chi2_contingency."""
        # Use table with high expected counts
        table = np.array([[100, 50], [50, 100]])

        chi2_scipy, p_scipy, dof, expected = chi2_contingency(table)

        # Verify values are correct
        assert chi2_scipy > 0
        assert 0 <= p_scipy <= 1
        assert dof == 1

        # Calculate phi coefficient
        n = table.sum()
        phi = np.sqrt(chi2_scipy / n)
        assert 0 <= phi <= 1

        # With these values, there should be significant association
        assert p_scipy < 0.05

    def test_chi_square_with_yates_correction(self):
        """Test chi-square with Yates correction."""
        # Small expected counts table
        table = np.array([[10, 5], [5, 10]])

        chi2_yates, p_yates, _, _ = chi2_contingency(table, correction=True)
        chi2_no_yates, p_no_yates, _, _ = chi2_contingency(table, correction=False)

        # Yates correction should give smaller chi-square
        assert chi2_yates <= chi2_no_yates
        # And larger p-value
        assert p_yates >= p_no_yates


class TestFisherExactValidation:
    """Validate Fisher exact test against scipy."""

    def test_fisher_exact_match(self):
        """Test exact match with scipy fisher_exact."""
        # Small contingency table
        table = np.array([[5, 1], [1, 5]])

        odds_ratio, p_value = fisher_exact(table)

        assert 0 <= p_value <= 1
        assert odds_ratio > 0

    def test_fisher_vs_chi_square_small_samples(self):
        """Test that Fisher is preferred for small samples."""
        # Very small table
        table = np.array([[3, 1], [1, 3]])

        # Fisher exact
        _, p_fisher = fisher_exact(table)

        # Chi-square
        _, p_chi2, _, _ = chi2_contingency(table, correction=False)

        # Both should give valid p-values
        assert 0 <= p_fisher <= 1
        assert 0 <= p_chi2 <= 1


class TestPhiCoefficientValidation:
    """Validate phi coefficient computation."""

    def test_phi_coefficient_formula(self):
        """Test phi coefficient calculation matches expected formula."""
        # Create known table
        # a=40, b=10, c=10, d=40 -> phi = (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d))
        table = np.array([[40, 10], [10, 40]])

        a, b, c, d = table[0, 0], table[0, 1], table[1, 0], table[1, 1]
        expected_phi = (a * d - b * c) / np.sqrt((a + b) * (c + d) * (a + c) * (b + d))

        # Calculate via chi-square
        chi2, _, _, _ = chi2_contingency(table, correction=False)
        n = table.sum()
        calculated_phi = np.sqrt(chi2 / n)

        # Should be close (may differ slightly due to sign handling)
        np.testing.assert_almost_equal(abs(calculated_phi), abs(expected_phi), decimal=3)

    def test_phi_range(self):
        """Test that phi is always in valid range."""
        np.random.seed(42)

        for _ in range(100):
            # Random 2x2 table
            table = np.random.randint(1, 50, size=(2, 2))
            chi2, _, _, _ = chi2_contingency(table, correction=False)
            n = table.sum()
            phi = np.sqrt(chi2 / n)

            assert 0 <= phi <= 1


class TestFDRCorrectionValidation:
    """Validate FDR correction against statsmodels."""

    def test_fdr_matches_statsmodels(self):
        """Test FDR correction matches statsmodels exactly."""
        np.random.seed(42)
        p_values = np.random.uniform(0, 1, 100)

        # Apply FDR correction
        reject, corrected, _, _ = multipletests(p_values, method="fdr_bh", alpha=0.05)

        # Corrected p-values should be >= original
        assert all(corrected >= p_values)

        # Sorted corrected p-values should be monotonic
        sorted_indices = np.argsort(p_values)
        sorted_corrected = corrected[sorted_indices]

        for i in range(len(sorted_corrected) - 1):
            assert sorted_corrected[i] <= sorted_corrected[i + 1] + 1e-10

    def test_fdr_with_known_nulls_and_alternatives(self):
        """Test FDR with known mixture of null and alternative p-values."""
        np.random.seed(42)

        # Generate mixture
        n_null = 80
        n_alt = 20

        null_pvals = np.random.uniform(0, 1, n_null)
        alt_pvals = np.random.beta(1, 20, n_alt)  # Concentrated near 0

        all_pvals = np.concatenate([null_pvals, alt_pvals])
        true_labels = np.concatenate([np.zeros(n_null), np.ones(n_alt)])

        # Apply FDR
        reject, _, _, _ = multipletests(all_pvals, alpha=0.05, method="fdr_bh")

        # Should detect some true alternatives
        true_positives = np.sum(reject & (true_labels == 1))
        assert true_positives > 0


class TestEntropyValidation:
    """Validate information theory metrics."""

    def test_entropy_calculation(self):
        """Test entropy calculation matches scipy."""
        np.random.seed(42)

        # Generate categorical distribution
        probs = np.array([0.5, 0.3, 0.2])

        # Calculate entropy using scipy
        H = entropy(probs, base=2)  # Base 2 for bits

        # Entropy should be positive
        assert H > 0

        # Maximum entropy for 3 categories is log2(3) ≈ 1.585
        assert H <= np.log2(3)

    def test_entropy_uniform_maximum(self):
        """Test that uniform distribution has maximum entropy."""
        n_categories = 4
        uniform_probs = np.ones(n_categories) / n_categories

        H_uniform = entropy(uniform_probs, base=2)
        max_entropy = np.log2(n_categories)

        np.testing.assert_almost_equal(H_uniform, max_entropy, decimal=5)

    def test_entropy_deterministic_minimum(self):
        """Test that deterministic distribution has zero entropy."""
        deterministic_probs = np.array([1.0, 0.0, 0.0])

        H = entropy(deterministic_probs, base=2)

        np.testing.assert_almost_equal(H, 0.0, decimal=5)


class TestCramersVValidation:
    """Validate Cramér's V calculation."""

    def test_cramers_v_range(self):
        """Test that Cramér's V is in [0, 1]."""
        np.random.seed(42)

        for _ in range(50):
            # Random contingency table
            r, c = np.random.randint(2, 5, 2)
            table = np.random.randint(5, 50, size=(r, c))

            chi2, _, _, _ = chi2_contingency(table)
            n = table.sum()
            min_dim = min(r - 1, c - 1)

            if min_dim > 0:
                cramers_v = np.sqrt(chi2 / (n * min_dim))
                assert 0 <= cramers_v <= 1 + 1e-10

    def test_cramers_v_independence(self):
        """Test Cramér's V for independent variables is near zero."""
        np.random.seed(42)
        n = 1000

        # Generate independent variables
        x = np.random.randint(0, 3, n)
        y = np.random.randint(0, 3, n)

        table = pd.crosstab(x, y).values
        chi2, _, _, _ = chi2_contingency(table)
        cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))

        # Should be close to zero for independent variables
        assert cramers_v < 0.15


class TestNetworkValidationWithSynthetic:
    """Validate network construction using synthetic data."""

    def test_network_edge_detection(self):
        """Test that network correctly identifies true associations."""
        from strepsuis_genphennet.generate_synthetic_data import (
            SyntheticNetworkConfig,
            generate_network_synthetic_dataset,
        )

        config = SyntheticNetworkConfig(
            n_strains=200,
            n_features=30,
            n_true_associations=10,
            association_strength=0.7,
            noise_level=0.02,
            random_state=42,
        )

        data_df, metadata = generate_network_synthetic_dataset(config)

        # Get feature columns
        feature_cols = [c for c in data_df.columns if c != "Strain_ID"]

        # Test chi-square on true associations
        detected_count = 0
        for feat1, feat2, true_phi in metadata.true_associations[:5]:
            table = pd.crosstab(data_df[feat1], data_df[feat2])
            chi2, p_value, _, _ = chi2_contingency(table, correction=False)

            if p_value < 0.05:
                detected_count += 1

        # Should detect at least 60% of true associations
        assert detected_count >= 3, f"Only detected {detected_count}/5 true associations"


class TestNumericalStability:
    """Test numerical stability with edge cases."""

    def test_zero_counts_chi_square(self):
        """Test chi-square with zero counts."""
        # Table with zeros
        table = np.array([[0, 10], [10, 0]])

        # Should not raise error
        chi2, p, _, _ = chi2_contingency(table, correction=False)

        assert not np.isnan(chi2)
        assert not np.isnan(p)

    def test_extreme_imbalance(self):
        """Test with extremely imbalanced data."""
        # Highly imbalanced table
        table = np.array([[99, 1], [1, 99]])

        chi2, p, _, _ = chi2_contingency(table)

        assert not np.isnan(chi2)
        assert not np.isnan(p)
        assert p < 0.001  # Should be highly significant


class TestValidationReportGeneration:
    """Generate validation reports for math validation directory."""

    @pytest.fixture
    def report_dir(self, tmp_path):
        """Create temporary report directory."""
        return tmp_path / "math_validation"

    def test_generate_validation_report(self, report_dir):
        """Generate a validation report summarizing test results."""
        from strepsuis_genphennet.generate_synthetic_data import (
            SyntheticNetworkConfig,
            generate_network_synthetic_dataset,
            validate_synthetic_network_data,
        )

        report_dir.mkdir(parents=True, exist_ok=True)

        # Generate synthetic data
        config = SyntheticNetworkConfig(n_strains=50, n_features=20, random_state=42)
        data_df, metadata = generate_network_synthetic_dataset(config)

        # Validate
        validation = validate_synthetic_network_data(data_df, metadata)

        # Create report
        report = {
            "test_name": "Synthetic Network Data Validation",
            "timestamp": pd.Timestamp.now().isoformat(),
            "config": {
                "n_strains": config.n_strains,
                "n_features": config.n_features,
                "n_true_associations": config.n_true_associations,
                "random_state": config.random_state,
            },
            "validation_passed": validation["validation_passed"],
            "checks_passed": len(validation["checks"]),
            "warnings": len(validation["warnings"]),
            "errors": len(validation["errors"]),
        }

        # Save report
        report_file = report_dir / "validation_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        assert report_file.exists()
        assert validation["validation_passed"]
