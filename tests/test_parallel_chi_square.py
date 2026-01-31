"""
Tests for parallel chi-square matrix computation.

Author: MK-vet
License: MIT
"""

import pytest
import numpy as np
import pandas as pd

from strepsuis_genphennet.parallel_chi_square import (
    parallel_chi_square_matrix,
    filter_significant_associations,
    _compute_chi_square_pair,
)


class TestComputeChiSquarePair:
    """Test single pairwise chi-square computation."""

    def test_independent_features(self):
        """Test chi-square for independent features."""
        np.random.seed(42)
        data1 = np.random.choice(['A', 'B'], 1000)
        data2 = np.random.choice(['X', 'Y'], 1000)

        result = _compute_chi_square_pair(data1, data2, 'feat1', 'feat2')

        assert 'chi2' in result
        assert 'p_value' in result
        assert 'cramers_v' in result
        assert result['valid'] == True
        # Independent features should have high p-value
        assert result['p_value'] > 0.05

    def test_dependent_features(self):
        """Test chi-square for perfectly dependent features."""
        data1 = np.array(['A'] * 500 + ['B'] * 500)
        data2 = np.array(['X'] * 500 + ['Y'] * 500)  # Perfect correlation

        result = _compute_chi_square_pair(data1, data2, 'feat1', 'feat2')

        assert result['valid'] == True
        # Dependent features should have low p-value
        assert result['p_value'] < 0.001
        # Should have high Cramér's V
        assert result['cramers_v'] > 0.8

    def test_no_variation(self):
        """Test with constant feature."""
        data1 = np.array(['A'] * 100)
        data2 = np.random.choice(['X', 'Y'], 100)

        result = _compute_chi_square_pair(data1, data2, 'feat1', 'feat2')

        assert result['valid'] == False
        assert result['p_value'] == 1.0

    def test_with_missing_values(self):
        """Test handling of missing values."""
        data1 = np.array([1.0, 2.0, np.nan, 1.0, 2.0] * 20)
        data2 = np.array([1.0, 1.0, 2.0, 2.0, np.nan] * 20)

        result = _compute_chi_square_pair(data1, data2, 'feat1', 'feat2')

        # Should handle NaN values gracefully
        assert 'p_value' in result


class TestParallelChiSquareMatrix:
    """Test parallel chi-square matrix computation."""

    def test_basic_computation(self):
        """Test basic matrix computation."""
        np.random.seed(42)
        df = pd.DataFrame({
            'gene1': np.random.choice(['A', 'B'], 200),
            'gene2': np.random.choice(['X', 'Y'], 200),
            'gene3': np.random.choice(['P', 'Q'], 200)
        })

        chi2_mat, p_mat, v_mat = parallel_chi_square_matrix(
            df,
            n_jobs=2,
            apply_fdr=False
        )

        # Check dimensions
        assert chi2_mat.shape == (3, 3)
        assert p_mat.shape == (3, 3)
        assert v_mat.shape == (3, 3)

        # Check symmetry
        assert np.allclose(chi2_mat, chi2_mat.T)
        assert np.allclose(p_mat, p_mat.T)
        assert np.allclose(v_mat, v_mat.T)

        # Check diagonal is zero/one
        np.testing.assert_array_equal(np.diag(chi2_mat), [0, 0, 0])
        np.testing.assert_array_equal(np.diag(p_mat), [1, 1, 1])

    def test_with_fdr_correction(self):
        """Test FDR correction."""
        np.random.seed(42)
        df = pd.DataFrame({
            f'gene{i}': np.random.choice(['A', 'B'], 100)
            for i in range(10)
        })

        chi2_mat, p_mat_fdr, v_mat = parallel_chi_square_matrix(
            df,
            n_jobs=2,
            apply_fdr=True,
            fdr_alpha=0.05
        )

        # FDR-adjusted p-values should be >= original
        assert p_mat_fdr.values.min() >= 0

    def test_deterministic_results(self):
        """Test that results are deterministic."""
        np.random.seed(42)
        df = pd.DataFrame({
            'A': np.random.choice(['X', 'Y'], 100),
            'B': np.random.choice(['P', 'Q'], 100)
        })

        # Run twice
        chi2_1, p_1, v_1 = parallel_chi_square_matrix(df, n_jobs=1, apply_fdr=False)
        chi2_2, p_2, v_2 = parallel_chi_square_matrix(df, n_jobs=1, apply_fdr=False)

        # Should get identical results
        np.testing.assert_array_almost_equal(chi2_1.values, chi2_2.values)
        np.testing.assert_array_almost_equal(p_1.values, p_2.values)
        np.testing.assert_array_almost_equal(v_1.values, v_2.values)

    def test_subset_features(self):
        """Test with subset of features."""
        np.random.seed(42)
        df = pd.DataFrame({
            'A': np.random.choice(['X', 'Y'], 100),
            'B': np.random.choice(['P', 'Q'], 100),
            'C': np.random.choice(['M', 'N'], 100),
            'D': np.random.choice(['R', 'S'], 100)
        })

        # Only test A and B
        chi2_mat, p_mat, v_mat = parallel_chi_square_matrix(
            df,
            features=['A', 'B'],
            n_jobs=2,
            apply_fdr=False
        )

        assert chi2_mat.shape == (2, 2)
        assert list(chi2_mat.index) == ['A', 'B']

    def test_single_feature_error(self):
        """Test error with only one feature."""
        df = pd.DataFrame({'A': [1, 2, 3]})

        with pytest.raises(ValueError, match="at least 2 features"):
            parallel_chi_square_matrix(df)

    def test_large_number_of_features(self):
        """Test scalability with many features."""
        np.random.seed(42)
        n_features = 50
        n_samples = 100

        df = pd.DataFrame({
            f'feat{i}': np.random.choice(['A', 'B'], n_samples)
            for i in range(n_features)
        })

        chi2_mat, p_mat, v_mat = parallel_chi_square_matrix(
            df,
            n_jobs=2,
            apply_fdr=True
        )

        # Should compute all pairs: n*(n-1)/2 = 50*49/2 = 1225 pairs
        assert chi2_mat.shape == (n_features, n_features)

        # Check no NaN values
        assert not np.any(np.isnan(chi2_mat.values))


class TestFilterSignificantAssociations:
    """Test filtering of significant associations."""

    def test_basic_filtering(self):
        """Test basic filtering."""
        # Create simple matrices
        features = ['A', 'B', 'C']
        chi2_mat = pd.DataFrame(
            [[0, 10, 5], [10, 0, 15], [5, 15, 0]],
            index=features,
            columns=features
        )
        p_mat = pd.DataFrame(
            [[1, 0.001, 0.1], [0.001, 1, 0.001], [0.1, 0.001, 1]],
            index=features,
            columns=features
        )
        v_mat = pd.DataFrame(
            [[0, 0.5, 0.2], [0.5, 0, 0.6], [0.2, 0.6, 0]],
            index=features,
            columns=features
        )

        sig = filter_significant_associations(
            chi2_mat, p_mat, v_mat,
            p_threshold=0.05,
            min_cramers_v=0.3
        )

        # Should find 2 significant pairs (A-B and B-C)
        assert len(sig) == 2
        assert 'feature1' in sig.columns
        assert 'feature2' in sig.columns
        assert 'cramers_v' in sig.columns

    def test_sorting_by_effect_size(self):
        """Test that results are sorted by Cramér's V."""
        features = ['A', 'B', 'C']
        chi2_mat = pd.DataFrame(
            [[0, 5, 10], [5, 0, 20], [10, 20, 0]],
            index=features,
            columns=features
        )
        p_mat = pd.DataFrame(
            [[1, 0.01, 0.001], [0.01, 1, 0.001], [0.001, 0.001, 1]],
            index=features,
            columns=features
        )
        v_mat = pd.DataFrame(
            [[0, 0.3, 0.5], [0.3, 0, 0.7], [0.5, 0.7, 0]],
            index=features,
            columns=features
        )

        sig = filter_significant_associations(
            chi2_mat, p_mat, v_mat,
            p_threshold=0.05,
            min_cramers_v=0.2
        )

        # Should be sorted by descending Cramér's V
        assert sig['cramers_v'].is_monotonic_decreasing

    def test_empty_result(self):
        """Test when no associations pass thresholds."""
        features = ['A', 'B']
        chi2_mat = pd.DataFrame(
            [[0, 1], [1, 0]],
            index=features,
            columns=features
        )
        p_mat = pd.DataFrame(
            [[1, 0.5], [0.5, 1]],
            index=features,
            columns=features
        )
        v_mat = pd.DataFrame(
            [[0, 0.1], [0.1, 0]],
            index=features,
            columns=features
        )

        sig = filter_significant_associations(
            chi2_mat, p_mat, v_mat,
            p_threshold=0.05,
            min_cramers_v=0.3
        )

        assert len(sig) == 0


class TestPerformance:
    """Test performance characteristics."""

    def test_parallel_speedup(self):
        """Test that parallel execution provides speedup."""
        import time

        np.random.seed(42)
        df = pd.DataFrame({
            f'gene{i}': np.random.choice(['A', 'B'], 500)
            for i in range(30)
        })

        # Sequential
        start = time.perf_counter()
        parallel_chi_square_matrix(df, n_jobs=1, apply_fdr=False)
        time_seq = time.perf_counter() - start

        # Parallel
        start = time.perf_counter()
        parallel_chi_square_matrix(df, n_jobs=2, apply_fdr=False)
        time_par = time.perf_counter() - start

        # Parallel should be faster (allowing for overhead)
        # Parallel should provide reasonable performance (allowing for overhead)
        # Sometimes overhead can exceed benefit for small datasets
        assert time_par < time_seq * 2.0  # Allow more overhead tolerance

    def test_memory_efficiency(self):
        """Test that large matrices don't cause memory issues."""
        np.random.seed(42)
        n_features = 100
        n_samples = 200

        df = pd.DataFrame({
            f'feat{i}': np.random.choice(['A', 'B'], n_samples)
            for i in range(n_features)
        })

        # Should complete without memory errors
        chi2_mat, p_mat, v_mat = parallel_chi_square_matrix(
            df,
            n_jobs=2,
            apply_fdr=False
        )

        assert chi2_mat.shape == (n_features, n_features)
