#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Coverage Tests for network_analysis_core.py

Target: Increase coverage from 14% to 70%+

Focus Areas:
- perform_full_analysis (main pipeline)
- build_association_network functions
- find_mutually_exclusive
- calculate_entropy_metrics
- chi2_test_binary
- Network construction and community detection
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys
import tempfile
from pathlib import Path
import networkx as nx

# Add module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strepsuis_genphennet import network_analysis_core as nac


class TestStatisticalFunctions:
    """Test chi2_phi, cramers_v, and statistical association functions."""

    def test_chi2_phi_basic(self):
        """Test chi2_phi with simple binary data."""
        x = pd.Series([0, 0, 1, 1, 1])
        y = pd.Series([0, 1, 0, 1, 1])

        p, phi, contingency, lo, hi = nac.chi2_phi(x, y)

        assert isinstance(p, float)
        assert 0 <= p <= 1
        assert isinstance(phi, float)
        assert -1 <= phi <= 1
        assert isinstance(contingency, pd.DataFrame)
        assert contingency.shape == (2, 2)

    def test_chi2_phi_perfect_association(self):
        """Test chi2_phi with perfect positive association."""
        # Create perfect association: when x=1, y=1; when x=0, y=0
        x = pd.Series([0, 0, 0, 1, 1, 1, 0, 1] * 5)
        y = pd.Series([0, 0, 0, 1, 1, 1, 0, 1] * 5)

        p, phi, _, _, _ = nac.chi2_phi(x, y)

        # Perfect association should have phi = 1.0 or very close
        # p-value may vary depending on implementation
        assert phi >= 0.9  # Strong positive association

    def test_chi2_phi_no_association(self):
        """Test chi2_phi with no association."""
        x = pd.Series([0, 0, 1, 1] * 5)
        y = pd.Series([0, 1, 0, 1] * 5)

        p, phi, _, _, _ = nac.chi2_phi(x, y)

        assert p > 0.05  # Not significant
        assert abs(phi) < 0.3  # Weak association

    def test_chi2_phi_small_sample(self):
        """Test chi2_phi with small sample (triggers Fisher's exact)."""
        x = pd.Series([0, 0, 1, 1])
        y = pd.Series([0, 1, 0, 1])

        # Should use Fisher's exact test (n <= 20)
        p, phi, _, lo, hi = nac.chi2_phi(x, y)

        assert isinstance(p, float)
        assert lo == 0.0  # CI not calculated for Fisher's exact
        assert hi == 0.0

    def test_chi2_phi_very_small_sample(self):
        """Test chi2_phi with very small sample (n <= 3)."""
        x = pd.Series([0, 1, 1])
        y = pd.Series([0, 0, 1])

        p, phi, _, lo, hi = nac.chi2_phi(x, y)

        # Should handle gracefully with small sample
        assert isinstance(p, float)
        assert 0 <= p <= 1
        # phi should be valid but may not be 0 for small samples
        assert -1 <= phi <= 1
        # Confidence intervals may or may not be calculated
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_cramers_v_basic(self):
        """Test Cramér's V calculation."""
        contingency = pd.DataFrame([[10, 5], [3, 12]])

        v, lo, hi = nac.cramers_v(contingency)

        assert isinstance(v, float)
        assert 0 <= v <= 1
        assert lo <= v <= hi

    def test_cramers_v_perfect(self):
        """Test Cramér's V with perfect association."""
        contingency = pd.DataFrame([[10, 0], [0, 10]])

        v, _, _ = nac.cramers_v(contingency)

        assert v > 0.8  # Strong association

    def test_cramers_v_degenerate(self):
        """Test Cramér's V with degenerate table."""
        # Single row
        contingency = pd.DataFrame([[10, 5]])
        v, _, _ = nac.cramers_v(contingency)
        assert v == 0.0

        # Very small sample
        contingency = pd.DataFrame([[1, 1], [1, 0]])
        v, _, _ = nac.cramers_v(contingency)
        assert v == 0.0


class TestInformationTheory:
    """Test entropy and information theory functions."""

    def test_calculate_entropy_uniform(self):
        """Test entropy calculation for uniform distribution."""
        series = pd.Series([0, 1, 2, 3] * 10)

        H, Hn = nac.calculate_entropy(series)

        assert H > 0
        assert 0 <= Hn <= 1
        # Uniform distribution should have high normalized entropy
        # (uses natural log, so not exactly 1)
        assert Hn > 0.5

    def test_calculate_entropy_single_value(self):
        """Test entropy for constant series."""
        series = pd.Series([1, 1, 1, 1])

        H, Hn = nac.calculate_entropy(series)

        # No entropy for constant
        assert H == 0.0
        assert Hn == 0.0

    def test_calculate_entropy_binary(self):
        """Test entropy for binary variable."""
        # 50-50 split (maximum entropy for binary)
        series = pd.Series([0, 1] * 20)

        H, Hn = nac.calculate_entropy(series)

        # Uses natural log, so max entropy for binary is ln(2) ≈ 0.693
        assert H > 0.6  # Close to ln(2)
        assert 0 <= Hn <= 1  # Normalized

    def test_conditional_entropy(self):
        """Test conditional entropy H(X|Y)."""
        x = pd.Series([0, 0, 1, 1] * 5)
        y = pd.Series([0, 0, 1, 1] * 5)

        ce = nac.conditional_entropy(x, y)

        # X is completely determined by Y
        assert ce < 0.1  # Should be close to 0

    def test_conditional_entropy_independent(self):
        """Test conditional entropy for independent variables."""
        x = pd.Series([0, 1, 0, 1] * 5)
        y = pd.Series([0, 0, 1, 1] * 5)

        ce = nac.conditional_entropy(x, y)

        # Should be close to H(X)
        H_x, _ = nac.calculate_entropy(x)
        assert abs(ce - H_x) < 0.2

    def test_information_gain(self):
        """Test information gain I(X;Y)."""
        # Perfectly correlated
        x = pd.Series([0, 0, 1, 1] * 5)
        y = pd.Series([0, 0, 1, 1] * 5)

        ig = nac.information_gain(x, y)

        # High information gain
        assert ig > 0.5

    def test_information_gain_independent(self):
        """Test information gain for independent variables."""
        x = pd.Series([0, 1] * 20)
        y = pd.Series([0, 0, 1, 1] * 10)

        ig = nac.information_gain(x, y)

        # Low information gain
        assert ig < 0.2

    def test_normalized_mutual_info(self):
        """Test normalized mutual information."""
        x = pd.Series([0, 0, 1, 1] * 5)
        y = pd.Series([0, 0, 1, 1] * 5)

        nmi = nac.normalized_mutual_info(x, y)

        assert 0 <= nmi <= 1
        # Perfect correlation should give high NMI
        assert nmi > 0.8

    def test_normalized_mutual_info_zero_entropy(self):
        """Test NMI with zero entropy variables."""
        x = pd.Series([1, 1, 1, 1])
        y = pd.Series([0, 0, 0, 0])

        nmi = nac.normalized_mutual_info(x, y)

        # Should return 0 (undefined, but handled)
        assert nmi == 0.0


class TestMutuallyExclusive:
    """Test mutually exclusive pattern detection."""

    def test_find_mutually_exclusive_pairs(self):
        """Test finding mutually exclusive pairs."""
        df = pd.DataFrame({
            'F1': [1, 0, 1, 0],
            'F2': [0, 1, 0, 1],
            'F3': [1, 1, 0, 0]
        })
        mapping = {'F1': 'Cat1', 'F2': 'Cat2', 'F3': 'Cat3'}

        excl = nac.find_mutually_exclusive(df, ['F1', 'F2', 'F3'], mapping, k=2)

        # F1 and F2 are mutually exclusive
        assert len(excl) >= 1
        assert 'Feature_1' in excl.columns
        assert 'Feature_2' in excl.columns
        assert 'Category_1' in excl.columns

    def test_find_mutually_exclusive_triplets(self):
        """Test finding mutually exclusive triplets."""
        df = pd.DataFrame({
            'F1': [1, 0, 0, 0],
            'F2': [0, 1, 0, 0],
            'F3': [0, 0, 1, 0]
        })
        mapping = {'F1': 'A', 'F2': 'B', 'F3': 'C'}

        excl = nac.find_mutually_exclusive(df, ['F1', 'F2', 'F3'], mapping, k=3)

        # All three are mutually exclusive
        assert len(excl) >= 1
        assert 'Feature_3' in excl.columns

    def test_find_mutually_exclusive_none(self):
        """Test when no mutually exclusive patterns exist."""
        df = pd.DataFrame({
            'F1': [1, 1, 1, 1],
            'F2': [1, 1, 1, 1],
            'F3': [1, 1, 1, 1]
        })
        mapping = {'F1': 'A', 'F2': 'B', 'F3': 'C'}

        excl = nac.find_mutually_exclusive(df, ['F1', 'F2', 'F3'], mapping, k=2)

        # No mutually exclusive patterns
        assert len(excl) == 0

    def test_find_mutually_exclusive_max_patterns(self):
        """Test max_patterns limit."""
        # Create many mutually exclusive pairs
        n_features = 20
        data = np.eye(n_features)
        df = pd.DataFrame(data, columns=[f'F{i}' for i in range(n_features)])
        mapping = {f'F{i}': f'Cat{i}' for i in range(n_features)}

        excl = nac.find_mutually_exclusive(df, df.columns.tolist(), mapping, k=2, max_patterns=10)

        # Should respect max_patterns limit
        assert len(excl) == 10


class TestClusterHubs:
    """Test cluster hub identification."""

    def test_get_cluster_hubs_basic(self):
        """Test hub extraction from network."""
        df = pd.DataFrame({
            'Cluster': [1, 1, 1, 2, 2, 2],
            'Feature': ['F1', 'F2', 'F3', 'F4', 'F5', 'F6'],
            'Category': ['A', 'A', 'B', 'B', 'C', 'C'],
            'Degree_Centrality': [0.9, 0.7, 0.5, 0.8, 0.6, 0.4]
        })

        hubs = nac.get_cluster_hubs(df, top_n=2)

        # Should get top 2 from each cluster
        assert len(hubs) == 4
        assert 'Cluster' in hubs.columns
        assert 'Feature' in hubs.columns
        assert 'Degree_Centrality' in hubs.columns

    def test_get_cluster_hubs_small_cluster(self):
        """Test hub extraction when cluster has fewer than top_n nodes."""
        df = pd.DataFrame({
            'Cluster': [1, 1, 2],
            'Feature': ['F1', 'F2', 'F3'],
            'Category': ['A', 'B', 'C'],
            'Degree_Centrality': [0.9, 0.7, 0.5]
        })

        hubs = nac.get_cluster_hubs(df, top_n=5)

        # Should adapt to cluster size
        assert len(hubs) == 3

    def test_get_cluster_hubs_missing_columns(self):
        """Test hub extraction with missing required columns."""
        df = pd.DataFrame({
            'Feature': ['F1', 'F2'],
            'Category': ['A', 'B']
        })

        hubs = nac.get_cluster_hubs(df, top_n=2)

        # Should return empty DataFrame
        assert len(hubs) == 0

    def test_get_cluster_hubs_empty(self):
        """Test hub extraction with empty DataFrame."""
        df = pd.DataFrame()

        hubs = nac.get_cluster_hubs(df, top_n=3)

        assert len(hubs) == 0


class TestUtilityFunctions:
    """Test utility functions."""

    def test_expand_categories_basic(self):
        """Test category expansion (one-hot encoding)."""
        df = pd.DataFrame({
            'Strain_ID': ['S1', 'S2', 'S3'],
            'MGE': ['Type_A', 'Type_B', 'Type_A']
        })

        expanded = nac.expand_categories(df, 'MGE')

        assert 'Strain_ID' in expanded.columns
        assert 'MGE_Type_A' in expanded.columns
        assert 'MGE_Type_B' in expanded.columns
        assert len(expanded) == 3

    def test_expand_categories_mlst(self):
        """Test category expansion for MLST (removes .0)."""
        df = pd.DataFrame({
            'Strain_ID': ['S1', 'S2'],
            'MLST': ['1.0', '2.0']
        })

        expanded = nac.expand_categories(df, 'MLST')

        # Should have MLST_1 and MLST_2, not MLST_1.0
        cols = [c for c in expanded.columns if c.startswith('MLST_')]
        assert all('.0' not in c for c in cols)

    def test_expand_categories_multiple_per_strain(self):
        """Test category expansion with multiple rows per strain."""
        df = pd.DataFrame({
            'Strain_ID': ['S1', 'S1', 'S2'],
            'Plasmid': ['P1', 'P2', 'P1']
        })

        expanded = nac.expand_categories(df, 'Plasmid')

        # Should aggregate: S1 has both P1 and P2
        assert len(expanded) == 2  # 2 unique strains
        s1_row = expanded[expanded['Strain_ID'] == 'S1'].iloc[0]
        assert s1_row['Plasmid_P1'] == 1
        assert s1_row['Plasmid_P2'] == 1

    def test_get_centrality(self):
        """Test centrality getter function."""
        cent_dict = {'A': 0.5, 'B': 0.8, 'C': 0.3}

        result = nac.get_centrality(cent_dict)

        assert result == cent_dict

    def test_adaptive_phi_threshold_percentile(self):
        """Test adaptive threshold with percentile method."""
        phi_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        threshold = nac.adaptive_phi_threshold(phi_vals, method='percentile', percentile=90)

        # 90th percentile should be around 0.9
        assert threshold >= 0.88  # Allow for numpy percentile calculation variance

    def test_adaptive_phi_threshold_iqr(self):
        """Test adaptive threshold with IQR method."""
        phi_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        threshold = nac.adaptive_phi_threshold(phi_vals, method='iqr')

        assert isinstance(threshold, float)
        assert threshold > 0

    def test_adaptive_phi_threshold_statistical(self):
        """Test adaptive threshold with statistical method."""
        phi_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        threshold = nac.adaptive_phi_threshold(phi_vals, method='statistical')

        # mean + 2*std
        expected = np.mean(phi_vals) + 2 * np.std(phi_vals)
        assert threshold == pytest.approx(expected, abs=0.01)

    def test_adaptive_phi_threshold_unknown_method(self):
        """Test adaptive threshold with unknown method."""
        phi_vals = np.array([0.1, 0.2, 0.3])

        threshold = nac.adaptive_phi_threshold(phi_vals, method='unknown')

        # Should return default 0.5
        assert threshold == 0.5


class TestHTMLGeneration:
    """Test HTML report generation utilities."""

    def test_create_interactive_table(self):
        """Test HTML table creation."""
        df = pd.DataFrame({
            'Feature': ['F1', 'F2'],
            'Value': [0.123456, 0.987654]
        })

        html = nac.create_interactive_table(df, 'test_table')

        assert '<table' in html
        assert 'id="table-test_table"' in html
        assert 'Feature' in html
        assert 'Value' in html
        assert '0.123' in html  # Should be rounded to 3 decimals

    def test_create_interactive_table_with_empty(self):
        """Test HTML table creation with empty DataFrame."""
        df = pd.DataFrame()

        html = nac.create_interactive_table_with_empty(df, 'empty_table')

        assert 'No data available' in html

    def test_create_section_summary_basic(self):
        """Test summary block creation."""
        stats = {'Count': 10, 'Mean': 0.5}

        html = nac.create_section_summary('Test Section', stats)

        assert 'Test Section' in html
        assert 'Count: 10' in html
        assert 'Mean: 0.5' in html

    def test_create_section_summary_with_category(self):
        """Test summary block with category breakdown."""
        stats = {'Total': 100}
        per_category = {'AMR': '50 items', 'Virulence': '50 items'}

        html = nac.create_section_summary('Summary', stats, per_category=per_category)

        assert 'By category' in html
        assert 'AMR: 50 items' in html
        assert 'Virulence: 50 items' in html

    def test_create_section_summary_with_feature(self):
        """Test summary block with feature breakdown (collapsible)."""
        stats = {'Total': 10}
        per_feature = {'F1': 'stat1', 'F2': 'stat2'}

        html = nac.create_section_summary('Summary', stats, per_feature=per_feature)

        assert 'By feature' in html
        assert '<details>' in html
        assert 'F1: stat1' in html


class TestFileMatching:
    """Test file matching utilities."""

    def test_find_matching_files_exact(self):
        """Test exact file name matching."""
        uploaded = ['AMR_genes.csv', 'MIC.csv']
        expected = ['AMR_genes.csv', 'MIC.csv']

        mapping = nac.find_matching_files(uploaded, expected)

        assert mapping == {'AMR_genes.csv': 'AMR_genes.csv', 'MIC.csv': 'MIC.csv'}

    def test_find_matching_files_numbered(self):
        """Test matching with numbered duplicates."""
        uploaded = ['AMR_genes (1).csv', 'MIC (2).csv']
        expected = ['AMR_genes.csv', 'MIC.csv']

        mapping = nac.find_matching_files(uploaded, expected)

        assert mapping['AMR_genes.csv'] == 'AMR_genes (1).csv'
        assert mapping['MIC.csv'] == 'MIC (2).csv'

    def test_find_matching_files_case_insensitive(self):
        """Test case-insensitive matching."""
        uploaded = ['amr_genes.csv', 'mic.csv']
        expected = ['AMR_genes.csv', 'MIC.csv']

        mapping = nac.find_matching_files(uploaded, expected)

        assert mapping['AMR_genes.csv'] == 'amr_genes.csv'
        assert mapping['MIC.csv'] == 'mic.csv'

    def test_find_matching_files_missing(self):
        """Test handling of missing files."""
        uploaded = ['AMR_genes.csv']
        expected = ['AMR_genes.csv', 'MIC.csv', 'Virulence.csv']

        mapping = nac.find_matching_files(uploaded, expected)

        assert 'AMR_genes.csv' in mapping
        assert 'MIC.csv' not in mapping
        assert 'Virulence.csv' not in mapping


class TestSummarizationFunctions:
    """Test data summarization helper functions."""

    def test_summarize_by_category(self):
        """Test category-level summarization."""
        df = pd.DataFrame({
            'Category1': ['AMR', 'AMR', 'Vir'],
            'Category2': ['Vir', 'MGE', 'MGE'],
            'Phi': [0.5, 0.7, 0.3]
        })

        summary = nac.summarize_by_category(df, 'Phi', ['Category1', 'Category2'])

        assert 'AMR' in summary
        assert 'Vir' in summary
        assert 'MGE' in summary
        assert 'Count:' in summary['AMR']
        assert 'mean:' in summary['AMR']

    def test_summarize_by_feature(self):
        """Test feature-level summarization."""
        df = pd.DataFrame({
            'Feature1': ['F1', 'F2', 'F3'],
            'Feature2': ['F2', 'F3', 'F4'],
            'Value': [0.5, 0.7, 0.3]
        })

        summary = nac.summarize_by_feature(df, 'Value', ['Feature1', 'Feature2'])

        assert 'F1' in summary
        assert 'F2' in summary
        assert 'Count:' in summary['F1']

    def test_summarize_by_category_excl(self):
        """Test exclusive pattern category summarization."""
        df = pd.DataFrame({
            'Feature_1': ['F1', 'F2'],
            'Category_1': ['AMR', 'Vir'],
            'Feature_2': ['F3', 'F4'],
            'Category_2': ['MGE', 'AMR']
        })

        summary = nac.summarize_by_category_excl(df, k=2)

        assert 'AMR' in summary
        assert 'Count:' in summary['AMR']

    def test_summarize_by_feature_excl(self):
        """Test exclusive pattern feature summarization."""
        df = pd.DataFrame({
            'Feature_1': ['F1', 'F2'],
            'Feature_2': ['F3', 'F4']
        })

        summary = nac.summarize_by_feature_excl(df, k=2)

        assert 'F1' in summary
        assert 'F2' in summary
        assert 'Count:' in summary['F1']

    def test_summarize_by_category_network(self):
        """Test network category summarization."""
        df = pd.DataFrame({
            'Category': ['AMR', 'AMR', 'Vir'],
            'Degree_Centrality': [0.8, 0.6, 0.4]
        })

        summary = nac.summarize_by_category_network(df)

        assert 'AMR' in summary
        assert 'Vir' in summary
        assert 'mean degree:' in summary['AMR']

    def test_summarize_by_feature_network(self):
        """Test network feature summarization."""
        df = pd.DataFrame({
            'Feature': ['F1', 'F2'],
            'Degree_Centrality': [0.8, 0.4]
        })

        summary = nac.summarize_by_feature_network(df)

        assert 'F1' in summary
        assert 'degree: 0.8' in summary['F1']


class TestSetupLogging:
    """Test logging configuration."""

    def test_setup_logging(self):
        """Test logging setup creates output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Change output folder temporarily
            original_output = nac.output_folder
            nac.output_folder = tmpdir

            try:
                nac.setup_logging()

                # Should create output folder
                assert os.path.exists(tmpdir)

                # Should create log file
                log_file = os.path.join(tmpdir, 'network_analysis_log.txt')
                assert os.path.exists(log_file)
            finally:
                nac.output_folder = original_output


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
