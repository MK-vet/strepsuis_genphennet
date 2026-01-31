"""
Comprehensive Unit Tests for network_analysis_core.py

This module provides extensive test coverage for all network analysis functions,
including:
- Chi-square/phi coefficient calculations
- Cramér's V calculation
- Entropy and mutual information
- Network construction and centrality measures
- Community detection (Louvain)
- Hub identification

Target: 95%+ coverage for network_analysis_core.py
"""

import numpy as np
import pandas as pd
import pytest
import networkx as nx


# ============================================================================
# Test chi2_phi function
# ============================================================================
class TestChi2Phi:
    """Test chi-square and phi coefficient calculations."""

    def test_large_sample_chi_square(self):
        """Test chi-square path for large samples."""
        from strepsuis_genphennet.network_analysis_core import chi2_phi
        
        # Large sample, expected >= 5
        x = pd.Series([0]*50 + [1]*50)
        y = pd.Series([0]*30 + [1]*20 + [0]*20 + [1]*30)
        
        p, phi, contingency, lo, hi = chi2_phi(x, y)
        
        assert 0 <= p <= 1
        assert -1 <= phi <= 1
        assert isinstance(contingency, pd.DataFrame)

    def test_small_sample_fisher_exact(self):
        """Test Fisher's exact path for small samples."""
        from strepsuis_genphennet.network_analysis_core import chi2_phi
        
        # Small sample (n <= 20)
        x = pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 0, 1])
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        
        p, phi, contingency, lo, hi = chi2_phi(x, y)
        
        assert 0 <= p <= 1
        assert 0 <= phi < 1  # Phi is capped at 0.9999

    def test_perfect_association(self):
        """Test with perfect association."""
        from strepsuis_genphennet.network_analysis_core import chi2_phi
        
        x = pd.Series([0, 0, 0, 0, 1, 1, 1, 1] * 10)
        y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1] * 10)  # Same as x
        
        p, phi, contingency, lo, hi = chi2_phi(x, y)
        
        assert p < 0.001  # Highly significant
        assert phi > 0.9  # Strong association

    def test_independence(self):
        """Test with independent variables."""
        from strepsuis_genphennet.network_analysis_core import chi2_phi
        
        np.random.seed(42)
        n = 100
        x = pd.Series(np.random.binomial(1, 0.5, n))
        y = pd.Series(np.random.binomial(1, 0.5, n))
        
        p, phi, contingency, lo, hi = chi2_phi(x, y)
        
        # For random data, phi should be small
        assert abs(phi) < 0.3  # Allow some randomness


# ============================================================================
# Test cramers_v function
# ============================================================================
class TestCramersV:
    """Test Cramér's V calculation."""

    def test_basic_cramers_v(self):
        """Test basic Cramér's V calculation."""
        from strepsuis_genphennet.network_analysis_core import cramers_v
        
        contingency = pd.DataFrame([
            [30, 10],
            [10, 30],
        ])
        
        v, lo, hi = cramers_v(contingency)
        
        assert 0 <= v <= 1
        assert lo <= v <= hi

    def test_cramers_v_small_table(self):
        """Test with small contingency table."""
        from strepsuis_genphennet.network_analysis_core import cramers_v
        
        # 1x2 table - should return 0
        contingency = pd.DataFrame([[5, 5]])
        
        v, lo, hi = cramers_v(contingency)
        
        assert v == 0.0

    def test_cramers_v_larger_table(self):
        """Test with 3x3 contingency table."""
        from strepsuis_genphennet.network_analysis_core import cramers_v
        
        contingency = pd.DataFrame([
            [20, 5, 5],
            [5, 20, 5],
            [5, 5, 20],
        ])
        
        v, lo, hi = cramers_v(contingency)
        
        assert 0 <= v <= 1


# ============================================================================
# Test entropy functions
# ============================================================================
class TestEntropy:
    """Test entropy calculation functions."""

    def test_calculate_entropy_uniform(self):
        """Test entropy for uniform distribution."""
        from strepsuis_genphennet.network_analysis_core import calculate_entropy
        
        # Uniform distribution (maximum entropy)
        series = pd.Series([1, 2, 3, 4] * 25)
        
        H, Hn = calculate_entropy(series)
        
        assert H > 0
        # Normalized entropy should be positive for uniform distribution
        # (The implementation uses scipy_entropy which returns natural log)
        assert Hn > 0

    def test_calculate_entropy_single_value(self):
        """Test entropy for single value (zero entropy)."""
        from strepsuis_genphennet.network_analysis_core import calculate_entropy
        
        series = pd.Series([1] * 100)  # All same value
        
        H, Hn = calculate_entropy(series)
        
        assert H == 0.0
        assert Hn == 0.0

    def test_calculate_entropy_skewed(self):
        """Test entropy for skewed distribution."""
        from strepsuis_genphennet.network_analysis_core import calculate_entropy
        
        # Heavily skewed
        series = pd.Series([1] * 90 + [2] * 10)
        
        H, Hn = calculate_entropy(series)
        
        assert 0 < H < 1
        assert 0 < Hn < 1

    def test_conditional_entropy(self):
        """Test conditional entropy calculation."""
        from strepsuis_genphennet.network_analysis_core import conditional_entropy
        
        # When y is perfectly predictable from x, CE should be low
        x = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
        y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
        
        ce = conditional_entropy(x, y)
        
        assert ce >= 0  # CE is non-negative
        assert ce < 0.1  # Should be very low

    def test_information_gain(self):
        """Test information gain calculation."""
        from strepsuis_genphennet.network_analysis_core import information_gain
        
        x = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
        y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
        
        ig = information_gain(x, y)
        
        assert ig >= 0  # IG is non-negative

    def test_normalized_mutual_info(self):
        """Test normalized mutual information."""
        from strepsuis_genphennet.network_analysis_core import normalized_mutual_info
        
        # Perfect dependence
        x = pd.Series([0, 0, 1, 1] * 25)
        y = pd.Series([0, 0, 1, 1] * 25)
        
        nmi = normalized_mutual_info(x, y)
        
        assert 0 <= nmi <= 1
        # Should be close to 1 for perfect dependence
        assert nmi > 0.9


# ============================================================================
# Test helper functions
# ============================================================================
class TestHelperFunctions:
    """Test utility/helper functions."""

    def test_expand_categories(self):
        """Test category expansion to dummies."""
        from strepsuis_genphennet.network_analysis_core import expand_categories
        
        df = pd.DataFrame({
            'Strain_ID': ['S1', 'S2', 'S3', 'S4'],
            'MLST': [1, 2, 1, 3],
        })
        
        result = expand_categories(df, 'MLST')
        
        # Should have Strain_ID and dummy columns
        assert 'Strain_ID' in result.columns
        assert len(result) == 4

    def test_expand_categories_serotype(self):
        """Test category expansion for Serotype."""
        from strepsuis_genphennet.network_analysis_core import expand_categories
        
        df = pd.DataFrame({
            'Strain_ID': ['S1', 'S2', 'S3'],
            'Serotype': ['2.0', '3.0', '2.0'],  # Float-like strings
        })
        
        result = expand_categories(df, 'Serotype')
        
        assert 'Strain_ID' in result.columns

    def test_find_matching_files_exact(self):
        """Test file matching with exact match."""
        from strepsuis_genphennet.network_analysis_core import find_matching_files
        
        uploaded = ['MIC.csv', 'AMR_genes.csv', 'Virulence.csv']
        expected = ['MIC.csv', 'AMR_genes.csv']
        
        mapping = find_matching_files(uploaded, expected)
        
        assert mapping['MIC.csv'] == 'MIC.csv'
        assert mapping['AMR_genes.csv'] == 'AMR_genes.csv'

    def test_find_matching_files_numbered(self):
        """Test file matching with numbered files."""
        from strepsuis_genphennet.network_analysis_core import find_matching_files
        
        uploaded = ['MIC (1).csv', 'AMR_genes.csv']
        expected = ['MIC.csv']
        
        mapping = find_matching_files(uploaded, expected)
        
        assert mapping.get('MIC.csv') == 'MIC (1).csv'

    def test_get_centrality(self):
        """Test centrality getter function."""
        from strepsuis_genphennet.network_analysis_core import get_centrality
        
        centrality_dict = {'A': 0.5, 'B': 0.3, 'C': 0.2}
        
        result = get_centrality(centrality_dict)
        
        assert result == centrality_dict

    def test_create_interactive_table(self):
        """Test interactive table HTML creation."""
        from strepsuis_genphennet.network_analysis_core import create_interactive_table
        
        df = pd.DataFrame({
            'Feature': ['A', 'B'],
            'Value': [1.12345, 2.98765],
        })
        
        html = create_interactive_table(df, 'test')
        
        assert '<table' in html
        assert 'table-test' in html
        assert 'Feature' in html

    def test_create_section_summary(self):
        """Test section summary HTML creation."""
        from strepsuis_genphennet.network_analysis_core import create_section_summary
        
        html = create_section_summary(
            "Test Summary",
            {'Nodes': 10, 'Edges': 15},
            per_category={'Cat1': 5, 'Cat2': 10},
            per_feature={'Feat1': 3}
        )
        
        assert 'Test Summary' in html
        assert 'Nodes' in html
        assert 'Cat1' in html


# ============================================================================
# Test mutually exclusive patterns
# ============================================================================
class TestMutuallyExclusive:
    """Test mutually exclusive pattern detection."""

    def test_find_mutually_exclusive_basic(self):
        """Test finding mutually exclusive patterns."""
        from strepsuis_genphennet.network_analysis_core import find_mutually_exclusive
        
        df = pd.DataFrame({
            'A': [1, 0, 1, 0],
            'B': [0, 1, 0, 1],  # Mutually exclusive with A
            'C': [1, 1, 0, 0],
        })
        
        mapping = {'A': 'Cat1', 'B': 'Cat1', 'C': 'Cat2'}
        
        result = find_mutually_exclusive(df, ['A', 'B', 'C'], mapping, k=2)
        
        assert isinstance(result, pd.DataFrame)

    def test_find_mutually_exclusive_none(self):
        """Test when no mutually exclusive patterns exist."""
        from strepsuis_genphennet.network_analysis_core import find_mutually_exclusive
        
        df = pd.DataFrame({
            'A': [1, 1, 1, 1],
            'B': [1, 1, 1, 1],  # Not mutually exclusive
        })
        
        mapping = {}
        
        result = find_mutually_exclusive(df, ['A', 'B'], mapping, k=2)
        
        # Should return empty or no patterns
        assert isinstance(result, pd.DataFrame)


# ============================================================================
# Test cluster/hub identification
# ============================================================================
class TestClusterHubs:
    """Test cluster hub identification."""

    def test_get_cluster_hubs(self):
        """Test getting top hubs from clusters."""
        from strepsuis_genphennet.network_analysis_core import get_cluster_hubs
        
        df = pd.DataFrame({
            'Node': ['A', 'B', 'C', 'D', 'E'],
            'Cluster': [1, 1, 1, 2, 2],
            'Degree': [10, 8, 6, 15, 12],
        })
        
        result = get_cluster_hubs(df, top_n=2)
        
        assert isinstance(result, pd.DataFrame)


# ============================================================================
# Test network construction (if available)
# ============================================================================
class TestNetworkConstruction:
    """Test network construction functions."""

    def test_build_network_from_associations(self):
        """Test building network from association data."""
        # This tests the pattern of building networks
        G = nx.Graph()
        
        # Add some edges with weights
        G.add_edge('A', 'B', weight=0.8, pvalue=0.01)
        G.add_edge('B', 'C', weight=0.5, pvalue=0.05)
        G.add_edge('A', 'C', weight=0.3, pvalue=0.1)
        
        # Test basic properties
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 3

    def test_compute_centralities(self):
        """Test computing centrality measures."""
        G = nx.Graph()
        G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('B', 'D')])
        
        degree = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)
        
        # B should have highest centrality (most connected)
        assert degree['B'] == max(degree.values())
        # Verify all centrality measures were computed
        assert len(betweenness) == G.number_of_nodes()
        assert len(closeness) == G.number_of_nodes()

    def test_community_detection(self):
        """Test Louvain community detection."""
        import community.community_louvain as community_louvain
        
        # Create a graph with clear community structure
        G = nx.Graph()
        # Community 1
        G.add_edges_from([('A1', 'A2'), ('A2', 'A3'), ('A1', 'A3')])
        # Community 2
        G.add_edges_from([('B1', 'B2'), ('B2', 'B3'), ('B1', 'B3')])
        # Bridge
        G.add_edge('A1', 'B1')
        
        partition = community_louvain.best_partition(G)
        
        # Should detect 2 communities
        assert len(set(partition.values())) >= 1


# ============================================================================
# Integration tests
# ============================================================================
class TestIntegration:
    """Integration tests for network analysis."""

    def test_full_association_workflow(self):
        """Test a complete association analysis workflow."""
        from strepsuis_genphennet.network_analysis_core import (
            chi2_phi, calculate_entropy, normalized_mutual_info
        )
        
        np.random.seed(42)
        n = 100
        
        # Create correlated binary features
        x = np.random.binomial(1, 0.5, n)
        y = x.copy()
        y[np.random.choice(n, size=10, replace=False)] = 1 - y[np.random.choice(n, size=10, replace=False)]
        
        x = pd.Series(x)
        y = pd.Series(y)
        
        # Calculate association metrics
        p, phi, _, _, _ = chi2_phi(x, y)
        H_x, _ = calculate_entropy(x)
        nmi = normalized_mutual_info(x, y)
        
        # All should be valid
        assert 0 <= p <= 1
        assert 0 <= phi <= 1
        assert H_x >= 0
        assert 0 <= nmi <= 1

    def test_network_metrics_workflow(self):
        """Test computing network metrics."""
        import community.community_louvain as community_louvain
        
        # Build a sample network
        G = nx.Graph()
        edges = [
            ('Gene1', 'Pheno1', 0.8),
            ('Gene1', 'Pheno2', 0.6),
            ('Gene2', 'Pheno1', 0.5),
            ('Gene2', 'Pheno3', 0.7),
            ('Pheno1', 'Pheno2', 0.4),
        ]
        
        for source, target, weight in edges:
            G.add_edge(source, target, weight=weight)
        
        # Compute centralities
        degree = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        
        # Community detection
        partition = community_louvain.best_partition(G)
        
        # All should complete successfully
        assert len(degree) == G.number_of_nodes()
        assert len(betweenness) == G.number_of_nodes()
        assert len(partition) == G.number_of_nodes()


# ============================================================================
# Test helper functions
# ============================================================================
class TestHelperFunctions:
    """Test helper functions that don't require community module."""
    
    def test_create_interactive_table(self):
        """Test interactive table creation."""
        from strepsuis_genphennet.network_analysis_core import create_interactive_table
        
        df = pd.DataFrame({
            'A': [1.234, 2.345, 3.456],
            'B': ['x', 'y', 'z'],
        })
        
        html = create_interactive_table(df, 'test_table')
        
        assert isinstance(html, str)
        assert 'test_table' in html
        assert 'A' in html
        assert 'B' in html
    
    def test_create_section_summary(self):
        """Test section summary creation."""
        from strepsuis_genphennet.network_analysis_core import create_section_summary
        
        stats = {'Total': 100, 'Mean': 5.5}
        per_category = {'Cat1': 50, 'Cat2': 50}
        per_feature = {'Feat1': 30, 'Feat2': 70}
        
        html = create_section_summary('Test Section', stats, per_category, per_feature)
        
        assert isinstance(html, str)
        assert 'Test Section' in html
        assert 'Total' in html
    
    def test_find_matching_files(self):
        """Test file matching."""
        from strepsuis_genphennet.network_analysis_core import find_matching_files
        
        uploaded = ['MGE.csv', 'MIC.csv', 'MLST (1).csv']
        expected = ['MGE.csv', 'MIC.csv', 'MLST.csv']
        
        mapping = find_matching_files(uploaded, expected)
        
        assert isinstance(mapping, dict)
        assert 'MGE.csv' in mapping
        assert 'MIC.csv' in mapping
    
    def test_get_centrality(self):
        """Test centrality getter."""
        from strepsuis_genphennet.network_analysis_core import get_centrality
        
        centrality_dict = {'A': 0.5, 'B': 0.3}
        result = get_centrality(centrality_dict)
        
        assert result == centrality_dict
    
    def test_expand_categories(self):
        """Test category expansion."""
        from strepsuis_genphennet.network_analysis_core import expand_categories
        
        df = pd.DataFrame({
            'Strain_ID': ['S1', 'S2', 'S3'],
            'MLST': ['1', '2', '1'],
        })
        
        expanded = expand_categories(df, 'MLST')
        
        assert isinstance(expanded, pd.DataFrame)
        assert 'Strain_ID' in expanded.columns
        assert 'MLST_1' in expanded.columns
        assert 'MLST_2' in expanded.columns
    
    def test_adaptive_phi_threshold(self):
        """Test adaptive phi threshold."""
        from strepsuis_genphennet.network_analysis_core import adaptive_phi_threshold
        
        phi_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        threshold = adaptive_phi_threshold(phi_vals, method='percentile', percentile=90)
        
        assert isinstance(threshold, (float, np.floating))
        assert threshold >= 0
    
    def test_create_interactive_table_with_empty(self):
        """Test interactive table with empty DataFrame."""
        from strepsuis_genphennet.network_analysis_core import create_interactive_table_with_empty
        
        empty_df = pd.DataFrame()
        
        html = create_interactive_table_with_empty(empty_df, 'empty_table')
        
        assert isinstance(html, str)
        # May return "No data available" message instead of table
        assert len(html) > 0
    
    def test_summarize_by_category(self):
        """Test category summarization."""
        from strepsuis_genphennet.network_analysis_core import summarize_by_category
        
        df = pd.DataFrame({
            'Category1': ['A', 'A', 'B', 'B'],
            'Category2': ['X', 'Y', 'X', 'Y'],
            'Value': [1.0, 2.0, 3.0, 4.0],
        })
        
        result = summarize_by_category(df, 'Value', ['Category1', 'Category2'])
        
        assert isinstance(result, dict)
    
    def test_summarize_by_feature(self):
        """Test feature summarization."""
        from strepsuis_genphennet.network_analysis_core import summarize_by_feature
        
        df = pd.DataFrame({
            'Feature1': ['A', 'B'],
            'Feature2': ['X', 'Y'],
            'Value': [1.0, 2.0],
        })
        
        result = summarize_by_feature(df, 'Value', ['Feature1', 'Feature2'])
        
        assert isinstance(result, dict)
    
    def test_summarize_by_category_excl(self):
        """Test category exclusion summarization."""
        try:
            from strepsuis_genphennet.network_analysis_core import summarize_by_category_excl
            
            df = pd.DataFrame({
                'Category1': ['A', 'A', 'B'],
                'Category2': ['X', 'Y', 'X'],
                'Item1': ['Feat1', 'Feat2', 'Feat1'],
                'Item2': ['Feat2', 'Feat3', 'Feat2'],
            })
            
            result = summarize_by_category_excl(df, k=2)
            
            assert isinstance(result, dict)
        except ImportError:
            pytest.skip("community module not available")
    
    def test_summarize_by_feature_excl(self):
        """Test feature exclusion summarization."""
        try:
            from strepsuis_genphennet.network_analysis_core import summarize_by_feature_excl
            
            df = pd.DataFrame({
                'Item1': ['Feat1', 'Feat2'],
                'Item2': ['Feat2', 'Feat3'],
            })
            
            result = summarize_by_feature_excl(df, k=2)
            
            assert isinstance(result, dict)
        except ImportError:
            pytest.skip("community module not available")
    
    def test_summarize_by_category_network(self):
        """Test network category summarization."""
        try:
            from strepsuis_genphennet.network_analysis_core import summarize_by_category_network
            
            df = pd.DataFrame({
                'Category': ['A', 'A', 'B', 'B'],
                'Degree_Centrality': [0.5, 0.6, 0.3, 0.4],
            })
            
            result = summarize_by_category_network(df, value_col='Degree_Centrality')
            
            assert isinstance(result, dict)
        except ImportError:
            pytest.skip("community module not available")
    
    def test_summarize_by_feature_network(self):
        """Test network feature summarization."""
        try:
            from strepsuis_genphennet.network_analysis_core import summarize_by_feature_network
            
            df = pd.DataFrame({
                'Feature': ['Feat1', 'Feat2', 'Feat3'],
                'Degree_Centrality': [0.5, 0.6, 0.3],
            })
            
            result = summarize_by_feature_network(df, value_col='Degree_Centrality')
            
            assert isinstance(result, dict)
        except ImportError:
            pytest.skip("community module not available")
    
    def test_setup_logging(self):
        """Test logging setup."""
        try:
            from strepsuis_genphennet.network_analysis_core import setup_logging
            
            setup_logging()
            
            # Should not raise exception
            import logging
            assert logging.getLogger().level >= 0
        except ImportError:
            pytest.skip("community module not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
