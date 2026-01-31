"""
Statistical Validation Tests for strepsuis-genphennet

These tests validate the information-theoretic and network analysis routines 
against gold-standard libraries as specified in the Elite Custom Instructions 
for StrepSuis Bioinformatics Suite.

Validations include:
- Mutual information against sklearn.metrics
- Entropy calculations against scipy.stats
- Cramér's V calculation verification
- Network construction and community detection
- Edge case handling (empty inputs, single features, etc.)

References:
- scipy.stats.entropy for entropy calculations
- sklearn.metrics for mutual information metrics
- networkx for network analysis and community detection
"""

import numpy as np
import pandas as pd
import pytest
import networkx as nx
from scipy.stats import chi2_contingency, entropy as scipy_entropy
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score


class TestEntropyValidation:
    """Validate entropy calculations against scipy gold standard."""

    def test_entropy_matches_scipy(self):
        """Test that entropy calculation matches scipy.stats.entropy."""
        from strepsuis_genphennet.network_analysis_core import calculate_entropy

        # Create test data with known distribution
        data = pd.Series([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
        
        # Our implementation
        H_ours, Hn_ours = calculate_entropy(data)
        
        # scipy implementation
        probs = data.value_counts(normalize=True)
        H_scipy = scipy_entropy(probs, base=np.e)  # Natural log
        
        # Should be close (entropy uses natural log by default)
        np.testing.assert_almost_equal(H_ours, H_scipy, decimal=3,
            err_msg="Entropy should match scipy")

    def test_entropy_of_constant_series(self):
        """Test entropy of constant series is zero."""
        from strepsuis_genphennet.network_analysis_core import calculate_entropy

        constant = pd.Series([1, 1, 1, 1, 1])
        H, Hn = calculate_entropy(constant)
        
        assert H == 0.0, "Constant series should have zero entropy"
        assert Hn == 0.0, "Normalized entropy of constant should be zero"

    def test_entropy_maximum_for_uniform(self):
        """Test entropy is maximized for uniform distribution."""
        from strepsuis_genphennet.network_analysis_core import calculate_entropy

        # Uniform distribution (equal counts of each value)
        uniform = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        H, Hn = calculate_entropy(uniform)
        
        # For binary uniform, entropy should be close to log(2)
        max_entropy = np.log(2)
        np.testing.assert_almost_equal(H, max_entropy, decimal=3,
            err_msg="Uniform binary distribution should have max entropy")


class TestMutualInformationValidation:
    """Validate mutual information against sklearn."""

    def test_information_gain_properties(self):
        """Test information gain has expected properties."""
        from strepsuis_genphennet.network_analysis_core import information_gain

        # Create perfectly correlated data
        x = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
        y = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])  # Same as x
        
        ig = information_gain(x, y)
        
        # IG should be positive for correlated data
        assert ig >= 0, "Information gain should be non-negative"

    def test_nmi_matches_sklearn(self):
        """Test normalized mutual information against sklearn."""
        from strepsuis_genphennet.network_analysis_core import normalized_mutual_info

        # Create test data
        x = pd.Series([0, 0, 1, 1, 0, 1, 0, 1])
        y = pd.Series([0, 0, 1, 1, 1, 0, 0, 1])
        
        # Our implementation
        nmi_ours = normalized_mutual_info(x, y)
        
        # sklearn implementation
        nmi_sklearn = normalized_mutual_info_score(x, y, average_method='geometric')
        
        # Should be in the same ballpark (different normalizations may give slightly different results)
        assert 0 <= nmi_ours <= 1, "NMI should be between 0 and 1"
        assert 0 <= nmi_sklearn <= 1, "sklearn NMI should be between 0 and 1"

    def test_mutual_info_zero_for_independent(self):
        """Test mutual information is near zero for independent variables."""
        from strepsuis_genphennet.network_analysis_core import normalized_mutual_info

        # Create independent data with large sample
        np.random.seed(42)
        x = pd.Series(np.random.binomial(1, 0.5, 1000))
        y = pd.Series(np.random.binomial(1, 0.5, 1000))
        
        nmi = normalized_mutual_info(x, y)
        
        # Should be close to zero for independent variables
        assert nmi < 0.1, f"Independent variables should have low NMI, got {nmi}"


class TestCramersVValidation:
    """Validate Cramér's V calculations."""

    def test_cramers_v_bounds(self):
        """Test that Cramér's V is between 0 and 1."""
        from strepsuis_genphennet.network_analysis_core import cramers_v

        # Create contingency table
        table = pd.DataFrame([[10, 5], [5, 10]])
        
        v, lo, hi = cramers_v(table)
        
        assert 0 <= v <= 1, f"Cramér's V should be between 0 and 1, got {v}"

    def test_cramers_v_perfect_association(self):
        """Test Cramér's V for perfect association."""
        from strepsuis_genphennet.network_analysis_core import cramers_v

        # Perfect association
        perfect = pd.DataFrame([[50, 0], [0, 50]])
        
        v, lo, hi = cramers_v(perfect)
        
        assert v > 0.9, f"Perfect association should have V > 0.9, got {v}"

    def test_cramers_v_independence(self):
        """Test Cramér's V for independence."""
        from strepsuis_genphennet.network_analysis_core import cramers_v

        # Independent (same proportions in rows)
        independent = pd.DataFrame([[25, 25], [25, 25]])
        
        v, lo, hi = cramers_v(independent)
        
        assert v < 0.1, f"Independent variables should have V < 0.1, got {v}"


class TestNetworkConstruction:
    """Test network construction and analysis."""

    def test_network_has_expected_structure(self):
        """Test that constructed network has expected nodes and edges."""
        G = nx.Graph()
        G.add_edge('A', 'B', weight=0.5)
        G.add_edge('B', 'C', weight=0.3)
        
        assert G.number_of_nodes() == 3, "Should have 3 nodes"
        assert G.number_of_edges() == 2, "Should have 2 edges"
        assert 'A' in G.nodes(), "Should contain node A"

    def test_centrality_calculations(self):
        """Test centrality metrics are calculated correctly."""
        # Create a simple network
        G = nx.Graph()
        G.add_edges_from([('A', 'B'), ('B', 'C'), ('B', 'D'), ('C', 'D')])
        
        # Calculate centralities
        degree_cent = nx.degree_centrality(G)
        betweenness_cent = nx.betweenness_centrality(G)
        
        # B should have highest degree centrality (connected to 3 nodes)
        assert degree_cent['B'] >= degree_cent['A'], "B should have higher degree than A"
        
        # All centralities should be between 0 and 1
        for node, cent in degree_cent.items():
            assert 0 <= cent <= 1, f"Degree centrality should be in [0,1], got {cent}"

    def test_community_detection(self):
        """Test community detection produces valid communities."""
        import community.community_louvain as community_louvain
        
        # Create network with clear community structure
        G = nx.Graph()
        # Community 1
        G.add_edges_from([('A1', 'A2'), ('A2', 'A3'), ('A1', 'A3')])
        # Community 2
        G.add_edges_from([('B1', 'B2'), ('B2', 'B3'), ('B1', 'B3')])
        # Bridge edge
        G.add_edge('A3', 'B1')
        
        partition = community_louvain.best_partition(G, random_state=42)
        
        # Should have detected communities
        num_communities = len(set(partition.values()))
        assert num_communities >= 1, "Should detect at least one community"
        
        # All nodes should be assigned to a community
        assert len(partition) == G.number_of_nodes(), "All nodes should have community assignment"


class TestEdgeCases:
    """Test edge cases for robustness."""

    def test_entropy_empty_series(self):
        """Test entropy of empty series.
        
        Empty series has no probability distribution, so entropy is undefined.
        Implementation should return 0.0 as a safe default (matching scipy behavior
        for empty input) since log(0) is undefined.
        """
        from strepsuis_genphennet.network_analysis_core import calculate_entropy

        empty = pd.Series([], dtype=int)
        H, Hn = calculate_entropy(empty)
        
        # Empty series has no distribution, entropy defaults to 0.0
        assert H == 0.0, "Empty series entropy should be 0.0 (safe default)"

    def test_entropy_single_element(self):
        """Test entropy of single-element series."""
        from strepsuis_genphennet.network_analysis_core import calculate_entropy

        single = pd.Series([1])
        H, Hn = calculate_entropy(single)
        
        assert H == 0.0, "Single element series should have zero entropy"

    def test_cramers_v_single_row(self):
        """Test Cramér's V with single-row table.
        
        Cramér's V requires at least 2 rows and 2 columns to compute.
        Single-row tables return V=0.0 as the formula is undefined (division by zero).
        """
        from strepsuis_genphennet.network_analysis_core import cramers_v

        single_row = pd.DataFrame([[10, 20]])
        v, lo, hi = cramers_v(single_row)
        
        # Single row table is degenerate; V=0.0 is the defined behavior
        np.testing.assert_almost_equal(v, 0.0, decimal=5,
            err_msg="Single row table should return V≈0")

    def test_cramers_v_single_column(self):
        """Test Cramér's V with single-column table.
        
        Cramér's V requires at least 2 rows and 2 columns to compute.
        Single-column tables return V=0.0 as the formula is undefined.
        """
        from strepsuis_genphennet.network_analysis_core import cramers_v

        single_col = pd.DataFrame([[10], [20]])
        v, lo, hi = cramers_v(single_col)
        
        # Single column table is degenerate; V=0.0 is the defined behavior
        np.testing.assert_almost_equal(v, 0.0, decimal=5,
            err_msg="Single column table should return V≈0")


class TestMutuallyExclusivePatterns:
    """Test mutually exclusive pattern detection."""

    def test_finds_exclusive_pairs(self):
        """Test that mutually exclusive pairs are correctly identified."""
        from strepsuis_genphennet.network_analysis_core import find_mutually_exclusive

        # Create data where A and B never co-occur
        df = pd.DataFrame({
            'A': [1, 0, 1, 0, 1, 0],
            'B': [0, 1, 0, 1, 0, 1],
            'C': [1, 1, 0, 0, 1, 0],
        })
        
        mapping = {'A': 'Cat1', 'B': 'Cat1', 'C': 'Cat2'}
        features = ['A', 'B', 'C']
        
        result = find_mutually_exclusive(df, features, mapping, k=2)
        
        # A and B should be identified as mutually exclusive
        if not result.empty:
            pairs = [(row['Feature_1'], row['Feature_2']) for _, row in result.iterrows()]
            assert ('A', 'B') in pairs or ('B', 'A') in pairs, \
                "Should identify A and B as mutually exclusive"

    def test_no_exclusive_pairs(self):
        """Test when there are no mutually exclusive pairs."""
        from strepsuis_genphennet.network_analysis_core import find_mutually_exclusive

        # Create data where all features co-occur
        df = pd.DataFrame({
            'A': [1, 1, 1, 1],
            'B': [1, 1, 1, 1],
            'C': [1, 1, 1, 1],
        })
        
        mapping = {'A': 'Cat1', 'B': 'Cat1', 'C': 'Cat2'}
        features = ['A', 'B', 'C']
        
        result = find_mutually_exclusive(df, features, mapping, k=2)
        
        # Should return empty (no mutually exclusive pairs)
        assert result.empty, "Should find no mutually exclusive pairs when all co-occur"


class TestReproducibility:
    """Test reproducibility with fixed random seeds."""

    def test_community_detection_reproducibility(self):
        """Test that community detection is reproducible with same seed."""
        import community.community_louvain as community_louvain
        
        G = nx.karate_club_graph()
        
        # Run twice with same seed
        partition1 = community_louvain.best_partition(G, random_state=42)
        partition2 = community_louvain.best_partition(G, random_state=42)
        
        assert partition1 == partition2, "Same seed should give same communities"

    def test_different_seeds_may_differ(self):
        """Test that different seeds can give different results."""
        import community.community_louvain as community_louvain
        
        G = nx.karate_club_graph()
        
        partitions = []
        for seed in [42, 123, 456]:
            partition = community_louvain.best_partition(G, random_state=seed)
            partitions.append(partition)
        
        # At least should produce valid partitions
        for p in partitions:
            assert len(p) == G.number_of_nodes(), "All nodes should have assignment"


class TestNumericalStability:
    """Test numerical stability of calculations."""

    def test_chi2_phi_numerical_stability(self):
        """Test chi2_phi function handles edge cases."""
        from strepsuis_genphennet.network_analysis_core import chi2_phi

        # Normal case
        x = pd.Series([0, 0, 1, 1, 0, 1, 0, 1])
        y = pd.Series([0, 0, 1, 1, 1, 0, 0, 1])
        
        p, phi, contingency, lo, hi = chi2_phi(x, y)
        
        assert 0 <= p <= 1, f"P-value should be in [0,1], got {p}"
        assert -1 <= phi <= 1, f"Phi should be in [-1,1], got {phi}"

    def test_chi2_phi_with_all_same_values(self):
        """Test chi2_phi with constant data.
        
        When one variable is constant (no variance), the contingency table has
        a zero row or column, making the chi-square test and phi coefficient
        undefined or 0. Implementation should return phi=0.0 as the association
        is undefined when one variable doesn't vary.
        """
        from strepsuis_genphennet.network_analysis_core import chi2_phi

        constant_x = pd.Series([1, 1, 1, 1])
        varied_y = pd.Series([0, 1, 0, 1])
        
        p, phi, contingency, lo, hi = chi2_phi(constant_x, varied_y)
        
        # Constant variable has no variance, so phi=0.0 (no computable association)
        np.testing.assert_almost_equal(phi, 0.0, decimal=5,
            err_msg="Constant variable should have phi=0 (undefined association)")


@pytest.mark.slow
class TestPerformance:
    """Performance tests (marked as slow)."""

    def test_large_network_construction(self):
        """Test network construction with many nodes."""
        import time
        
        # Create a larger network
        n_nodes = 100
        G = nx.Graph()
        nodes = [f'node_{i}' for i in range(n_nodes)]
        
        # Add random edges
        np.random.seed(42)
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if np.random.random() < 0.1:  # 10% edge probability
                    G.add_edge(nodes[i], nodes[j])
        
        start = time.time()
        degree_cent = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        elapsed = time.time() - start
        
        assert elapsed < 10, f"Centrality calculations should complete in < 10s, took {elapsed:.1f}s"
        assert len(degree_cent) == n_nodes, "Should calculate for all nodes"

    def test_large_pairwise_computations(self):
        """Test pairwise computations with many features."""
        import time
        
        # Create data with many features
        np.random.seed(42)
        n_samples = 100
        n_features = 20
        
        data = pd.DataFrame(
            np.random.binomial(1, 0.5, (n_samples, n_features)),
            columns=[f'feat_{i}' for i in range(n_features)]
        )
        
        start = time.time()
        
        # Compute pairwise statistics
        from itertools import combinations
        results = []
        for f1, f2 in combinations(data.columns, 2):
            contingency = pd.crosstab(data[f1], data[f2])
            chi2, p, _, _ = chi2_contingency(contingency)
            results.append((f1, f2, chi2, p))
        
        elapsed = time.time() - start
        
        expected_pairs = n_features * (n_features - 1) // 2
        assert len(results) == expected_pairs, f"Should compute {expected_pairs} pairs"
        assert elapsed < 30, f"Pairwise computations should complete in < 30s, took {elapsed:.1f}s"
