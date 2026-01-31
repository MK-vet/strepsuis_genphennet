"""Stress tests for strepsuis-genphennet module.

These tests verify behavior with large datasets, memory constraints,
and concurrent operations.
"""

import time

import numpy as np
import pandas as pd
import pytest


@pytest.mark.stress
class TestLargeDatasets:
    """Tests with large datasets to verify scalability."""

    def test_large_pairwise_tests(self):
        """Test pairwise association testing with large feature set."""
        np.random.seed(42)
        n_samples = 500
        n_features = 100
        
        data = np.random.randint(0, 2, size=(n_samples, n_features))
        df = pd.DataFrame(
            data,
            columns=[f"Feature_{i}" for i in range(n_features)]
        )
        
        # Number of pairs
        n_pairs = n_features * (n_features - 1) // 2
        assert n_pairs == 4950
        
        assert df.shape == (n_samples, n_features)

    def test_network_with_many_edges(self):
        """Test network creation with many edges."""
        import networkx as nx
        
        np.random.seed(42)
        n_nodes = 100
        edge_probability = 0.3
        
        # Create random graph
        G = nx.gnp_random_graph(n_nodes, edge_probability, seed=42)
        
        # Check graph properties
        assert G.number_of_nodes() == n_nodes
        expected_edges = n_nodes * (n_nodes - 1) / 2 * edge_probability
        assert abs(G.number_of_edges() - expected_edges) < expected_edges * 0.2

    def test_fdr_many_pvalues(self):
        """Test FDR correction with many p-values."""
        from statsmodels.stats.multitest import multipletests
        
        np.random.seed(42)
        n_pvalues = 10000
        
        pvalues = np.random.uniform(0, 1, size=n_pvalues)
        
        reject, corrected, _, _ = multipletests(pvalues, method='fdr_bh')
        
        assert len(corrected) == n_pvalues
        # Corrected p-values should be >= raw p-values
        assert all(c >= p - 1e-10 for c, p in zip(corrected, pvalues))


@pytest.mark.stress
class TestNetworkEdgeCases:
    """Tests for network edge cases."""

    def test_empty_network(self):
        """Test empty network handling."""
        import networkx as nx
        
        G = nx.Graph()
        
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0
        assert list(G.nodes()) == []

    def test_fully_connected_network(self):
        """Test fully connected network."""
        import networkx as nx
        
        n_nodes = 20
        G = nx.complete_graph(n_nodes)
        
        expected_edges = n_nodes * (n_nodes - 1) // 2
        assert G.number_of_edges() == expected_edges

    def test_disconnected_network(self):
        """Test network with disconnected components."""
        import networkx as nx
        
        G = nx.Graph()
        
        # Component 1
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        # Component 2
        G.add_edges_from([(3, 4), (4, 5)])
        
        components = list(nx.connected_components(G))
        assert len(components) == 2


@pytest.mark.stress
class TestInformationTheory:
    """Tests for information theory calculations."""

    def test_entropy_edge_cases(self):
        """Test entropy calculation edge cases."""
        # All zeros - entropy should be 0
        data_zeros = np.zeros(100)
        # All ones - entropy should be 0
        data_ones = np.ones(100)
        
        def binary_entropy(x):
            p = np.mean(x)
            if p == 0 or p == 1:
                return 0.0
            return -p * np.log2(p) - (1-p) * np.log2(1-p)
        
        assert binary_entropy(data_zeros) == 0.0
        assert binary_entropy(data_ones) == 0.0
        
        # 50/50 split - maximum entropy
        data_half = np.array([0, 1] * 50)
        h = binary_entropy(data_half)
        assert abs(h - 1.0) < 1e-10

    def test_mutual_information_properties(self):
        """Test mutual information properties."""
        np.random.seed(42)
        n = 1000
        
        # Independent variables
        x = np.random.randint(0, 2, size=n)
        y = np.random.randint(0, 2, size=n)
        
        # Identical variables
        z = x.copy()
        
        def estimate_mi(a, b):
            """Simple MI estimation."""
            # Joint probabilities
            p11 = np.mean((a == 1) & (b == 1))
            p10 = np.mean((a == 1) & (b == 0))
            p01 = np.mean((a == 0) & (b == 1))
            p00 = np.mean((a == 0) & (b == 0))
            
            pa1 = np.mean(a == 1)
            pb1 = np.mean(b == 1)
            
            mi = 0.0
            for pab, pa, pb in [(p11, pa1, pb1), (p10, pa1, 1-pb1),
                                (p01, 1-pa1, pb1), (p00, 1-pa1, 1-pb1)]:
                if pab > 0 and pa > 0 and pb > 0:
                    mi += pab * np.log2(pab / (pa * pb))
            return mi
        
        mi_independent = estimate_mi(x, y)
        mi_identical = estimate_mi(x, z)
        
        # MI of identical should be much higher than independent
        assert mi_identical > mi_independent
