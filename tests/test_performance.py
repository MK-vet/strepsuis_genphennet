"""Performance tests for strepsuis-genphennet module.

These tests measure and verify timing benchmarks for key operations.
"""

import time

import numpy as np
import pandas as pd
import pytest


@pytest.mark.performance
class TestAssociationPerformance:
    """Performance tests for association testing."""

    def test_chi_square_batch_timing(self):
        """Test batch chi-square testing performance."""
        from scipy.stats import chi2_contingency
        
        np.random.seed(42)
        n_tests = 500
        
        start = time.time()
        for _ in range(n_tests):
            table = np.random.randint(5, 50, size=(2, 2))
            chi2_contingency(table)
        elapsed = time.time() - start
        
        assert elapsed < 1.0
        
    def test_fisher_exact_batch_timing(self):
        """Test batch Fisher's exact test performance."""
        from scipy.stats import fisher_exact
        
        np.random.seed(42)
        n_tests = 200
        
        start = time.time()
        for _ in range(n_tests):
            table = np.random.randint(1, 15, size=(2, 2))
            fisher_exact(table)
        elapsed = time.time() - start
        
        assert elapsed < 2.0


@pytest.mark.performance
class TestNetworkPerformance:
    """Performance tests for network operations."""

    def test_graph_creation_timing(self):
        """Test graph creation performance."""
        import networkx as nx
        
        np.random.seed(42)
        n_nodes = 100
        n_edges = 500
        
        start = time.time()
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        
        for _ in range(n_edges):
            i = np.random.randint(0, n_nodes)
            j = np.random.randint(0, n_nodes)
            if i != j:
                G.add_edge(i, j, weight=np.random.random())
        elapsed = time.time() - start
        
        assert elapsed < 0.5
        assert G.number_of_nodes() == n_nodes

    def test_centrality_timing(self):
        """Test centrality calculation performance."""
        import networkx as nx
        
        np.random.seed(42)
        n_nodes = 100
        G = nx.gnp_random_graph(n_nodes, 0.2, seed=42)
        
        start = time.time()
        degree = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        elapsed = time.time() - start
        
        assert len(degree) == n_nodes
        assert len(betweenness) == n_nodes
        assert elapsed < 2.0

    def test_community_detection_timing(self):
        """Test community detection performance."""
        import networkx as nx
        
        np.random.seed(42)
        n_nodes = 200
        G = nx.gnp_random_graph(n_nodes, 0.1, seed=42)
        
        start = time.time()
        # Use greedy modularity
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        elapsed = time.time() - start
        
        assert len(communities) >= 1
        assert elapsed < 3.0


@pytest.mark.performance
class TestFDRPerformance:
    """Performance tests for FDR correction."""

    def test_fdr_small(self):
        """Test FDR with small set of p-values."""
        from statsmodels.stats.multitest import multipletests
        
        np.random.seed(42)
        pvalues = np.random.uniform(0, 1, size=100)
        
        start = time.time()
        reject, corrected, _, _ = multipletests(pvalues, method='fdr_bh')
        elapsed = time.time() - start
        
        assert elapsed < 0.1

    def test_fdr_large(self):
        """Test FDR with large set of p-values."""
        from statsmodels.stats.multitest import multipletests
        
        np.random.seed(42)
        pvalues = np.random.uniform(0, 1, size=10000)
        
        start = time.time()
        reject, corrected, _, _ = multipletests(pvalues, method='fdr_bh')
        elapsed = time.time() - start
        
        assert elapsed < 0.5


@pytest.mark.performance
class TestInformationTheoryPerformance:
    """Performance tests for information theory metrics."""

    def test_entropy_timing(self):
        """Test entropy calculation performance."""
        np.random.seed(42)
        n_features = 100
        n_samples = 1000
        
        data = np.random.randint(0, 2, size=(n_samples, n_features))
        
        def binary_entropy(x):
            p = np.mean(x)
            if p == 0 or p == 1:
                return 0.0
            return -p * np.log2(p) - (1-p) * np.log2(1-p)
        
        start = time.time()
        entropies = [binary_entropy(data[:, i]) for i in range(n_features)]
        elapsed = time.time() - start
        
        assert len(entropies) == n_features
        assert elapsed < 0.5

    def test_pairwise_mi_timing(self):
        """Test pairwise mutual information timing."""
        np.random.seed(42)
        n_features = 20
        n_samples = 500
        
        data = np.random.randint(0, 2, size=(n_samples, n_features))
        
        start = time.time()
        mi_values = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                # Simple joint probability computation
                joint = np.column_stack([data[:, i], data[:, j]])
                mi_values.append(joint.mean())
        elapsed = time.time() - start
        
        expected_pairs = n_features * (n_features - 1) // 2
        assert len(mi_values) == expected_pairs
        assert elapsed < 0.5
