#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Tests for optimizations.py

Target: Increase coverage from 0% to 70%+

Critical Coverage Areas:
- SparseNetwork class
- build_sparse_network()
- parallel_chi_square_tests()
- Fast centrality metrics
- approximate_betweenness_centrality()
- fast_modularity_communities()
- fast_fdr_correction()
- fast_mutual_information()
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from scipy import sparse

# Add module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strepsuis_genphennet.optimizations import (
    SparseNetwork,
    build_sparse_network,
    parallel_chi_square_tests,
    cached_degree_centrality,
    fast_degree_centrality,
    approximate_betweenness_centrality,
    fast_connected_components,
    fast_modularity_communities,
    fast_fdr_correction,
    fast_mutual_information,
    fast_mutual_information_matrix,
    benchmark_network_construction,
    get_optimization_status
)


class TestSparseNetwork:
    """Test SparseNetwork class."""

    def test_initialization(self):
        """Test sparse network initialization."""
        network = SparseNetwork(n_nodes=10)

        assert network.n_nodes == 10
        assert network.edges.shape == (10, 10)
        assert len(network.node_names) == 10
        assert isinstance(network.edge_attributes, dict)

    def test_add_edge_basic(self):
        """Test adding edges to sparse network."""
        network = SparseNetwork(n_nodes=5)

        network.add_edge(0, 1, weight=0.8, p_value=0.01, statistic=10.5)

        # Check edge exists (undirected)
        assert network.edges[0, 1] == 0.8
        assert network.edges[1, 0] == 0.8

        # Check attributes
        edge_key = (0, 1)
        assert edge_key in network.edge_attributes
        assert network.edge_attributes[edge_key]['weight'] == 0.8
        assert network.edge_attributes[edge_key]['p_value'] == 0.01
        assert network.edge_attributes[edge_key]['statistic'] == 10.5

    def test_add_edge_reverse_key(self):
        """Test edge attributes use consistent key ordering."""
        network = SparseNetwork(n_nodes=5)

        network.add_edge(3, 1, weight=0.7)

        # Should use (min, max) ordering
        assert (1, 3) in network.edge_attributes
        assert network.edge_attributes[(1, 3)]['weight'] == 0.7

    def test_to_csr(self):
        """Test conversion to CSR format."""
        network = SparseNetwork(n_nodes=5)
        network.add_edge(0, 1, weight=0.8)
        network.add_edge(1, 2, weight=0.6)

        csr = network.to_csr()

        assert isinstance(csr, sparse.csr_matrix)
        assert csr.shape == (5, 5)
        assert csr[0, 1] == 0.8
        assert csr[1, 2] == 0.6

    def test_get_degree(self):
        """Test degree calculation for nodes."""
        network = SparseNetwork(n_nodes=5)
        network.add_edge(0, 1, weight=1.0)
        network.add_edge(0, 2, weight=1.0)
        network.add_edge(0, 3, weight=1.0)

        degree_0 = network.get_degree(0)
        degree_1 = network.get_degree(1)
        degree_4 = network.get_degree(4)

        assert degree_0 == 3
        assert degree_1 == 1
        assert degree_4 == 0

    def test_get_neighbors(self):
        """Test neighbor retrieval."""
        network = SparseNetwork(n_nodes=5)
        network.add_edge(0, 1, weight=1.0)
        network.add_edge(0, 2, weight=1.0)
        network.add_edge(1, 3, weight=1.0)

        neighbors_0 = network.get_neighbors(0)
        neighbors_1 = network.get_neighbors(1)

        assert set(neighbors_0) == {1, 2}
        assert set(neighbors_1) == {0, 3}

    def test_get_density(self):
        """Test network density calculation."""
        network = SparseNetwork(n_nodes=4)
        # Complete graph: 6 edges (4 choose 2)
        network.add_edge(0, 1, weight=1.0)
        network.add_edge(0, 2, weight=1.0)
        network.add_edge(0, 3, weight=1.0)
        network.add_edge(1, 2, weight=1.0)
        network.add_edge(1, 3, weight=1.0)
        network.add_edge(2, 3, weight=1.0)

        density = network.get_density()

        assert density == pytest.approx(1.0, abs=0.01)  # Complete graph

    def test_get_density_empty(self):
        """Test density for empty network."""
        network = SparseNetwork(n_nodes=5)

        density = network.get_density()

        assert density == 0.0

    def test_memory_usage(self):
        """Test memory usage calculation."""
        network = SparseNetwork(n_nodes=100)
        for i in range(50):
            network.add_edge(i, i + 1, weight=1.0)

        memory_mb = network.memory_usage_mb()

        assert isinstance(memory_mb, float)
        assert memory_mb > 0


class TestBuildSparseNetwork:
    """Test sparse network construction from data."""

    def test_build_sparse_network_basic(self):
        """Test building network from binary data."""
        # Create simple binary data
        data = np.array([
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 1]
        ])

        network = build_sparse_network(data, p_threshold=0.5, min_support=2)

        assert isinstance(network, SparseNetwork)
        assert network.n_nodes == 4

    def test_build_sparse_network_with_significance(self):
        """Test network construction with significance filtering."""
        # Create data with clear associations
        np.random.seed(42)
        n_samples = 100
        data = np.zeros((n_samples, 5))

        # Feature 0 and 1 are correlated
        data[:50, 0] = 1
        data[:50, 1] = 1

        # Feature 2 and 3 are correlated
        data[25:75, 2] = 1
        data[25:75, 3] = 1

        network = build_sparse_network(data, p_threshold=0.05, min_support=10)

        # Should create edges between correlated features
        assert network.get_degree(0) > 0 or network.get_degree(1) > 0

    def test_build_sparse_network_min_support(self):
        """Test minimum support filtering."""
        data = np.array([
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])

        network = build_sparse_network(data, p_threshold=0.99, min_support=5)

        # All co-occurrences are below min_support
        assert network.get_density() == 0.0


class TestParallelProcessing:
    """Test parallel chi-square testing."""

    def test_parallel_chi_square_tests_basic(self):
        """Test parallel chi-square testing."""
        np.random.seed(42)
        data = np.random.binomial(1, 0.5, (50, 5))

        results = parallel_chi_square_tests(data, n_jobs=1)

        assert isinstance(results, pd.DataFrame)
        assert 'feature1' in results.columns
        assert 'feature2' in results.columns
        assert 'chi2' in results.columns
        assert 'p_value' in results.columns
        assert 'phi' in results.columns

        # Should have 5 choose 2 = 10 pairs
        assert len(results) == 10

    def test_parallel_chi_square_tests_multicore(self):
        """Test parallel processing with multiple cores."""
        np.random.seed(43)
        data = np.random.binomial(1, 0.5, (50, 6))

        # Test with n_jobs=-1 (all cores)
        results = parallel_chi_square_tests(data, n_jobs=-1)

        assert len(results) == 15  # 6 choose 2

    def test_parallel_chi_square_tests_values(self):
        """Test chi-square test result values."""
        # Create data with known association
        data = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [0, 0, 1]
        ])

        results = parallel_chi_square_tests(data, n_jobs=1)

        # Feature 0 and 1 should be perfectly correlated
        row_0_1 = results[(results['feature1'] == 0) & (results['feature2'] == 1)]
        if len(row_0_1) > 0:
            assert row_0_1['p_value'].values[0] < 0.05
            assert abs(row_0_1['phi'].values[0]) > 0.5


class TestFastCentrality:
    """Test fast centrality calculations."""

    def test_cached_degree_centrality(self):
        """Test cached degree centrality."""
        edges = ((0, 1), (1, 2), (2, 3), (0, 3))
        n_nodes = 4

        centrality = cached_degree_centrality(edges, n_nodes)

        assert isinstance(centrality, dict)
        # Node 0, 1, 2, 3 all have degree 2
        assert all(cent == 2/3 for cent in centrality.values())

    def test_fast_degree_centrality(self):
        """Test fast degree centrality using sparse operations."""
        network = SparseNetwork(n_nodes=5)
        network.add_edge(0, 1, weight=1.0)
        network.add_edge(0, 2, weight=1.0)
        network.add_edge(1, 2, weight=1.0)

        centrality = fast_degree_centrality(network)

        assert isinstance(centrality, np.ndarray)
        assert len(centrality) == 5
        assert centrality[0] > centrality[3]  # Node 0 has higher degree

    def test_fast_degree_centrality_empty(self):
        """Test degree centrality for empty network."""
        network = SparseNetwork(n_nodes=3)

        centrality = fast_degree_centrality(network)

        assert np.all(centrality == 0.0)

    def test_approximate_betweenness_centrality(self):
        """Test approximate betweenness centrality."""
        network = SparseNetwork(n_nodes=6)
        # Create path: 0-1-2-3-4-5
        for i in range(5):
            network.add_edge(i, i + 1, weight=1.0)

        betweenness = approximate_betweenness_centrality(
            network,
            k=6,
            random_state=42
        )

        assert isinstance(betweenness, np.ndarray)
        assert len(betweenness) == 6
        # Middle nodes should have higher betweenness
        assert betweenness[2] > betweenness[0]
        assert betweenness[3] > betweenness[5]

    def test_approximate_betweenness_centrality_small_k(self):
        """Test approximate betweenness with limited sampling."""
        network = SparseNetwork(n_nodes=10)
        for i in range(9):
            network.add_edge(i, i + 1, weight=1.0)

        betweenness = approximate_betweenness_centrality(
            network,
            k=3,  # Sample only 3 nodes
            random_state=42
        )

        assert len(betweenness) == 10
        assert np.all(betweenness >= 0)


class TestCommunityDetection:
    """Test fast community detection algorithms."""

    def test_fast_connected_components(self):
        """Test connected components identification."""
        network = SparseNetwork(n_nodes=7)
        # Component 1: 0-1-2
        network.add_edge(0, 1, weight=1.0)
        network.add_edge(1, 2, weight=1.0)
        # Component 2: 3-4
        network.add_edge(3, 4, weight=1.0)
        # Component 3: 5-6
        network.add_edge(5, 6, weight=1.0)

        n_components, labels = fast_connected_components(network)

        assert n_components == 3
        assert len(labels) == 7
        # Nodes in same component should have same label
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4]
        assert labels[5] == labels[6]

    def test_fast_modularity_communities(self):
        """Test label propagation community detection."""
        network = SparseNetwork(n_nodes=6)
        # Dense community 1: 0-1-2
        network.add_edge(0, 1, weight=1.0)
        network.add_edge(1, 2, weight=1.0)
        network.add_edge(0, 2, weight=1.0)
        # Dense community 2: 3-4-5
        network.add_edge(3, 4, weight=1.0)
        network.add_edge(4, 5, weight=1.0)
        network.add_edge(3, 5, weight=1.0)
        # Weak inter-community link
        network.add_edge(2, 3, weight=0.1)

        labels = fast_modularity_communities(network, resolution=1.0)

        assert isinstance(labels, np.ndarray)
        assert len(labels) == 6

    def test_fast_modularity_communities_isolated_nodes(self):
        """Test community detection with isolated nodes."""
        network = SparseNetwork(n_nodes=5)
        network.add_edge(0, 1, weight=1.0)
        # Nodes 2, 3, 4 are isolated

        labels = fast_modularity_communities(network)

        assert len(labels) == 5


class TestFDRCorrection:
    """Test fast FDR correction."""

    def test_fast_fdr_correction_basic(self):
        """Test Benjamini-Hochberg FDR correction."""
        p_values = np.array([0.001, 0.01, 0.05, 0.1, 0.5])

        reject, corrected = fast_fdr_correction(p_values, alpha=0.05)

        assert isinstance(reject, np.ndarray)
        assert isinstance(corrected, np.ndarray)
        assert len(reject) == len(p_values)
        assert len(corrected) == len(p_values)
        assert reject[0] == True  # 0.001 should be significant
        assert reject[-1] == False  # 0.5 should not be significant

    def test_fast_fdr_correction_all_significant(self):
        """Test FDR correction when all p-values are significant."""
        p_values = np.array([0.001, 0.002, 0.003, 0.004, 0.005])

        reject, corrected = fast_fdr_correction(p_values, alpha=0.05)

        assert np.all(reject)

    def test_fast_fdr_correction_none_significant(self):
        """Test FDR correction when no p-values are significant."""
        p_values = np.array([0.6, 0.7, 0.8, 0.9, 0.95])

        reject, corrected = fast_fdr_correction(p_values, alpha=0.05)

        assert np.all(~reject)

    def test_fast_fdr_correction_monotonicity(self):
        """Test that corrected p-values are monotonic."""
        p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        _, corrected = fast_fdr_correction(p_values, alpha=0.05)

        # Corrected p-values should be non-decreasing
        assert np.all(np.diff(corrected) >= -1e-10)  # Allow small numerical errors

    def test_fast_fdr_correction_bounds(self):
        """Test that corrected p-values are bounded [0, 1]."""
        p_values = np.random.random(100)

        _, corrected = fast_fdr_correction(p_values, alpha=0.05)

        assert np.all(corrected >= 0)
        assert np.all(corrected <= 1)


class TestMutualInformation:
    """Test fast mutual information calculations."""

    def test_fast_mutual_information_independent(self):
        """Test MI for independent variables."""
        np.random.seed(42)
        x = np.random.binomial(1, 0.5, 100)
        y = np.random.binomial(1, 0.5, 100)

        mi = fast_mutual_information(x, y)

        assert isinstance(mi, float)
        # Independent variables should have low MI
        assert mi < 0.3

    def test_fast_mutual_information_identical(self):
        """Test MI for identical variables."""
        x = np.array([0, 1, 0, 1, 0, 1] * 10)

        mi = fast_mutual_information(x, x)

        # MI(X, X) = H(X), should be positive for non-constant
        assert mi > 0

    def test_fast_mutual_information_correlated(self):
        """Test MI for correlated variables."""
        x = np.array([0, 0, 1, 1] * 25)
        y = x.copy()  # Perfect correlation

        mi = fast_mutual_information(x, y)

        # Perfect correlation should give high MI
        assert mi > 0.5

    def test_fast_mutual_information_constant(self):
        """Test MI for constant variable."""
        x = np.ones(100)
        y = np.random.binomial(1, 0.5, 100)

        mi = fast_mutual_information(x, y)

        # Constant variable has no information
        assert mi == 0.0 or mi < 0.01

    def test_fast_mutual_information_matrix(self):
        """Test MI matrix calculation."""
        np.random.seed(42)
        data = np.random.binomial(1, 0.5, (50, 4))

        mi_matrix = fast_mutual_information_matrix(data)

        assert isinstance(mi_matrix, np.ndarray)
        assert mi_matrix.shape == (4, 4)
        # Diagonal should be non-negative (MI with self >= 0)
        for i in range(4):
            assert mi_matrix[i, i] >= 0
        # Symmetric
        assert np.allclose(mi_matrix, mi_matrix.T)
        # All MI values should be non-negative
        assert np.all(mi_matrix >= 0)


class TestBenchmarking:
    """Test benchmarking utilities."""

    def test_benchmark_network_construction(self):
        """Test network construction benchmarking."""
        np.random.seed(42)
        data = np.random.binomial(1, 0.3, (30, 10))

        results = benchmark_network_construction(data, n_runs=3)

        assert isinstance(results, dict)
        assert 'sparse_network' in results
        assert 'mean_ms' in results['sparse_network']
        assert 'std_ms' in results['sparse_network']
        assert 'memory_mb' in results['sparse_network']
        assert results['sparse_network']['mean_ms'] > 0

    def test_get_optimization_status(self):
        """Test optimization availability status."""
        status = get_optimization_status()

        assert isinstance(status, dict)
        assert 'numba_jit' in status
        assert 'parallel_processing' in status
        assert 'sparse_networks' in status
        assert 'fast_fdr' in status
        assert 'approximate_centrality' in status

        # These should always be True
        assert status['sparse_networks'] is True
        assert status['fast_fdr'] is True
        assert status['approximate_centrality'] is True


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_sparse_network_single_node(self):
        """Test network with single node."""
        network = SparseNetwork(n_nodes=1)

        assert network.n_nodes == 1
        assert network.get_degree(0) == 0
        assert network.get_density() == 0.0

    def test_fast_degree_centrality_single_node(self):
        """Test degree centrality with single node."""
        network = SparseNetwork(n_nodes=1)

        centrality = fast_degree_centrality(network)

        # Normalization by (n-1) with n=1 should be handled
        assert len(centrality) == 1

    def test_approximate_betweenness_disconnected(self):
        """Test betweenness on disconnected graph."""
        network = SparseNetwork(n_nodes=4)
        network.add_edge(0, 1, weight=1.0)
        # Nodes 2 and 3 are disconnected

        betweenness = approximate_betweenness_centrality(network, k=4, random_state=42)

        assert len(betweenness) == 4
        assert betweenness[2] == 0.0  # Isolated node
        assert betweenness[3] == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
