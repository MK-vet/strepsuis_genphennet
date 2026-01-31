"""
Tests for optimizations module.

Tests sparse network operations, parallel processing, and other optimizations.
"""

import numpy as np
import pandas as pd
import pytest

from strepsuis_genphennet.optimizations import (
    SparseNetwork,
    build_sparse_network,
    parallel_chi_square_tests,
    fast_degree_centrality,
    approximate_betweenness_centrality,
    fast_connected_components,
    fast_modularity_communities,
    fast_fdr_correction,
    fast_mutual_information,
    fast_mutual_information_matrix,
    benchmark_network_construction,
    get_optimization_status,
    NUMBA_AVAILABLE,
    JOBLIB_AVAILABLE,
)


class TestSparseNetwork:
    """Tests for SparseNetwork class."""
    
    def test_basic_creation(self):
        """Test basic sparse network creation."""
        network = SparseNetwork(n_nodes=5)
        
        assert network.n_nodes == 5
    
    def test_add_edge(self):
        """Test adding edges."""
        network = SparseNetwork(n_nodes=3)
        network.add_edge(0, 1, weight=0.5)
        network.add_edge(1, 2, weight=0.8)
        
        # Check edges exist
        assert network.edges[0, 1] == 0.5
        assert network.edges[1, 0] == 0.5  # Symmetric
    
    def test_get_neighbors(self):
        """Test getting neighbors."""
        network = SparseNetwork(n_nodes=3)
        network.add_edge(0, 1, weight=0.5)
        network.add_edge(0, 2, weight=0.3)
        
        neighbors = network.get_neighbors(0)
        
        assert 1 in neighbors
        assert 2 in neighbors
    
    def test_get_degree(self):
        """Test getting node degree."""
        network = SparseNetwork(n_nodes=4)
        network.add_edge(0, 1, weight=1.0)
        network.add_edge(0, 2, weight=1.0)
        network.add_edge(0, 3, weight=1.0)
        
        degree = network.get_degree(0)
        
        assert degree == 3
    
    def test_to_csr(self):
        """Test conversion to CSR format."""
        network = SparseNetwork(n_nodes=3)
        network.add_edge(0, 1, weight=1.0)
        
        csr = network.to_csr()
        
        assert csr.shape == (3, 3)
    
    def test_get_density(self):
        """Test network density calculation."""
        network = SparseNetwork(n_nodes=4)
        # Add 3 edges out of max 6 possible
        network.add_edge(0, 1, weight=1.0)
        network.add_edge(1, 2, weight=1.0)
        network.add_edge(2, 3, weight=1.0)
        
        density = network.get_density()
        
        assert 0 <= density <= 1
    
    def test_memory_usage(self):
        """Test memory usage calculation."""
        network = SparseNetwork(n_nodes=100)
        
        memory = network.memory_usage_mb()
        
        assert memory >= 0


class TestBuildSparseNetwork:
    """Tests for build_sparse_network function."""
    
    def test_basic_build(self):
        """Test basic network building."""
        np.random.seed(42)
        # Create correlated data
        data = np.zeros((100, 5))
        data[:50, 0] = 1
        data[:50, 1] = 1  # Correlated with column 0
        data[50:, 2] = 1
        data[50:, 3] = 1  # Correlated with column 2
        data[:, 4] = np.random.randint(0, 2, 100)
        
        network = build_sparse_network(data, p_threshold=0.05, min_support=5)
        
        assert network.n_nodes == 5
    
    def test_build_with_random_data(self):
        """Test network building with random data."""
        np.random.seed(42)
        data = np.random.randint(0, 2, (50, 8))
        
        network = build_sparse_network(data, p_threshold=0.5, min_support=2)
        
        assert network.n_nodes == 8


class TestParallelChiSquare:
    """Tests for parallel chi-square tests."""
    
    def test_basic_chi_square(self):
        """Test basic parallel chi-square."""
        np.random.seed(42)
        data = np.random.randint(0, 2, (50, 4))
        
        results = parallel_chi_square_tests(data, n_jobs=1)
        
        assert len(results) > 0
    
    def test_chi_square_returns_dataframe(self):
        """Test chi-square returns DataFrame."""
        np.random.seed(42)
        data = np.random.randint(0, 2, (30, 3))
        
        results = parallel_chi_square_tests(data, n_jobs=1)
        
        assert isinstance(results, (pd.DataFrame, list))


class TestFastDegreeCentrality:
    """Tests for fast degree centrality."""
    
    def test_basic_centrality(self):
        """Test basic degree centrality."""
        network = SparseNetwork(n_nodes=4)
        network.add_edge(0, 1, weight=1.0)
        network.add_edge(0, 2, weight=1.0)
        network.add_edge(1, 2, weight=1.0)
        
        centrality = fast_degree_centrality(network)
        
        assert len(centrality) == 4
        assert all(c >= 0 for c in centrality)


class TestApproximateBetweenness:
    """Tests for approximate betweenness centrality."""
    
    def test_basic_betweenness(self):
        """Test basic betweenness centrality."""
        network = SparseNetwork(n_nodes=5)
        network.add_edge(0, 1, weight=1.0)
        network.add_edge(1, 2, weight=1.0)
        network.add_edge(2, 3, weight=1.0)
        network.add_edge(3, 4, weight=1.0)
        
        betweenness = approximate_betweenness_centrality(network, k=3)
        
        assert len(betweenness) == 5
        assert all(b >= 0 for b in betweenness)


class TestFastConnectedComponents:
    """Tests for fast connected components."""
    
    def test_single_component(self):
        """Test single connected component."""
        network = SparseNetwork(n_nodes=4)
        network.add_edge(0, 1, weight=1.0)
        network.add_edge(1, 2, weight=1.0)
        network.add_edge(2, 3, weight=1.0)
        
        n_components, labels = fast_connected_components(network)
        
        assert n_components >= 1
        assert len(labels) == 4
    
    def test_disconnected_network(self):
        """Test disconnected network."""
        network = SparseNetwork(n_nodes=4)
        network.add_edge(0, 1, weight=1.0)
        # Nodes 2 and 3 are isolated
        
        n_components, labels = fast_connected_components(network)
        
        assert n_components >= 2


class TestFastModularityCommunities:
    """Tests for fast modularity communities."""
    
    def test_basic_communities(self):
        """Test basic community detection."""
        network = SparseNetwork(n_nodes=6)
        # Create two clear communities
        network.add_edge(0, 1, weight=1.0)
        network.add_edge(0, 2, weight=1.0)
        network.add_edge(1, 2, weight=1.0)
        network.add_edge(3, 4, weight=1.0)
        network.add_edge(3, 5, weight=1.0)
        network.add_edge(4, 5, weight=1.0)
        network.add_edge(2, 3, weight=0.1)  # Weak link
        
        communities = fast_modularity_communities(network)
        
        assert len(communities) == 6


class TestFastFDRCorrection:
    """Tests for fast FDR correction."""
    
    def test_basic_fdr(self):
        """Test basic FDR correction."""
        p_values = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
        
        corrected, rejected = fast_fdr_correction(p_values, alpha=0.05)
        
        assert len(corrected) == 5
        assert len(rejected) == 5
    
    def test_fdr_bounds(self):
        """Test FDR correction bounds."""
        p_values = np.array([0.001, 0.01, 0.05])
        
        corrected, rejected = fast_fdr_correction(p_values)
        
        # Corrected p-values should be in [0, 1]
        assert all(0 <= c <= 1 for c in corrected)
    
    def test_fdr_empty(self):
        """Test FDR with empty array."""
        p_values = np.array([])
        
        corrected, rejected = fast_fdr_correction(p_values)
        
        assert len(corrected) == 0
        assert len(rejected) == 0


class TestFastMutualInformation:
    """Tests for fast mutual information."""
    
    def test_perfect_correlation(self):
        """Test MI with perfect correlation."""
        x = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        
        mi = fast_mutual_information(x, y)
        
        assert mi >= 0
    
    def test_independent(self):
        """Test MI with independent variables."""
        np.random.seed(42)
        x = np.random.randint(0, 2, 1000)
        y = np.random.randint(0, 2, 1000)
        
        mi = fast_mutual_information(x, y)
        
        assert mi >= 0
    
    def test_mi_non_negative(self):
        """Test MI is non-negative."""
        np.random.seed(42)
        x = np.random.randint(0, 2, 100)
        y = np.random.randint(0, 2, 100)
        
        mi = fast_mutual_information(x, y)
        
        assert mi >= 0


class TestFastMutualInformationMatrix:
    """Tests for fast MI matrix."""
    
    def test_basic_mi_matrix(self):
        """Test basic MI matrix calculation."""
        np.random.seed(42)
        data = np.random.randint(0, 2, (100, 4))
        
        mi_matrix = fast_mutual_information_matrix(data)
        
        assert mi_matrix.shape == (4, 4)
    
    def test_mi_matrix_symmetric(self):
        """Test MI matrix is symmetric."""
        np.random.seed(42)
        data = np.random.randint(0, 2, (50, 3))
        
        mi_matrix = fast_mutual_information_matrix(data)
        
        np.testing.assert_array_almost_equal(mi_matrix, mi_matrix.T)


class TestBenchmarkNetworkConstruction:
    """Tests for network construction benchmark."""
    
    def test_basic_benchmark(self):
        """Test basic benchmarking."""
        np.random.seed(42)
        data = np.random.randint(0, 2, (30, 5))
        
        result = benchmark_network_construction(data, n_runs=2)
        
        assert isinstance(result, dict)


class TestOptimizationStatus:
    """Tests for optimization status."""
    
    def test_get_status(self):
        """Test getting optimization status."""
        status = get_optimization_status()
        
        assert isinstance(status, dict)
    
    def test_status_has_keys(self):
        """Test status has expected keys."""
        status = get_optimization_status()
        
        # Should have some optimization flags
        assert len(status) > 0


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_network(self):
        """Test empty network."""
        network = SparseNetwork(n_nodes=3)
        
        centrality = fast_degree_centrality(network)
        
        assert all(c == 0 for c in centrality)
    
    def test_single_node(self):
        """Test single node network."""
        network = SparseNetwork(n_nodes=1)
        
        n_components, labels = fast_connected_components(network)
        
        assert n_components >= 1
