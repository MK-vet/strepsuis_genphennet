#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for causal discovery framework.

Tests conditional independence testing and causal network discovery.
"""

import numpy as np
import pandas as pd
import pytest

from strepsuis_genphennet.causal_discovery import CausalDiscoveryFramework


class TestCausalDiscoveryFramework:
    """Test CausalDiscoveryFramework class."""
    
    @pytest.fixture
    def sample_gene_data(self):
        """Create sample gene data."""
        np.random.seed(42)
        n_samples = 100
        n_genes = 10
        
        data = {}
        for i in range(n_genes):
            data[f"Gene_{i}"] = np.random.binomial(1, 0.3, n_samples)
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_phenotype_data(self):
        """Create sample phenotype data."""
        np.random.seed(42)
        n_samples = 100
        n_phenos = 5
        
        data = {}
        for i in range(n_phenos):
            data[f"Pheno_{i}"] = np.random.binomial(1, 0.4, n_samples)
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_initial_associations(self):
        """Create sample initial associations."""
        return pd.DataFrame({
            'Feature1': ['Gene_0', 'Gene_1', 'Gene_2'],
            'Feature2': ['Pheno_0', 'Pheno_1', 'Pheno_2'],
            'Phi': [0.5, 0.6, 0.4],
            'FDR_corrected_p': [0.01, 0.02, 0.03],
            'Significant': [True, True, True],
        })
    
    def test_initialization(self, sample_gene_data, sample_phenotype_data):
        """Test framework initialization."""
        framework = CausalDiscoveryFramework(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data,
        )
        
        assert framework.gene_data.shape[0] == sample_gene_data.shape[0]
        assert framework.pheno_data.shape[0] == sample_phenotype_data.shape[0]
        assert framework.combined_data.shape[0] == sample_gene_data.shape[0]
    
    def test_initialization_with_associations(
        self, sample_gene_data, sample_phenotype_data, sample_initial_associations
    ):
        """Test initialization with initial associations."""
        framework = CausalDiscoveryFramework(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data,
            initial_associations=sample_initial_associations,
        )
        
        assert framework.initial_associations is not None
        assert len(framework.initial_associations) == 3
    
    def test_mutual_information(self, sample_gene_data, sample_phenotype_data):
        """Test mutual information calculation."""
        framework = CausalDiscoveryFramework(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data,
        )
        
        X = sample_gene_data['Gene_0'].values
        Y = sample_phenotype_data['Pheno_0'].values
        
        mi = framework._mutual_information(X, Y)
        
        assert isinstance(mi, float)
        assert mi >= 0.0
        assert not np.isnan(mi)
    
    def test_conditional_independence_test(
        self, sample_gene_data, sample_phenotype_data
    ):
        """Test conditional independence testing."""
        framework = CausalDiscoveryFramework(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data,
        )
        
        # Test with small number of permutations for speed
        is_independent, p_value = framework.test_conditional_independence(
            X='Gene_0',
            Y='Pheno_0',
            conditioning_genes=sample_gene_data.drop(columns=['Gene_0']),
            n_permutations=10,  # Reduced for testing
        )
        
        assert isinstance(is_independent, bool)
        assert isinstance(p_value, float)
        assert 0.0 <= p_value <= 1.0
        # Note: is_independent can be True or False - both are valid results
    
    def test_discover_causal_network_empty_associations(
        self, sample_gene_data, sample_phenotype_data
    ):
        """Test causal network discovery with empty associations."""
        framework = CausalDiscoveryFramework(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data,
            initial_associations=pd.DataFrame(),
        )
        
        result = framework.discover_causal_network(alpha=0.05, n_permutations=100)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_discover_causal_network_with_associations(
        self, sample_gene_data, sample_phenotype_data, sample_initial_associations
    ):
        """Test causal network discovery with associations."""
        framework = CausalDiscoveryFramework(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data,
            initial_associations=sample_initial_associations,
        )
        
        result = framework.discover_causal_network(alpha=0.05, n_permutations=50)
        
        assert isinstance(result, pd.DataFrame)
        # Should have columns: gene, phenotype, type, strength, p_value, mediator
        expected_cols = ['gene', 'phenotype', 'type', 'strength', 'p_value', 'mediator']
        for col in expected_cols:
            assert col in result.columns
    
    def test_find_mediator(self, sample_gene_data, sample_phenotype_data):
        """Test mediator finding."""
        framework = CausalDiscoveryFramework(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data,
        )
        
        mediator = framework._find_mediator('Gene_0', 'Pheno_0')
        
        # Mediator should be a gene name or None
        assert mediator is None or isinstance(mediator, str)
        if mediator is not None:
            assert mediator in sample_gene_data.columns
            assert mediator != 'Gene_0'
    
    def test_conditional_mutual_information(
        self, sample_gene_data, sample_phenotype_data
    ):
        """Test conditional mutual information calculation."""
        framework = CausalDiscoveryFramework(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data,
        )
        
        X = sample_gene_data['Gene_0'].values
        Y = sample_phenotype_data['Pheno_0'].values
        Z = sample_gene_data.drop(columns=['Gene_0'])
        
        cmi = framework._conditional_mutual_information(X, Y, Z)
        
        assert isinstance(cmi, float)
        assert cmi >= 0.0
        assert not np.isnan(cmi)
    
    def test_edge_cases_empty_data(self):
        """Test with empty data."""
        empty_gene = pd.DataFrame()
        empty_pheno = pd.DataFrame()
        
        framework = CausalDiscoveryFramework(
            gene_data=empty_gene,
            phenotype_data=empty_pheno,
        )
        
        result = framework.discover_causal_network()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_edge_cases_single_gene_pheno(self):
        """Test with single gene and phenotype."""
        gene_data = pd.DataFrame({'Gene_0': [1, 0, 1, 0, 1]})
        pheno_data = pd.DataFrame({'Pheno_0': [1, 1, 0, 0, 1]})
        
        framework = CausalDiscoveryFramework(
            gene_data=gene_data,
            phenotype_data=pheno_data,
        )
        
        # Should not crash
        result = framework.discover_causal_network(n_permutations=10)
        assert isinstance(result, pd.DataFrame)
