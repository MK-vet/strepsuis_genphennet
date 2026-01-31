#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Tests for Causal Discovery and Predictive Modeling

Target: Increase coverage for causal_discovery.py (15%) and predictive_modeling.py (59%)

Critical Coverage Areas:
- CausalDiscoveryFramework.discover_causal_network()
- CausalDiscoveryFramework.test_conditional_independence()
- GenotypePhenotypePredictor.build_prediction_models()
- Feature importance extraction
- Report generation
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strepsuis_genphennet.causal_discovery import CausalDiscoveryFramework
from strepsuis_genphennet.predictive_modeling import GenotypePhenotypePredictor


class TestCausalDiscoveryFramework:
    """Test causal discovery functionality."""

    @pytest.fixture
    def sample_gene_data(self):
        """Create sample binary gene data."""
        np.random.seed(42)
        return pd.DataFrame(
            np.random.binomial(1, 0.5, (50, 10)),
            columns=[f'Gene_{i}' for i in range(10)],
            index=[f'Strain_{i}' for i in range(50)]
        )

    @pytest.fixture
    def sample_phenotype_data(self):
        """Create sample binary phenotype data."""
        np.random.seed(43)
        return pd.DataFrame(
            np.random.binomial(1, 0.5, (50, 5)),
            columns=[f'Pheno_{i}' for i in range(5)],
            index=[f'Strain_{i}' for i in range(50)]
        )

    @pytest.fixture
    def sample_associations(self):
        """Create sample initial associations."""
        return pd.DataFrame({
            'Feature1': ['Gene_0', 'Gene_1', 'Gene_2'],
            'Feature2': ['Pheno_0', 'Pheno_1', 'Pheno_2'],
            'Phi': [0.8, 0.7, 0.6],
            'FDR_corrected_p': [0.001, 0.01, 0.02],
            'Significant': [True, True, True]
        })

    def test_initialization(self, sample_gene_data, sample_phenotype_data, sample_associations):
        """Test CausalDiscoveryFramework initialization."""
        framework = CausalDiscoveryFramework(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data,
            initial_associations=sample_associations
        )

        assert framework.gene_data is not None
        assert framework.pheno_data is not None
        assert framework.initial_associations is not None
        assert framework.combined_data is not None
        assert len(framework.combined_data.columns) == 15  # 10 genes + 5 phenotypes

    def test_initialization_without_associations(self, sample_gene_data, sample_phenotype_data):
        """Test initialization without initial associations."""
        framework = CausalDiscoveryFramework(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data,
            initial_associations=None
        )

        assert framework.initial_associations is None

    def test_discover_causal_network(self, sample_gene_data, sample_phenotype_data, sample_associations):
        """Test causal network discovery."""
        framework = CausalDiscoveryFramework(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data,
            initial_associations=sample_associations
        )

        # Run discovery with limited permutations for speed
        causal_edges = framework.discover_causal_network(
            alpha=0.05,
            n_permutations=100
        )

        assert isinstance(causal_edges, pd.DataFrame)
        assert 'gene' in causal_edges.columns
        assert 'phenotype' in causal_edges.columns
        assert 'type' in causal_edges.columns
        assert 'strength' in causal_edges.columns
        assert 'p_value' in causal_edges.columns

        if len(causal_edges) > 0:
            assert all(causal_edges['type'].isin(['direct', 'indirect']))

    def test_discover_causal_network_no_associations(self, sample_gene_data, sample_phenotype_data):
        """Test causal discovery with no initial associations."""
        framework = CausalDiscoveryFramework(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data,
            initial_associations=None
        )

        causal_edges = framework.discover_causal_network(alpha=0.05, n_permutations=50)

        # Should return empty DataFrame with correct columns
        assert isinstance(causal_edges, pd.DataFrame)
        assert len(causal_edges) == 0
        assert 'gene' in causal_edges.columns
        assert 'phenotype' in causal_edges.columns

    def test_discover_causal_network_no_significant_associations(self, sample_gene_data, sample_phenotype_data):
        """Test causal discovery when no associations are significant."""
        # Create associations with all non-significant
        non_sig_assoc = pd.DataFrame({
            'Feature1': ['Gene_0', 'Gene_1'],
            'Feature2': ['Pheno_0', 'Pheno_1'],
            'Phi': [0.1, 0.2],
            'FDR_corrected_p': [0.9, 0.8],
            'Significant': [False, False]
        })

        framework = CausalDiscoveryFramework(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data,
            initial_associations=non_sig_assoc
        )

        causal_edges = framework.discover_causal_network(alpha=0.05)

        # Should return empty DataFrame
        assert len(causal_edges) == 0

    def test_test_conditional_independence(self, sample_gene_data, sample_phenotype_data):
        """Test conditional independence testing."""
        framework = CausalDiscoveryFramework(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data
        )

        # Test independence between Gene_0 and Pheno_0
        is_independent, p_value = framework.test_conditional_independence(
            'Gene_0',
            'Pheno_0',
            conditioning_genes=sample_gene_data.drop(columns=['Gene_0']),
            n_permutations=100
        )

        assert isinstance(is_independent, bool)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1

    def test_test_conditional_independence_missing_features(self, sample_gene_data, sample_phenotype_data):
        """Test conditional independence with missing features."""
        framework = CausalDiscoveryFramework(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data
        )

        # Test with non-existent features
        is_independent, p_value = framework.test_conditional_independence(
            'NonExistentGene',
            'Pheno_0',
            conditioning_genes=sample_gene_data,
            n_permutations=50
        )

        # Should return independent with p=1
        assert is_independent is True
        assert p_value == 1.0

    def test_conditional_mutual_information(self, sample_gene_data, sample_phenotype_data):
        """Test conditional mutual information calculation."""
        framework = CausalDiscoveryFramework(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data
        )

        X = sample_gene_data['Gene_0'].values
        Y = sample_phenotype_data['Pheno_0'].values
        Z = sample_gene_data.drop(columns=['Gene_0'])

        cmi = framework._conditional_mutual_information(X, Y, Z)

        assert isinstance(cmi, float)
        assert cmi >= 0  # MI is non-negative

    def test_conditional_mutual_information_empty_conditioning(self, sample_gene_data, sample_phenotype_data):
        """Test CMI with empty conditioning set."""
        framework = CausalDiscoveryFramework(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data
        )

        X = sample_gene_data['Gene_0'].values
        Y = sample_phenotype_data['Pheno_0'].values
        Z = pd.DataFrame()  # Empty conditioning set

        cmi = framework._conditional_mutual_information(X, Y, Z)

        # Should fall back to standard MI
        assert isinstance(cmi, float)
        assert cmi >= 0

    def test_mutual_information(self, sample_gene_data, sample_phenotype_data):
        """Test mutual information calculation."""
        framework = CausalDiscoveryFramework(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data
        )

        X = sample_gene_data['Gene_0'].values
        Y = sample_phenotype_data['Pheno_0'].values

        mi = framework._mutual_information(X, Y)

        assert isinstance(mi, float)
        assert mi >= 0

    def test_mutual_information_identical_variables(self, sample_gene_data):
        """Test MI for identical variables (maximum MI)."""
        framework = CausalDiscoveryFramework(
            gene_data=sample_gene_data,
            phenotype_data=sample_gene_data.iloc[:, :5]
        )

        X = sample_gene_data['Gene_0'].values

        mi = framework._mutual_information(X, X)

        # MI(X, X) = H(X)
        assert mi > 0  # Should be > 0 for non-constant variable

    def test_find_mediator(self, sample_gene_data, sample_phenotype_data):
        """Test mediator identification."""
        framework = CausalDiscoveryFramework(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data
        )

        mediator = framework._find_mediator('Gene_0', 'Pheno_0')

        if mediator is not None:
            assert isinstance(mediator, str)
            assert mediator in sample_gene_data.columns
            assert mediator != 'Gene_0'

    def test_find_mediator_no_other_genes(self, sample_phenotype_data):
        """Test mediator finding when no other genes available."""
        # Gene data with only one gene
        single_gene_data = pd.DataFrame({
            'Gene_0': np.random.binomial(1, 0.5, 50)
        }, index=[f'Strain_{i}' for i in range(50)])

        framework = CausalDiscoveryFramework(
            gene_data=single_gene_data,
            phenotype_data=sample_phenotype_data
        )

        mediator = framework._find_mediator('Gene_0', 'Pheno_0')

        assert mediator is None

    def test_find_mediator_invalid_gene(self, sample_gene_data, sample_phenotype_data):
        """Test mediator finding with invalid gene."""
        framework = CausalDiscoveryFramework(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data
        )

        mediator = framework._find_mediator('NonExistentGene', 'Pheno_0')

        assert mediator is None


class TestGenotypePhenotypePredictor:
    """Test predictive modeling functionality."""

    @pytest.fixture
    def sample_gene_data(self):
        """Create sample binary gene data."""
        np.random.seed(44)
        return pd.DataFrame(
            np.random.binomial(1, 0.5, (100, 20)),
            columns=[f'Gene_{i}' for i in range(20)],
            index=[f'Strain_{i}' for i in range(100)]
        )

    @pytest.fixture
    def sample_phenotype_data(self):
        """Create sample binary phenotype data with clear signal."""
        np.random.seed(45)
        # Create phenotypes with some association to genes
        gene_data = np.random.binomial(1, 0.5, (100, 20))
        pheno_data = np.zeros((100, 3))

        # Pheno_0 associated with Gene_0
        pheno_data[:, 0] = (gene_data[:, 0] & np.random.binomial(1, 0.8, 100))
        # Pheno_1 associated with Gene_1 and Gene_2
        pheno_data[:, 1] = ((gene_data[:, 1] | gene_data[:, 2]) & np.random.binomial(1, 0.7, 100))
        # Pheno_2 random
        pheno_data[:, 2] = np.random.binomial(1, 0.5, 100)

        return pd.DataFrame(
            pheno_data.astype(int),
            columns=[f'Pheno_{i}' for i in range(3)],
            index=[f'Strain_{i}' for i in range(100)]
        )

    def test_initialization(self, sample_gene_data, sample_phenotype_data):
        """Test GenotypePhenotypePredictor initialization."""
        predictor = GenotypePhenotypePredictor(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data,
            test_size=0.3,
            random_state=42
        )

        assert predictor.X is not None
        assert predictor.y is not None
        assert predictor.test_size == 0.3
        assert predictor.random_state == 42

    def test_build_prediction_models(self, sample_gene_data, sample_phenotype_data):
        """Test building prediction models."""
        predictor = GenotypePhenotypePredictor(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data,
            test_size=0.3,
            random_state=42
        )

        results = predictor.build_prediction_models(min_samples=10)

        assert isinstance(results, dict)

        # Check results structure
        for phenotype, models in results.items():
            assert isinstance(models, dict)
            assert 'Logistic Regression' in models
            assert 'Random Forest' in models

            # Check metrics
            for model_name, metrics in models.items():
                if metrics is not None:
                    assert 'accuracy' in metrics
                    assert 'precision' in metrics
                    assert 'recall' in metrics
                    assert 'f1' in metrics
                    assert 'roc_auc' in metrics
                    assert 'top_predictive_genes' in metrics

                    # Check metric ranges
                    assert 0 <= metrics['accuracy'] <= 1
                    assert 0 <= metrics['roc_auc'] <= 1

    def test_build_prediction_models_insufficient_samples(self, sample_gene_data):
        """Test model building with insufficient samples per class."""
        # Create phenotype with imbalanced classes
        small_pheno = pd.DataFrame({
            'Pheno_0': [1, 1, 1, 0, 0] + [0] * 95  # Only 3 positives
        }, index=[f'Strain_{i}' for i in range(100)])

        predictor = GenotypePhenotypePredictor(
            gene_data=sample_gene_data,
            phenotype_data=small_pheno,
            test_size=0.3,
            random_state=42
        )

        results = predictor.build_prediction_models(min_samples=10)

        # Should skip phenotypes with insufficient samples
        assert len(results) == 0 or 'Pheno_0' not in results

    def test_build_prediction_models_non_binary(self, sample_gene_data):
        """Test model building with non-binary phenotype."""
        # Create multi-class phenotype
        non_binary_pheno = pd.DataFrame({
            'Pheno_0': np.random.choice([0, 1, 2], 100)
        }, index=[f'Strain_{i}' for i in range(100)])

        predictor = GenotypePhenotypePredictor(
            gene_data=sample_gene_data,
            phenotype_data=non_binary_pheno,
            test_size=0.3,
            random_state=42
        )

        results = predictor.build_prediction_models(min_samples=10)

        # Should skip non-binary phenotypes
        assert len(results) == 0 or 'Pheno_0' not in results

    def test_get_top_features_logistic_regression(self, sample_gene_data, sample_phenotype_data):
        """Test feature importance extraction from logistic regression."""
        from sklearn.linear_model import LogisticRegression

        predictor = GenotypePhenotypePredictor(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data
        )

        # Train a simple model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(sample_gene_data, sample_phenotype_data['Pheno_0'])

        top_features = predictor._get_top_features(model, sample_gene_data.columns, n=5)

        assert isinstance(top_features, list)
        assert len(top_features) <= 5
        if len(top_features) > 0:
            assert all(isinstance(item, tuple) for item in top_features)
            assert all(len(item) == 2 for item in top_features)

    def test_get_top_features_random_forest(self, sample_gene_data, sample_phenotype_data):
        """Test feature importance extraction from random forest."""
        from sklearn.ensemble import RandomForestClassifier

        predictor = GenotypePhenotypePredictor(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data
        )

        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(sample_gene_data, sample_phenotype_data['Pheno_0'])

        top_features = predictor._get_top_features(model, sample_gene_data.columns, n=10)

        assert isinstance(top_features, list)
        assert len(top_features) <= 10
        if len(top_features) > 0:
            # Check sorted by importance
            importances = [imp for _, imp in top_features]
            assert importances == sorted(importances, reverse=True)

    def test_generate_prediction_report(self, sample_gene_data, sample_phenotype_data, tmp_path):
        """Test prediction report generation."""
        predictor = GenotypePhenotypePredictor(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data,
            test_size=0.3,
            random_state=42
        )

        # Build models
        results = predictor.build_prediction_models(min_samples=10)

        # Generate report
        output_path = tmp_path / "prediction_report.txt"
        report = predictor.generate_prediction_report(results, output_path=str(output_path))

        assert isinstance(report, str)
        assert 'GENOTYPE-TO-PHENOTYPE PREDICTION REPORT' in report
        assert 'Phenotypes analyzed:' in report

        # Verify file created
        assert output_path.exists()
        with open(output_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        assert file_content == report

    def test_generate_prediction_report_no_results(self, sample_gene_data, sample_phenotype_data):
        """Test report generation with no results."""
        predictor = GenotypePhenotypePredictor(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data
        )

        report = predictor.generate_prediction_report({})

        assert 'Phenotypes analyzed: 0' in report

    def test_xgboost_availability(self):
        """Test XGBoost availability detection."""
        try:
            from xgboost import XGBClassifier
            from strepsuis_genphennet.predictive_modeling import XGBOOST_AVAILABLE
            assert XGBOOST_AVAILABLE is True
        except ImportError:
            from strepsuis_genphennet.predictive_modeling import XGBOOST_AVAILABLE
            assert XGBOOST_AVAILABLE is False


class TestIntegrationCausalPredictive:
    """Integration tests combining causal discovery and predictive modeling."""

    @pytest.fixture
    def integrated_data(self):
        """Create integrated dataset with causal structure."""
        np.random.seed(100)

        # Gene -> Mediator -> Phenotype causal chain
        n = 100
        gene_x = np.random.binomial(1, 0.5, n)
        mediator = (gene_x & np.random.binomial(1, 0.8, n))
        phenotype = (mediator & np.random.binomial(1, 0.8, n))

        gene_data = pd.DataFrame({
            'Gene_X': gene_x,
            'Mediator': mediator,
            'Gene_Noise': np.random.binomial(1, 0.5, n)
        }, index=[f'S{i}' for i in range(n)])

        pheno_data = pd.DataFrame({
            'Pheno_Y': phenotype
        }, index=[f'S{i}' for i in range(n)])

        # Initial associations
        associations = pd.DataFrame({
            'Feature1': ['Gene_X', 'Mediator'],
            'Feature2': ['Pheno_Y', 'Pheno_Y'],
            'Phi': [0.6, 0.8],
            'FDR_corrected_p': [0.001, 0.0001],
            'Significant': [True, True]
        })

        return gene_data, pheno_data, associations

    def test_causal_then_predictive(self, integrated_data):
        """Test causal discovery followed by predictive modeling."""
        gene_data, pheno_data, associations = integrated_data

        # Step 1: Causal discovery
        causal_framework = CausalDiscoveryFramework(
            gene_data=gene_data,
            phenotype_data=pheno_data,
            initial_associations=associations
        )

        causal_edges = causal_framework.discover_causal_network(
            alpha=0.05,
            n_permutations=100
        )

        # Step 2: Predictive modeling
        predictor = GenotypePhenotypePredictor(
            gene_data=gene_data,
            phenotype_data=pheno_data,
            test_size=0.3,
            random_state=42
        )

        results = predictor.build_prediction_models(min_samples=10)

        # Verify both analyses completed
        assert isinstance(causal_edges, pd.DataFrame)
        assert isinstance(results, dict)

        # Mediator should be identified as important predictor
        if 'Pheno_Y' in results and results['Pheno_Y']['Random Forest'] is not None:
            top_genes = results['Pheno_Y']['Random Forest']['top_predictive_genes']
            top_gene_names = [gene for gene, _ in top_genes]
            assert 'Mediator' in top_gene_names or 'Gene_X' in top_gene_names


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
