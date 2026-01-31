#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for predictive modeling module.

Tests genotype-to-phenotype prediction using multiple ML algorithms.
"""

import numpy as np
import pandas as pd
import pytest

from strepsuis_genphennet.predictive_modeling import GenotypePhenotypePredictor


class TestGenotypePhenotypePredictor:
    """Test GenotypePhenotypePredictor class."""
    
    @pytest.fixture
    def sample_gene_data(self):
        """Create sample gene data."""
        np.random.seed(42)
        n_samples = 100
        n_genes = 20
        
        data = {}
        for i in range(n_genes):
            data[f"Gene_{i}"] = np.random.binomial(1, 0.3, n_samples)
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_phenotype_data(self):
        """Create sample phenotype data with some correlation to genes."""
        np.random.seed(42)
        n_samples = 100
        
        # Create phenotypes that correlate with genes
        gene_data = pd.DataFrame({
            f"Gene_{i}": np.random.binomial(1, 0.3, n_samples) 
            for i in range(20)
        })
        
        # Phenotype 0: correlated with Gene_0
        pheno_0 = (gene_data['Gene_0'].values + np.random.binomial(1, 0.2, n_samples) > 0).astype(int)
        
        # Phenotype 1: correlated with Gene_1 and Gene_2
        pheno_1 = ((gene_data['Gene_1'].values + gene_data['Gene_2'].values + 
                   np.random.binomial(1, 0.1, n_samples)) > 0).astype(int)
        
        # Phenotype 2: random (no correlation)
        pheno_2 = np.random.binomial(1, 0.4, n_samples)
        
        return pd.DataFrame({
            'Pheno_0': pheno_0,
            'Pheno_1': pheno_1,
            'Pheno_2': pheno_2,
        })
    
    def test_initialization(self, sample_gene_data, sample_phenotype_data):
        """Test predictor initialization."""
        predictor = GenotypePhenotypePredictor(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data,
        )
        
        assert predictor.X.shape[0] == sample_gene_data.shape[0]
        assert predictor.y.shape[0] == sample_phenotype_data.shape[0]
        assert predictor.test_size == 0.3
        assert predictor.random_state == 42
    
    def test_build_prediction_models(
        self, sample_gene_data, sample_phenotype_data
    ):
        """Test building prediction models."""
        predictor = GenotypePhenotypePredictor(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data,
            test_size=0.3,
            random_state=42,
        )
        
        results = predictor.build_prediction_models(min_samples=5)
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check structure of results
        for phenotype, models in results.items():
            assert isinstance(models, dict)
            for model_name, metrics in models.items():
                if metrics is not None:
                    assert 'accuracy' in metrics
                    assert 'roc_auc' in metrics
                    assert 'f1' in metrics
                    assert isinstance(metrics['accuracy'], (float, np.floating))
                    assert isinstance(metrics['roc_auc'], (float, np.floating))
    
    def test_get_top_features(self, sample_gene_data, sample_phenotype_data):
        """Test feature importance extraction."""
        predictor = GenotypePhenotypePredictor(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data,
        )
        
        # Train a simple model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            sample_gene_data,
            sample_phenotype_data['Pheno_0'],
            test_size=0.3,
            random_state=42,
            stratify=sample_phenotype_data['Pheno_0'],
        )
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        top_features = predictor._get_top_features(model, sample_gene_data.columns, n=5)
        
        assert isinstance(top_features, list)
        assert len(top_features) <= 5
        for gene, importance in top_features:
            assert isinstance(gene, str)
            assert isinstance(importance, float)
            assert importance >= 0.0
    
    def test_generate_prediction_report(
        self, sample_gene_data, sample_phenotype_data, tmp_path
    ):
        """Test prediction report generation."""
        predictor = GenotypePhenotypePredictor(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data,
        )
        
        results = predictor.build_prediction_models(min_samples=5)
        
        if results:
            output_path = tmp_path / "prediction_report.txt"
            report = predictor.generate_prediction_report(
                results, output_path=str(output_path)
            )
            
            assert isinstance(report, str)
            assert len(report) > 0
            assert "PREDICTION REPORT" in report
            assert output_path.exists()
    
    def test_edge_cases_insufficient_samples(self):
        """Test with insufficient samples per class."""
        # Create data with very few samples
        gene_data = pd.DataFrame({
            'Gene_0': [1, 0, 1, 0, 1],
            'Gene_1': [0, 1, 0, 1, 0],
        })
        
        # Phenotype with only 1 sample in one class
        pheno_data = pd.DataFrame({
            'Pheno_0': [1, 1, 1, 1, 0],  # Only 1 sample in class 0
        })
        
        predictor = GenotypePhenotypePredictor(
            gene_data=gene_data,
            phenotype_data=pheno_data,
        )
        
        results = predictor.build_prediction_models(min_samples=2)
        
        # Should skip phenotypes with insufficient samples
        assert isinstance(results, dict)
    
    def test_edge_cases_non_binary_phenotype(self):
        """Test with non-binary phenotype."""
        gene_data = pd.DataFrame({
            'Gene_0': [1, 0, 1, 0, 1] * 20,
            'Gene_1': [0, 1, 0, 1, 0] * 20,
        })
        
        # Phenotype with 3 values (not binary)
        pheno_data = pd.DataFrame({
            'Pheno_0': [0, 1, 2, 0, 1] * 20,
        })
        
        predictor = GenotypePhenotypePredictor(
            gene_data=gene_data,
            phenotype_data=pheno_data,
        )
        
        results = predictor.build_prediction_models()
        
        # Should skip non-binary phenotypes
        assert isinstance(results, dict)
        # Pheno_0 should not be in results
        assert 'Pheno_0' not in results or results.get('Pheno_0') is None
    
    def test_model_metrics_range(self, sample_gene_data, sample_phenotype_data):
        """Test that model metrics are in valid ranges."""
        predictor = GenotypePhenotypePredictor(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data,
        )
        
        results = predictor.build_prediction_models(min_samples=5)
        
        for phenotype, models in results.items():
            for model_name, metrics in models.items():
                if metrics is not None:
                    # Accuracy should be between 0 and 1
                    assert 0.0 <= metrics['accuracy'] <= 1.0
                    
                    # ROC-AUC should be between 0 and 1
                    assert 0.0 <= metrics['roc_auc'] <= 1.0
                    
                    # F1 should be between 0 and 1
                    assert 0.0 <= metrics['f1'] <= 1.0
    
    def test_reproducibility(self, sample_gene_data, sample_phenotype_data):
        """Test that results are reproducible with same random seed."""
        predictor1 = GenotypePhenotypePredictor(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data,
            random_state=42,
        )
        
        predictor2 = GenotypePhenotypePredictor(
            gene_data=sample_gene_data,
            phenotype_data=sample_phenotype_data,
            random_state=42,
        )
        
        results1 = predictor1.build_prediction_models(min_samples=5)
        results2 = predictor2.build_prediction_models(min_samples=5)
        
        # Results should be identical with same seed
        assert results1.keys() == results2.keys()
        
        for phenotype in results1.keys():
            if phenotype in results2:
                models1 = results1[phenotype]
                models2 = results2[phenotype]
                assert models1.keys() == models2.keys()
                
                for model_name in models1.keys():
                    if models1[model_name] is not None and models2[model_name] is not None:
                        m1 = models1[model_name]
                        m2 = models2[model_name]
                        # Metrics should be very close (allowing for small floating point differences)
                        assert abs(m1['accuracy'] - m2['accuracy']) < 1e-6
                        assert abs(m1['roc_auc'] - m2['roc_auc']) < 1e-6
