#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predictive Modeling Module

This module implements genotype-to-phenotype prediction using multiple machine
learning algorithms.

Innovation: PREDICTIVE (not just descriptive) - identifies most predictive genes
for each resistance phenotype, providing practical value for diagnostic markers.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenotypePhenotypePredictor:
    """
    Predict resistance phenotype from genotype using multiple ML algorithms.
    
    For each phenotype, builds classifier: genes → resistance
    
    Algorithms:
    - Logistic Regression
    - Random Forest
    - XGBoost (if available)
    """
    
    def __init__(
        self,
        gene_data: pd.DataFrame,
        phenotype_data: pd.DataFrame,
        test_size: float = 0.3,
        random_state: int = 42,
    ):
        """
        Initialize GenotypePhenotypePredictor.
        
        Args:
            gene_data: Binary DataFrame of gene presence/absence
            phenotype_data: Binary DataFrame of phenotype presence/absence
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.X = gene_data
        self.y = phenotype_data
        self.test_size = test_size
        self.random_state = random_state
    
    def build_prediction_models(
        self,
        min_samples: int = 10,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Build prediction models for each phenotype.
        
        Args:
            min_samples: Minimum number of samples per class (default: 10)
        
        Returns:
            Nested dictionary: {phenotype: {model_name: {metric: value}}}
        """
        results = {}
        
        logger.info(f"Building prediction models for {len(self.y.columns)} phenotypes...")
        
        for phenotype in tqdm(self.y.columns, desc="Phenotypes"):
            y_pheno = self.y[phenotype]
            
            # Skip if not binary
            unique_vals = y_pheno.unique()
            if len(unique_vals) != 2:
                logger.warning(f"Skipping {phenotype}: not binary (values: {unique_vals})")
                continue
            
            # Check minimum samples per class
            class_counts = y_pheno.value_counts()
            if any(count < min_samples for count in class_counts.values):
                logger.warning(f"Skipping {phenotype}: insufficient samples per class")
                continue
            
            # Train/test split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    self.X,
                    y_pheno,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    stratify=y_pheno,
                )
            except ValueError as e:
                logger.warning(f"Skipping {phenotype}: {e}")
                continue
            
            # Train multiple models
            models = {
                'Logistic Regression': LogisticRegression(
                    max_iter=1000,
                    random_state=self.random_state,
                ),
                'Random Forest': RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                ),
            }
            
            if XGBOOST_AVAILABLE:
                models['XGBoost'] = XGBClassifier(
                    random_state=self.random_state,
                    eval_metric='logloss',
                )
            
            phenotype_results = {}
            
            for model_name, model in models.items():
                try:
                    # Train
                    model.fit(X_train, y_train)
                    
                    # Predict
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]
                    
                    # Evaluate
                    phenotype_results[model_name] = {
                        'accuracy': round(accuracy_score(y_test, y_pred), 4),
                        'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
                        'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
                        'f1': round(f1_score(y_test, y_pred, zero_division=0), 4),
                        'roc_auc': round(roc_auc_score(y_test, y_prob), 4),
                        'top_predictive_genes': self._get_top_features(
                            model, self.X.columns, n=10
                        ),
                    }
                except Exception as e:
                    logger.warning(f"Model {model_name} failed for {phenotype}: {e}")
                    phenotype_results[model_name] = None
            
            results[phenotype] = phenotype_results
        
        return results
    
    def _get_top_features(
        self,
        model,
        feature_names: pd.Index,
        n: int = 10,
    ) -> List[Tuple[str, float]]:
        """Extract top N most important features from model."""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                return []
            
            top_indices = np.argsort(importances)[-n:][::-1]
            return [(feature_names[i], float(importances[i])) for i in top_indices]
        except Exception as e:
            logger.warning(f"Error extracting features: {e}")
            return []
    
    def generate_prediction_report(
        self,
        results: Dict[str, Dict[str, Dict[str, float]]],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate formatted prediction report.
        
        Args:
            results: Results from build_prediction_models()
            output_path: Optional path to save report
        
        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 80,
            "GENOTYPE-TO-PHENOTYPE PREDICTION REPORT",
            "=" * 80,
            "",
            f"Phenotypes analyzed: {len(results)}",
            "",
            "-" * 80,
        ]
        
        for phenotype, models in results.items():
            report_lines.extend([
                f"\nPhenotype: {phenotype}",
                "-" * 40,
            ])
            
            # Find best model
            best_model = None
            best_auc = 0.0
            
            for model_name, metrics in models.items():
                if metrics is None:
                    continue
                
                auc = metrics.get('roc_auc', 0.0)
                if auc > best_auc:
                    best_auc = auc
                    best_model = (model_name, metrics)
            
            if best_model:
                model_name, metrics = best_model
                report_lines.extend([
                    f"Best Model: {model_name}",
                    f"  ROC-AUC: {metrics['roc_auc']:.4f}",
                    f"  Accuracy: {metrics['accuracy']:.4f}",
                    f"  F1-Score: {metrics['f1']:.4f}",
                    "",
                    "Top Predictive Genes:",
                ])
                
                for gene, importance in metrics.get('top_predictive_genes', [])[:10]:
                    report_lines.append(f"  {gene}: {importance:.4f}")
            
            report_lines.append("")
        
        report_lines.extend([
            "=" * 80,
            f"Report generated: {pd.Timestamp.now()}",
            "=" * 80,
        ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Prediction report saved to: {output_path}")
        
        return report
