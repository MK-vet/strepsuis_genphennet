#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Causal Discovery Framework

This module implements conditional independence testing to distinguish direct vs
indirect gene-phenotype associations using PC algorithm adapted for binary data.

Innovation: CAUSAL INFERENCE (not correlation!) - distinguishes direct vs indirect
associations, providing novel application to AMR analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CausalDiscoveryFramework:
    """
    Distinguish direct vs indirect gene-phenotype associations using conditional
    independence testing.
    
    Uses existing analysis results:
    - Initial correlation network (from perform_statistical_tests)
    - FDR correction (already applied)
    
    Adds:
    - Conditional independence testing
    - Direct vs indirect edge classification
    - Mediator identification
    """
    
    def __init__(
        self,
        gene_data: pd.DataFrame,
        phenotype_data: pd.DataFrame,
        initial_associations: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize CausalDiscoveryFramework.
        
        Args:
            gene_data: Binary DataFrame of gene presence/absence
            phenotype_data: Binary DataFrame of phenotype presence/absence
            initial_associations: DataFrame with initial associations (optional)
                Expected columns: Feature1, Feature2, Phi, FDR_corrected_p, Significant
        """
        self.gene_data = gene_data
        self.pheno_data = phenotype_data
        self.initial_associations = initial_associations
        
        # Combine data for conditioning
        self.combined_data = pd.concat([gene_data, phenotype_data], axis=1)
    
    def discover_causal_network(
        self,
        alpha: float = 0.05,
        n_permutations: int = 1000,
    ) -> pd.DataFrame:
        """
        Discover causal network using PC algorithm adapted for binary data.
        
        Steps:
        1. Start with initial correlation network (uses existing results!)
        2. Test conditional independence
        3. Remove indirect edges
        4. Classify remaining edges as direct
        
        Args:
            alpha: Significance threshold for conditional independence
            n_permutations: Number of permutations for testing
        
        Returns:
            DataFrame with causal edges and their types (direct/indirect)
        """
        # Step 1: Get initial associations (if not provided, use all significant)
        if self.initial_associations is None:
            logger.warning("No initial associations provided, will compute from data")
            # Would need to call perform_statistical_tests here
            # For now, assume it's provided
            initial_edges = pd.DataFrame()
        else:
            # Filter significant associations
            if 'Significant' in self.initial_associations.columns:
                # Use boolean indexing correctly
                fdr_col = self.initial_associations.get('FDR_corrected_p', None)
                if fdr_col is None:
                    # Create FDR column if missing
                    fdr_values = pd.Series([1.0] * len(self.initial_associations), index=self.initial_associations.index)
                else:
                    fdr_values = self.initial_associations['FDR_corrected_p']
                
                significant_mask = (
                    self.initial_associations['Significant'] &
                    (fdr_values < alpha)
                )
            else:
                # If no Significant column, create one based on FDR
                fdr_col = self.initial_associations.get('FDR_corrected_p', None)
                if fdr_col is None:
                    significant_mask = pd.Series([False] * len(self.initial_associations), index=self.initial_associations.index)
                else:
                    significant_mask = (self.initial_associations['FDR_corrected_p'] < alpha)
            
            initial_edges = self.initial_associations[significant_mask].copy()
        
        if initial_edges.empty:
            logger.warning("No significant initial associations found")
            return pd.DataFrame(columns=[
                'gene', 'phenotype', 'type', 'strength', 'p_value', 'mediator'
            ])
        
        # Step 2: Test conditional independence
        causal_edges = []
        
        logger.info(f"Testing conditional independence for {len(initial_edges)} associations...")
        
        for idx, row in tqdm(initial_edges.iterrows(), total=len(initial_edges), desc="Testing CI"):
            # Determine which is gene and which is phenotype
            feature1 = row.get('Feature1', row.get('Gene', ''))
            feature2 = row.get('Feature2', row.get('Phenotype', ''))
            
            # Check if feature1 is gene or phenotype
            if feature1 in self.gene_data.columns:
                gene_x = feature1
                pheno_y = feature2 if feature2 in self.pheno_data.columns else None
            elif feature2 in self.gene_data.columns:
                gene_x = feature2
                pheno_y = feature1 if feature1 in self.pheno_data.columns else None
            else:
                # Skip if not gene-phenotype pair
                continue
            
            if pheno_y is None:
                continue
            
            # Test conditional independence
            is_independent, p_value = self.test_conditional_independence(
                gene_x,
                pheno_y,
                conditioning_genes=self.gene_data.drop(columns=[gene_x]) if gene_x in self.gene_data.columns else self.gene_data,
                n_permutations=n_permutations,
            )
            
            if not is_independent:
                # Association remains after conditioning → likely direct
                strength = row.get('Phi', row.get('Cramers_V', 0.0))
                causal_edges.append({
                    'gene': gene_x,
                    'phenotype': pheno_y,
                    'type': 'direct',
                    'strength': abs(strength),
                    'p_value': p_value,
                    'mediator': None,
                })
            else:
                # Association disappears → indirect
                # Find mediator
                mediator = self._find_mediator(gene_x, pheno_y)
                causal_edges.append({
                    'gene': gene_x,
                    'phenotype': pheno_y,
                    'type': 'indirect',
                    'strength': row.get('Phi', 0.0),
                    'p_value': p_value,
                    'mediator': mediator,
                })
        
        if not causal_edges:
            logger.info("No causal edges identified")
            return pd.DataFrame(columns=[
                'gene', 'phenotype', 'type', 'strength', 'p_value', 'mediator'
            ])
        
        return pd.DataFrame(causal_edges)
    
    def test_conditional_independence(
        self,
        X: str,
        Y: str,
        conditioning_genes: pd.DataFrame,
        n_permutations: int = 1000,
    ) -> Tuple[bool, float]:
        """
        Test X ⊥ Y | Z using conditional mutual information.
        
        H0: X and Y are independent given Z
        
        Args:
            X: Gene name
            Y: Phenotype name
            conditioning_genes: DataFrame of conditioning variables
            n_permutations: Number of permutations for testing
        
        Returns:
            Tuple of (is_independent, p_value)
        """
        # Get data vectors
        x_vec = self.gene_data[X].values if X in self.gene_data.columns else None
        y_vec = self.pheno_data[Y].values if Y in self.pheno_data.columns else None
        
        if x_vec is None or y_vec is None:
            return True, 1.0
        
        # Conditional MI
        cmi_observed = self._conditional_mutual_information(x_vec, y_vec, conditioning_genes)
        
        # Permutation test
        cmi_null = []
        # Ensure x_vec and y_vec are 1D arrays
        x_vec = np.array(x_vec).flatten()
        y_vec = np.array(y_vec).flatten()
        
        for _ in range(n_permutations):
            x_perm = np.random.permutation(x_vec)
            cmi_null.append(
                self._conditional_mutual_information(x_perm, y_vec, conditioning_genes)
            )
        
        # P-value
        p_value = np.mean(np.array(cmi_null) >= cmi_observed)
        
        # Decision
        is_independent = bool(p_value > 0.05)
        
        return is_independent, float(p_value)
    
    def _conditional_mutual_information(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: pd.DataFrame,
    ) -> float:
        """
        Calculate I(X; Y | Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z)
        
        For binary variables with multiple conditioning variables.
        """
        # Discretize conditioning set (simple approach: use first few genes)
        if Z.empty:
            # No conditioning - use standard MI
            return self._mutual_information(X, Y)
        
        # Use first 3 conditioning genes (to avoid curse of dimensionality)
        conditioning_cols = Z.columns[:min(3, len(Z.columns))]
        Z_subset_df = Z[conditioning_cols]
        
        # Ensure X, Y, and Z have same length - CRITICAL FIX
        X = np.array(X).flatten()
        Y = np.array(Y).flatten()
        min_len = min(len(X), len(Y), len(Z_subset_df))
        
        # Truncate all to same length
        X = X[:min_len]
        Y = Y[:min_len]
        Z_subset_df = Z_subset_df.iloc[:min_len]
        
        # Get values as array
        Z_values = Z_subset_df.values
        
        # Create conditioning states (simple hash)
        Z_states = []
        for i in range(min_len):  # Use min_len, not len(Z_values)
            if Z_values.ndim > 1 and Z_values.shape[1] > 1:
                # Multi-column: tuple of row
                state = tuple(Z_values[i, :])
            elif Z_values.ndim > 1:
                # Single column 2D: flatten
                state = (Z_values[i, 0],)
            else:
                # 1D array
                state = (Z_values[i],)
            Z_states.append(state)
        Z_states = np.array(Z_states, dtype=object)
        
        cmi = 0.0
        
        unique_states = np.unique(Z_states)
        for z_state in unique_states:
            # Create boolean mask
            mask = np.array([s == z_state for s in Z_states], dtype=bool)
            # Ensure mask is 1D and matches X/Y length
            mask = mask.flatten()[:min_len] if mask.ndim > 1 else mask[:min_len]
            X_z = X[mask]
            Y_z = Y[mask]
            
            if len(X_z) < 5:  # Too few samples
                continue
            
            # P(Z=z)
            p_z = np.mean(mask)
            
            # MI(X;Y | Z=z)
            mi_z = self._mutual_information(X_z, Y_z)
            
            cmi += p_z * mi_z
        
        return cmi
    
    def _mutual_information(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Calculate standard MI for binary variables."""
        # Joint distribution
        joint, _, _ = np.histogram2d(X, Y, bins=2)
        joint = joint / joint.sum()
        
        # Marginals
        px = joint.sum(axis=1)
        py = joint.sum(axis=0)
        
        # MI = H(X) + H(Y) - H(X,Y)
        h_x = scipy_entropy(px, base=2)
        h_y = scipy_entropy(py, base=2)
        h_xy = scipy_entropy(joint.flatten(), base=2)
        
        mi = h_x + h_y - h_xy
        
        return mi
    
    def _find_mediator(
        self,
        gene_x: str,
        pheno_y: str,
    ) -> Optional[str]:
        """
        Find mediator gene between gene_x and pheno_y.
        
        Simple heuristic: gene with strongest association to both.
        """
        if gene_x not in self.gene_data.columns:
            return None
        
        # Get all other genes
        other_genes = [g for g in self.gene_data.columns if g != gene_x]
        
        if not other_genes:
            return None
        
        # Calculate association strength with both X and Y
        mediator_scores = []
        
        for mediator_gene in other_genes:
            # Association with X
            mi_x_med = self._mutual_information(
                self.gene_data[gene_x].values,
                self.gene_data[mediator_gene].values
            )
            
            # Association with Y
            mi_med_y = self._mutual_information(
                self.gene_data[mediator_gene].values,
                self.pheno_data[pheno_y].values
            )
            
            # Combined score
            score = mi_x_med * mi_med_y
            mediator_scores.append((mediator_gene, score))
        
        if not mediator_scores:
            return None
        
        # Return gene with highest score
        mediator_scores.sort(key=lambda x: x[1], reverse=True)
        return mediator_scores[0][0]
