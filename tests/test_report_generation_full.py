#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full tests for report generation functions.
"""

import os
import tempfile
import shutil
import pytest
import pandas as pd
import numpy as np

from strepsuis_genphennet.network_analysis_core import (
    generate_report_with_cluster_stats,
    create_interactive_table,
    create_section_summary,
    create_interactive_table_with_empty,
)


class TestGenerateReportWithClusterStatsFull:
    """Full tests for generate_report_with_cluster_stats."""
    
    @pytest.fixture
    def complete_data(self):
        """Create complete data for report generation."""
        chi2_df = pd.DataFrame({
            'Feature1': ['Gene_A', 'Gene_B', 'Gene_C', 'Gene_D'],
            'Feature2': ['Pheno_X', 'Pheno_Y', 'Pheno_Z', 'Pheno_W'],
            'Category1': ['AMR', 'Vir', 'MGE', 'AMR'],
            'Category2': ['MIC', 'MIC', 'MIC', 'MIC'],
            'P_Value': [0.001, 0.01, 0.05, 0.1],
            'FDR_corrected_p': [0.004, 0.02, 0.08, 0.15],
            'Phi': [0.8, 0.6, 0.4, 0.2],
            'CI_low': [0.6, 0.4, 0.2, 0.0],
            'CI_high': [0.9, 0.8, 0.6, 0.4],
        })
        
        network_df = pd.DataFrame({
            'Feature': ['Gene_A', 'Gene_B', 'Gene_C', 'Gene_D', 'Gene_E'],
            'Category': ['AMR', 'Vir', 'MGE', 'AMR', 'Vir'],
            'Degree_Centrality': [0.9, 0.7, 0.5, 0.3, 0.1],
            'Betweenness_Centrality': [0.8, 0.6, 0.4, 0.2, 0.05],
            'Cluster': [1, 1, 2, 2, 3],
        })
        
        entropy_df = pd.DataFrame({
            'Feature': ['Gene_A', 'Gene_B', 'Gene_C'],
            'Category': ['AMR', 'Vir', 'MGE'],
            'Entropy': [0.9, 0.6, 0.3],
            'Normalized_Entropy': [0.95, 0.65, 0.35],
        })
        
        cramers_df = pd.DataFrame({
            'Feature1': ['Gene_A', 'Gene_B'],
            'Feature2': ['Pheno_X', 'Pheno_Y'],
            'Category1': ['AMR', 'Vir'],
            'Category2': ['MIC', 'MIC'],
            'Cramers_V': [0.7, 0.5],
            'P_Value': [0.01, 0.05],
        })
        
        feature_summary_df = pd.DataFrame({
            'Category': ['AMR', 'Vir', 'MGE', 'MIC'],
            'Count': [10, 15, 5, 8],
            'Prevalence': [0.5, 0.6, 0.3, 0.4],
        })
        
        excl2_df = pd.DataFrame({
            'Feature1': ['Gene_A', 'Gene_B'],
            'Feature2': ['Gene_X', 'Gene_Y'],
            'Category1': ['AMR', 'Vir'],
            'Category2': ['Vir', 'MGE'],
            'P_Value': [0.01, 0.02],
        })
        
        excl3_df = pd.DataFrame({
            'Feature1': ['Gene_A'],
            'Feature2': ['Gene_B'],
            'Feature3': ['Gene_C'],
            'Category1': ['AMR'],
            'Category2': ['Vir'],
            'Category3': ['MGE'],
            'P_Value': [0.01],
        })
        
        hubs_df = pd.DataFrame({
            'Feature': ['Gene_A', 'Gene_B', 'Gene_C'],
            'Category': ['AMR', 'Vir', 'MGE'],
            'Degree_Centrality': [0.95, 0.85, 0.75],
            'Cluster': [1, 1, 2],
        })
        
        return {
            'chi2_df': chi2_df,
            'network_df': network_df,
            'entropy_df': entropy_df,
            'cramers_df': cramers_df,
            'feature_summary_df': feature_summary_df,
            'excl2_df': excl2_df,
            'excl3_df': excl3_df,
            'hubs_df': hubs_df,
        }
    
    @pytest.fixture
    def complete_summaries(self):
        """Create complete summaries."""
        return {
            'chi2_summary': {
                'Total tests': 100,
                'Significant (FDR<0.05)': 50,
                'Mean Phi': 0.45,
            },
            'entropy_summary': {
                'Mean entropy': 0.6,
                'Max entropy': 0.9,
                'Min entropy': 0.3,
            },
            'cramers_summary': {
                'Total tests': 50,
                'Significant (FDR<0.05)': 25,
                'Mean Cramér\'s V': 0.55,
            },
            'excl2_summary': {
                'Total patterns': 20,
                'Significant (FDR<0.05)': 10,
            },
            'excl3_summary': {
                'Total patterns': 5,
                'Significant (FDR<0.05)': 2,
            },
            'network_summary': {
                'Total nodes': 50,
                'Total edges': 100,
                'Clusters': 5,
                'Modularity': 0.65,
            },
            'hubs_summary': {
                'Total hubs': 10,
                'Mean hub degree': 0.8,
            },
        }
    
    @pytest.fixture
    def complete_categories(self):
        """Create complete category/feature dictionaries."""
        return {
            'chi2_cat': {'AMR': 30, 'Vir': 20, 'MGE': 10},
            'chi2_feat': {'Gene_A': 15, 'Gene_B': 10, 'Gene_C': 5},
            'entropy_cat': {'AMR': 25, 'Vir': 15, 'MGE': 10},
            'entropy_feat': {'Gene_A': 12, 'Gene_B': 8, 'Gene_C': 5},
            'cramers_cat': {'AMR': 20, 'Vir': 15},
            'cramers_feat': {'Gene_A': 10, 'Gene_B': 8},
            'excl2_cat': {'AMR-Vir': 10, 'Vir-MGE': 5},
            'excl2_feat': {'Gene_A-Gene_X': 5, 'Gene_B-Gene_Y': 3},
            'excl3_cat': {'AMR-Vir-MGE': 3},
            'excl3_feat': {'Gene_A-Gene_B-Gene_C': 2},
            'network_cat': {'AMR': 15, 'Vir': 12, 'MGE': 8},
            'network_feat': {'Gene_A': 8, 'Gene_B': 6, 'Gene_C': 4},
        }
    
    def test_generate_report_full(self, complete_data, complete_summaries, complete_categories):
        """Test full report generation."""
        network_html = '<div id="network-plot">Network visualization placeholder</div>'
        
        result = generate_report_with_cluster_stats(
            chi2_df=complete_data['chi2_df'],
            network_df=complete_data['network_df'],
            entropy_df=complete_data['entropy_df'],
            cramers_df=complete_data['cramers_df'],
            feature_summary_df=complete_data['feature_summary_df'],
            excl2_df=complete_data['excl2_df'],
            excl3_df=complete_data['excl3_df'],
            hubs_df=complete_data['hubs_df'],
            network_html=network_html,
            chi2_summary=complete_summaries['chi2_summary'],
            entropy_summary=complete_summaries['entropy_summary'],
            cramers_summary=complete_summaries['cramers_summary'],
            excl2_summary=complete_summaries['excl2_summary'],
            excl3_summary=complete_summaries['excl3_summary'],
            network_summary=complete_summaries['network_summary'],
            hubs_summary=complete_summaries['hubs_summary'],
            chi2_cat=complete_categories['chi2_cat'],
            chi2_feat=complete_categories['chi2_feat'],
            entropy_cat=complete_categories['entropy_cat'],
            entropy_feat=complete_categories['entropy_feat'],
            cramers_cat=complete_categories['cramers_cat'],
            cramers_feat=complete_categories['cramers_feat'],
            excl2_cat=complete_categories['excl2_cat'],
            excl2_feat=complete_categories['excl2_feat'],
            excl3_cat=complete_categories['excl3_cat'],
            excl3_feat=complete_categories['excl3_feat'],
            network_cat=complete_categories['network_cat'],
            network_feat=complete_categories['network_feat'],
        )
        
        assert isinstance(result, str)
        assert '<html>' in result.lower() or 'html' in result.lower()
        assert len(result) > 1000  # Should be substantial HTML
    
    def test_generate_report_with_empty_excl3(self, complete_data, complete_summaries, complete_categories):
        """Test report generation with empty excl3_df."""
        complete_data['excl3_df'] = pd.DataFrame()
        network_html = '<div>Network</div>'
        
        result = generate_report_with_cluster_stats(
            chi2_df=complete_data['chi2_df'],
            network_df=complete_data['network_df'],
            entropy_df=complete_data['entropy_df'],
            cramers_df=complete_data['cramers_df'],
            feature_summary_df=complete_data['feature_summary_df'],
            excl2_df=complete_data['excl2_df'],
            excl3_df=complete_data['excl3_df'],
            hubs_df=complete_data['hubs_df'],
            network_html=network_html,
            chi2_summary=complete_summaries['chi2_summary'],
            entropy_summary=complete_summaries['entropy_summary'],
            cramers_summary=complete_summaries['cramers_summary'],
            excl2_summary=complete_summaries['excl2_summary'],
            excl3_summary=complete_summaries['excl3_summary'],
            network_summary=complete_summaries['network_summary'],
            hubs_summary=complete_summaries['hubs_summary'],
            chi2_cat=complete_categories['chi2_cat'],
            chi2_feat=complete_categories['chi2_feat'],
            entropy_cat=complete_categories['entropy_cat'],
            entropy_feat=complete_categories['entropy_feat'],
            cramers_cat=complete_categories['cramers_cat'],
            cramers_feat=complete_categories['cramers_feat'],
            excl2_cat=complete_categories['excl2_cat'],
            excl2_feat=complete_categories['excl2_feat'],
            excl3_cat=complete_categories['excl3_cat'],
            excl3_feat=complete_categories['excl3_feat'],
            network_cat=complete_categories['network_cat'],
            network_feat=complete_categories['network_feat'],
        )
        
        assert isinstance(result, str)


class TestCreateInteractiveTableFull:
    """Full tests for create_interactive_table."""
    
    def test_table_with_all_types(self):
        """Test table with all column types."""
        df = pd.DataFrame({
            'Integer': [1, 2, 3],
            'Float': [1.5, 2.5, 3.5],
            'String': ['a', 'b', 'c'],
            'Boolean': [True, False, True],
        })
        
        result = create_interactive_table(df, 'all_types')
        
        assert isinstance(result, str)
        assert 'Integer' in result
        assert 'Float' in result
        assert 'String' in result
    
    def test_table_with_special_characters(self):
        """Test table with special characters."""
        df = pd.DataFrame({
            'Name': ['Gene_A', 'Gene<B>', 'Gene&C'],
            'Value': [1, 2, 3],
        })
        
        result = create_interactive_table(df, 'special_chars')
        
        assert isinstance(result, str)


class TestCreateInteractiveTableWithEmptyFull:
    """Full tests for create_interactive_table_with_empty."""
    
    def test_with_valid_df(self):
        """Test with valid DataFrame."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
        })
        
        result = create_interactive_table_with_empty(df, 'valid_table')
        
        assert isinstance(result, str)
        assert 'table' in result.lower()
    
    def test_with_empty_df(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        
        result = create_interactive_table_with_empty(df, 'empty_table')
        
        assert isinstance(result, str)
        assert 'No data' in result or 'available' in result.lower()
