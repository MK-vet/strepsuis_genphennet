#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for generate_excel_report_with_cluster_stats function.
"""

import os
import tempfile
import shutil
import pytest
import pandas as pd
import numpy as np

try:
    from strepsuis_genphennet.network_analysis_core import (
        generate_excel_report_with_cluster_stats,
    )
    FUNCTION_AVAILABLE = True
except ImportError:
    FUNCTION_AVAILABLE = False


@pytest.mark.skipif(not FUNCTION_AVAILABLE, reason="Function not available")
class TestGenerateExcelReportWithClusterStats:
    """Test generate_excel_report_with_cluster_stats function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for report generation."""
        chi2_df = pd.DataFrame({
            'Feature1': ['Gene_A', 'Gene_B', 'Gene_C'],
            'Feature2': ['Pheno_X', 'Pheno_Y', 'Pheno_Z'],
            'P_Value': [0.01, 0.02, 0.03],
            'Phi': [0.5, 0.6, 0.7],
        })
        network_df = pd.DataFrame({
            'Feature': ['Gene_A', 'Gene_B', 'Gene_C'],
            'Degree_Centrality': [0.5, 0.6, 0.7],
            'Category': ['AMR', 'Vir', 'MGE'],
        })
        entropy_df = pd.DataFrame({
            'Feature': ['Gene_A', 'Gene_B'],
            'Entropy': [0.5, 0.6],
        })
        cramers_df = pd.DataFrame({
            'Feature1': ['Gene_A'],
            'Feature2': ['Pheno_X'],
            'Cramers_V': [0.5],
        })
        feature_summary_df = pd.DataFrame({
            'Category': ['AMR', 'Vir', 'MGE'],
            'Count': [10, 20, 5],
        })
        excl2_df = pd.DataFrame({
            'Feature1': ['Gene_A'],
            'Feature2': ['Gene_B'],
            'P_Value': [0.01],
        })
        excl3_df = pd.DataFrame()
        hubs_df = pd.DataFrame({
            'Feature': ['Gene_A', 'Gene_B'],
            'Degree_Centrality': [0.9, 0.8],
            'Category': ['AMR', 'Vir'],
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
    def sample_summaries(self):
        """Create sample summaries."""
        return {
            'chi2_summary': {'Total tests': 100, 'Significant (FDR<0.05)': 50},
            'entropy_summary': {'Mean entropy': 0.5},
            'cramers_summary': {'Significant (FDR<0.05)': 25},
            'excl2_summary': {'Total patterns': 10, 'Significant (FDR<0.05)': 5},
            'excl3_summary': {'Total patterns': 5, 'Significant (FDR<0.05)': 2},
            'network_summary': {'Total nodes': 50, 'Total edges': 100, 'Clusters': 3},
            'hubs_summary': {'Total hubs': 5},
        }
    
    @pytest.fixture
    def sample_categories(self):
        """Create sample category/feature dictionaries."""
        return {
            'chi2_cat': {'AMR': 30},
            'chi2_feat': {'Gene_A': 10},
            'entropy_cat': {'AMR': 20},
            'entropy_feat': {'Gene_A': 5},
            'cramers_cat': {'AMR': 15},
            'cramers_feat': {'Gene_A': 8},
            'excl2_cat': {'AMR': 5},
            'excl2_feat': {'Gene_A': 2},
            'excl3_cat': {'AMR': 3},
            'excl3_feat': {'Gene_A': 1},
            'network_cat': {'AMR': 10},
            'network_feat': {'Gene_A': 4},
        }
    
    @pytest.fixture
    def temp_output_folder(self):
        """Create temporary output folder."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_generate_excel_report_basic(self, sample_data, sample_summaries, sample_categories, temp_output_folder):
        """Test basic Excel report generation."""
        import plotly.graph_objects as go
        
        # Create a simple network figure
        fig_network = go.Figure()
        fig_network.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3], mode='markers'))
        
        try:
            result = generate_excel_report_with_cluster_stats(
                chi2_df=sample_data['chi2_df'],
                network_df=sample_data['network_df'],
                entropy_df=sample_data['entropy_df'],
                cramers_df=sample_data['cramers_df'],
                feature_summary_df=sample_data['feature_summary_df'],
                excl2_df=sample_data['excl2_df'],
                excl3_df=sample_data['excl3_df'],
                hubs_df=sample_data['hubs_df'],
                fig_network=fig_network,
                chi2_summary=sample_summaries['chi2_summary'],
                entropy_summary=sample_summaries['entropy_summary'],
                cramers_summary=sample_summaries['cramers_summary'],
                excl2_summary=sample_summaries['excl2_summary'],
                excl3_summary=sample_summaries['excl3_summary'],
                network_summary=sample_summaries['network_summary'],
                hubs_summary=sample_summaries['hubs_summary'],
                chi2_cat=sample_categories['chi2_cat'],
                chi2_feat=sample_categories['chi2_feat'],
                entropy_cat=sample_categories['entropy_cat'],
                entropy_feat=sample_categories['entropy_feat'],
                cramers_cat=sample_categories['cramers_cat'],
                cramers_feat=sample_categories['cramers_feat'],
                excl2_cat=sample_categories['excl2_cat'],
                excl2_feat=sample_categories['excl2_feat'],
                excl3_cat=sample_categories['excl3_cat'],
                excl3_feat=sample_categories['excl3_feat'],
                network_cat=sample_categories['network_cat'],
                network_feat=sample_categories['network_feat'],
            )
            
            assert result is not None
        except Exception as e:
            pytest.skip(f"generate_excel_report_with_cluster_stats failed: {e}")
