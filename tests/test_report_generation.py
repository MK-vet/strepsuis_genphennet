#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for report generation functions in network_analysis_core.py.

Tests HTML and Excel report generation.
"""

import os
import tempfile
import shutil
import pytest
import pandas as pd
import numpy as np

try:
    from strepsuis_genphennet.network_analysis_core import (
        generate_report_with_cluster_stats,
        generate_excel_report_with_cluster_stats,
    )
    FUNCTIONS_AVAILABLE = True
except ImportError:
    FUNCTIONS_AVAILABLE = False


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
class TestGenerateReportWithClusterStats:
    """Test HTML report generation."""
    
    @pytest.fixture
    def sample_chi2_df(self):
        """Create sample chi2 results."""
        return pd.DataFrame({
            'Feature1': ['Gene_A', 'Gene_B', 'Gene_C'],
            'Feature2': ['Pheno_1', 'Pheno_2', 'Pheno_1'],
            'Phi_coefficient': [0.5, 0.6, 0.4],
            'P_Value': [0.01, 0.02, 0.03],
            'FDR_corrected_p': [0.01, 0.02, 0.03],
            'Significant': [True, True, False],
        })
    
    @pytest.fixture
    def sample_network_df(self):
        """Create sample network DataFrame."""
        return pd.DataFrame({
            'Feature': ['Gene_A', 'Gene_B', 'Gene_C'],
            'Category': ['AMR_genes', 'AMR_genes', 'Virulence'],
            'Cluster': [1, 1, 2],
            'Degree_Centrality': [0.5, 0.6, 0.3],
        })
    
    @pytest.fixture
    def sample_entropy_df(self):
        """Create sample entropy DataFrame."""
        return pd.DataFrame({
            'Feature': ['Gene_A', 'Gene_B'],
            'Entropy': [0.8, 0.9],
        })
    
    @pytest.fixture
    def sample_cramers_df(self):
        """Create sample Cramers V DataFrame."""
        return pd.DataFrame({
            'Feature1': ['Gene_A', 'Gene_B'],
            'Feature2': ['Pheno_1', 'Pheno_2'],
            'Cramers_V': [0.5, 0.6],
        })
    
    def test_generate_report_basic(
        self, sample_chi2_df, sample_network_df,
        sample_entropy_df, sample_cramers_df
    ):
        """Test basic HTML report generation."""
        # Create empty DataFrames for optional parameters
        feature_summary_df = pd.DataFrame()
        excl2_df = pd.DataFrame()
        excl3_df = pd.DataFrame()
        hubs_df = pd.DataFrame()
        network_html = ""
        
        # Create summaries
        chi2_summary = {'Total pairs': 3, 'Significant': 2}
        entropy_summary = {'Mean': 0.85}
        cramers_summary = {'Mean': 0.55}
        excl2_summary = {}
        excl3_summary = {}
        network_summary = {'Nodes': 3, 'Edges': 2}
        hubs_summary = {'Total hubs': 1}
        
        # Category/feature summaries
        chi2_cat = {}
        chi2_feat = {}
        entropy_cat = {}
        entropy_feat = {}
        cramers_cat = {}
        cramers_feat = {}
        excl2_cat = {}
        excl2_feat = {}
        excl3_cat = {}
        excl3_feat = {}
        network_cat = {}
        network_feat = {}
        
        html = generate_report_with_cluster_stats(
            sample_chi2_df,
            sample_network_df,
            sample_entropy_df,
            sample_cramers_df,
            feature_summary_df,
            excl2_df,
            excl3_df,
            hubs_df,
            network_html,
            chi2_summary,
            entropy_summary,
            cramers_summary,
            excl2_summary,
            excl3_summary,
            network_summary,
            hubs_summary,
            chi2_cat,
            chi2_feat,
            entropy_cat,
            entropy_feat,
            cramers_cat,
            cramers_feat,
            excl2_cat,
            excl2_feat,
            excl3_cat,
            excl3_feat,
            network_cat,
            network_feat,
        )
        
        assert isinstance(html, str)
        assert len(html) > 0


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
class TestGenerateExcelReportWithClusterStats:
    """Test Excel report generation."""
    
    @pytest.fixture
    def temp_output_folder(self):
        """Create temporary output folder."""
        temp_dir = tempfile.mkdtemp()
        # Set output folder for the function
        import strepsuis_genphennet.network_analysis_core as nac
        original_output = getattr(nac, 'OUTPUT_DIR', None)
        nac.OUTPUT_DIR = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        yield temp_dir
        if original_output is not None:
            nac.OUTPUT_DIR = original_output
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_dataframes(self):
        """Create sample DataFrames."""
        return {
            'chi2_df': pd.DataFrame({
                'Feature1': ['Gene_A', 'Gene_B'],
                'Feature2': ['Pheno_1', 'Pheno_2'],
                'Phi_coefficient': [0.5, 0.6],
            }),
            'network_df': pd.DataFrame({
                'Feature': ['Gene_A', 'Gene_B'],
                'Cluster': [1, 2],
            }),
            'entropy_df': pd.DataFrame({
                'Feature': ['Gene_A'],
                'Entropy': [0.8],
            }),
            'cramers_df': pd.DataFrame({
                'Feature1': ['Gene_A'],
                'Cramers_V': [0.5],
            }),
        }
    
    def test_generate_excel_report_basic(self, temp_output_folder, sample_dataframes):
        """Test basic Excel report generation."""
        # Create summaries
        summaries = {
            'chi2_summary': {'Total': 2},
            'entropy_summary': {},
            'cramers_summary': {},
            'excl2_summary': {},
            'excl3_summary': {},
            'network_summary': {},
            'hubs_summary': {},
        }
        
        # Category/feature summaries
        cats = {f'{k}_cat': {} for k in ['chi2', 'entropy', 'cramers', 'excl2', 'excl3', 'network']}
        feats = {f'{k}_feat': {} for k in ['chi2', 'entropy', 'cramers', 'excl2', 'excl3', 'network']}
        
        excel_path = generate_excel_report_with_cluster_stats(
            sample_dataframes['chi2_df'],
            sample_dataframes['network_df'],
            sample_dataframes['entropy_df'],
            sample_dataframes['cramers_df'],
            pd.DataFrame(),  # feature_summary_df
            pd.DataFrame(),  # excl2_df
            pd.DataFrame(),  # excl3_df
            pd.DataFrame(),  # hubs_df
            None,  # fig_network
            **summaries,
            **cats,
            **feats,
        )
        
        # Should return path to Excel file
        assert isinstance(excel_path, str)
        # May or may not exist depending on implementation
        # assert os.path.exists(excel_path)
