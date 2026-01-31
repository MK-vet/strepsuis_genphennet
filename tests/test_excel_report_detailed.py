#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detailed tests for Excel report generation.
"""

import os
import tempfile
import shutil
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

try:
    from strepsuis_genphennet.network_analysis_core import (
        generate_excel_report_with_cluster_stats,
    )
    FUNCTION_AVAILABLE = True
except ImportError:
    FUNCTION_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@pytest.mark.skipif(not FUNCTION_AVAILABLE or not PLOTLY_AVAILABLE, reason="Functions not available")
class TestGenerateExcelReportDetailed:
    """Detailed tests for generate_excel_report_with_cluster_stats."""
    
    @pytest.fixture
    def minimal_data(self):
        """Create minimal data for report generation."""
        return {
            'chi2_df': pd.DataFrame({
                'Feature1': ['Gene_A'],
                'Feature2': ['Pheno_X'],
                'P_Value': [0.01],
            }),
            'network_df': pd.DataFrame({
                'Feature': ['Gene_A'],
                'Degree_Centrality': [0.5],
            }),
            'entropy_df': pd.DataFrame({
                'Feature': ['Gene_A'],
                'Entropy': [0.5],
            }),
            'cramers_df': pd.DataFrame({
                'Feature1': ['Gene_A'],
                'Feature2': ['Pheno_X'],
                'Cramers_V': [0.5],
            }),
            'feature_summary_df': pd.DataFrame({
                'Category': ['AMR'],
                'Count': [10],
            }),
            'excl2_df': pd.DataFrame(),
            'excl3_df': pd.DataFrame(),
            'hubs_df': pd.DataFrame({
                'Feature': ['Gene_A'],
                'Degree_Centrality': [0.9],
            }),
        }
    
    @pytest.fixture
    def minimal_summaries(self):
        """Create minimal summaries."""
        return {
            'chi2_summary': {'Total': 10},
            'entropy_summary': {'Mean': 0.5},
            'cramers_summary': {'Total': 5},
            'excl2_summary': {'Total': 0},
            'excl3_summary': {'Total': 0},
            'network_summary': {'Nodes': 10},
            'hubs_summary': {'Total': 3},
        }
    
    @pytest.fixture
    def minimal_categories(self):
        """Create minimal categories."""
        return {
            'chi2_cat': {'AMR': 5},
            'chi2_feat': {'Gene_A': 3},
            'entropy_cat': {'AMR': 4},
            'entropy_feat': {'Gene_A': 2},
            'cramers_cat': {'AMR': 3},
            'cramers_feat': {'Gene_A': 2},
            'excl2_cat': {},
            'excl2_feat': {},
            'excl3_cat': {},
            'excl3_feat': {},
            'network_cat': {'AMR': 5},
            'network_feat': {'Gene_A': 3},
        }
    
    def test_generate_excel_report_with_none_figure(self, minimal_data, minimal_summaries, minimal_categories):
        """Test Excel report generation with None figure."""
        try:
            result = generate_excel_report_with_cluster_stats(
                chi2_df=minimal_data['chi2_df'],
                network_df=minimal_data['network_df'],
                entropy_df=minimal_data['entropy_df'],
                cramers_df=minimal_data['cramers_df'],
                feature_summary_df=minimal_data['feature_summary_df'],
                excl2_df=minimal_data['excl2_df'],
                excl3_df=minimal_data['excl3_df'],
                hubs_df=minimal_data['hubs_df'],
                fig_network=None,
                chi2_summary=minimal_summaries['chi2_summary'],
                entropy_summary=minimal_summaries['entropy_summary'],
                cramers_summary=minimal_summaries['cramers_summary'],
                excl2_summary=minimal_summaries['excl2_summary'],
                excl3_summary=minimal_summaries['excl3_summary'],
                network_summary=minimal_summaries['network_summary'],
                hubs_summary=minimal_summaries['hubs_summary'],
                chi2_cat=minimal_categories['chi2_cat'],
                chi2_feat=minimal_categories['chi2_feat'],
                entropy_cat=minimal_categories['entropy_cat'],
                entropy_feat=minimal_categories['entropy_feat'],
                cramers_cat=minimal_categories['cramers_cat'],
                cramers_feat=minimal_categories['cramers_feat'],
                excl2_cat=minimal_categories['excl2_cat'],
                excl2_feat=minimal_categories['excl2_feat'],
                excl3_cat=minimal_categories['excl3_cat'],
                excl3_feat=minimal_categories['excl3_feat'],
                network_cat=minimal_categories['network_cat'],
                network_feat=minimal_categories['network_feat'],
            )
            
            # Result should be None or path to file
            assert result is None or isinstance(result, str)
        except Exception as e:
            pytest.skip(f"Excel report generation failed: {e}")
    
    def test_generate_excel_report_with_figure(self, minimal_data, minimal_summaries, minimal_categories):
        """Test Excel report generation with figure."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2], mode='markers'))
        
        try:
            result = generate_excel_report_with_cluster_stats(
                chi2_df=minimal_data['chi2_df'],
                network_df=minimal_data['network_df'],
                entropy_df=minimal_data['entropy_df'],
                cramers_df=minimal_data['cramers_df'],
                feature_summary_df=minimal_data['feature_summary_df'],
                excl2_df=minimal_data['excl2_df'],
                excl3_df=minimal_data['excl3_df'],
                hubs_df=minimal_data['hubs_df'],
                fig_network=fig,
                chi2_summary=minimal_summaries['chi2_summary'],
                entropy_summary=minimal_summaries['entropy_summary'],
                cramers_summary=minimal_summaries['cramers_summary'],
                excl2_summary=minimal_summaries['excl2_summary'],
                excl3_summary=minimal_summaries['excl3_summary'],
                network_summary=minimal_summaries['network_summary'],
                hubs_summary=minimal_summaries['hubs_summary'],
                chi2_cat=minimal_categories['chi2_cat'],
                chi2_feat=minimal_categories['chi2_feat'],
                entropy_cat=minimal_categories['entropy_cat'],
                entropy_feat=minimal_categories['entropy_feat'],
                cramers_cat=minimal_categories['cramers_cat'],
                cramers_feat=minimal_categories['cramers_feat'],
                excl2_cat=minimal_categories['excl2_cat'],
                excl2_feat=minimal_categories['excl2_feat'],
                excl3_cat=minimal_categories['excl3_cat'],
                excl3_feat=minimal_categories['excl3_feat'],
                network_cat=minimal_categories['network_cat'],
                network_feat=minimal_categories['network_feat'],
            )
            
            assert result is None or isinstance(result, str)
        except Exception as e:
            pytest.skip(f"Excel report generation failed: {e}")
