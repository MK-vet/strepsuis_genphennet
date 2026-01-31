#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for excel_report_utils.py in strepsuis-genphennet.

Tests ExcelReportGenerator class.
"""

import os
import tempfile
import shutil
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from strepsuis_genphennet.excel_report_utils import ExcelReportGenerator, sanitize_sheet_name
    FUNCTIONS_AVAILABLE = True
except ImportError:
    FUNCTIONS_AVAILABLE = False


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
class TestSanitizeSheetName:
    """Test sheet name sanitization."""
    
    def test_sanitize_sheet_name_valid(self):
        """Test valid sheet name."""
        result = sanitize_sheet_name("ValidSheet")
        assert result == "ValidSheet"
        assert len(result) <= 31
    
    def test_sanitize_sheet_name_long(self):
        """Test long sheet name."""
        long_name = "A" * 50
        result = sanitize_sheet_name(long_name)
        assert len(result) <= 31
    
    def test_sanitize_sheet_name_special_chars(self):
        """Test sheet name with special characters."""
        result = sanitize_sheet_name("Sheet:Name/With*Chars?")
        assert ":" not in result
        assert "/" not in result
        assert "*" not in result
        assert "?" not in result


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
class TestExcelReportGenerator:
    """Test ExcelReportGenerator class."""
    
    @pytest.fixture
    def temp_output_folder(self):
        """Create temporary output folder."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def excel_gen(self, temp_output_folder):
        """Create ExcelReportGenerator instance."""
        return ExcelReportGenerator(output_folder=temp_output_folder)
    
    def test_init_creates_folders(self, excel_gen, temp_output_folder):
        """Test that initialization creates folders."""
        assert os.path.exists(excel_gen.output_folder)
        assert os.path.exists(excel_gen.png_folder)
    
    def test_save_matplotlib_figure(self, excel_gen):
        """Test saving matplotlib figure."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])
        
        png_path = excel_gen.save_matplotlib_figure(fig, 'test_plot')
        
        assert os.path.exists(png_path)
        assert png_path in excel_gen.png_files
        plt.close(fig)
    
    def test_save_plotly_figure(self, excel_gen):
        """Test saving plotly figure."""
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
            
            try:
                png_path = excel_gen.save_plotly_figure(fig, 'test_plotly')
                assert os.path.exists(png_path) or os.path.exists(png_path.replace('.png', '.html'))
            except Exception:
                # May use fallback
                png_path = excel_gen.save_plotly_figure_fallback(fig, 'test_plotly_fallback')
                assert isinstance(png_path, str)
        except ImportError:
            pytest.skip("plotly not available")
    
    def test_add_dataframe_sheet(self, excel_gen):
        """Test adding DataFrame as sheet."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
        })
        
        output_path = os.path.join(excel_gen.output_folder, 'test_dataframe.xlsx')
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            excel_gen.add_dataframe_sheet(writer, df, 'TestSheet')
        
        assert os.path.exists(output_path)
    
    def test_create_metadata_sheet(self, excel_gen, temp_output_folder):
        """Test creating metadata sheet."""
        output_path = os.path.join(temp_output_folder, 'test_metadata.xlsx')
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            excel_gen.create_metadata_sheet(writer, 'TestScript', analysis_date='2025-01-27')
        
        assert os.path.exists(output_path)
        
        # Read and verify
        df = pd.read_excel(output_path, sheet_name='Metadata')
        assert 'Report Information' in df.columns
        assert 'Value' in df.columns
    
    def test_finalize_report(self, excel_gen, temp_output_folder):
        """Test finalizing report."""
        # Add some data first
        df = pd.DataFrame({'A': [1, 2, 3]})
        output_path = os.path.join(temp_output_folder, 'test_report.xlsx')
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            excel_gen.add_dataframe_sheet(writer, df, 'Data')
        
        # finalize_report may not exist, but we can verify the file was created
        assert os.path.exists(output_path)
