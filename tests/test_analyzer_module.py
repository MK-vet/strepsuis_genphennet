#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for Analyzer module.
"""

import os
import pytest
import pandas as pd
import numpy as np
import tempfile

from strepsuis_genphennet.analyzer import NetworkAnalyzer as GenPhenNetAnalyzer


class TestGenPhenNetAnalyzer:
    """Test GenPhenNetAnalyzer class."""
    
    def test_analyzer_init_default(self):
        """Test analyzer initialization with default config."""
        analyzer = GenPhenNetAnalyzer()
        
        assert analyzer is not None
        assert hasattr(analyzer, 'config')
    
    def test_analyzer_init_with_config(self):
        """Test analyzer initialization with custom config."""
        from strepsuis_genphennet.config import Config
        
        config = Config()
        analyzer = GenPhenNetAnalyzer(config=config)
        
        assert analyzer is not None
        assert analyzer.config is not None
    
    def test_analyzer_init_with_kwargs(self):
        """Test analyzer initialization with kwargs."""
        analyzer = GenPhenNetAnalyzer(
            data_dir='.',
            output_dir='output'
        )
        
        assert analyzer is not None
    
    def test_analyzer_with_real_data_path(self):
        """Test analyzer with real data paths."""
        real_data_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'data'
        )
        
        if os.path.exists(real_data_path):
            analyzer = GenPhenNetAnalyzer(data_dir=real_data_path)
            
            assert analyzer is not None
            assert analyzer.data_dir == real_data_path
        else:
            pytest.skip("Real data not found")
    
    def test_analyzer_has_run_method(self):
        """Test that analyzer has run method."""
        analyzer = GenPhenNetAnalyzer()
        
        assert hasattr(analyzer, 'run')
    
    def test_analyzer_results_attribute(self):
        """Test results attribute."""
        analyzer = GenPhenNetAnalyzer()
        
        assert hasattr(analyzer, 'results')
        assert analyzer.results is None  # Initially None


class TestAnalyzerWithCategories:
    """Test analyzer with real data directories."""
    
    def test_analyzer_with_real_data_dir(self):
        """Test analyzer with real data directory."""
        real_data_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'data'
        )
        
        if os.path.exists(real_data_path):
            analyzer = GenPhenNetAnalyzer(
                data_dir=real_data_path,
                output_dir=tempfile.mkdtemp()
            )
            
            assert analyzer is not None
        else:
            pytest.skip("Real data not found")
    
    def test_analyzer_output_dir_creation(self):
        """Test that output dir is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = GenPhenNetAnalyzer(output_dir=tmpdir)
            
            assert analyzer.output_dir == tmpdir
