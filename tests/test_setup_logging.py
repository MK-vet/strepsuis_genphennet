#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for setup_logging function.
"""

import os
import tempfile
import shutil
import pytest
import logging

from strepsuis_genphennet.network_analysis_core import setup_logging


class TestSetupLogging:
    """Tests for setup_logging function."""
    
    @pytest.fixture
    def temp_cwd(self):
        """Change to temporary directory."""
        original_cwd = os.getcwd()
        temp_dir = tempfile.mkdtemp()
        os.chdir(temp_dir)

        # Clear existing logging handlers
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        yield temp_dir

        # Clean up logging handlers after test
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        os.chdir(original_cwd)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_setup_logging_creates_output_dir(self, temp_cwd):
        """Test that setup_logging creates output directory."""
        setup_logging()
        
        assert os.path.exists('output')
    
    def test_setup_logging_creates_log_file(self, temp_cwd):
        """Test that setup_logging creates log file."""
        setup_logging()
        
        # Log something
        logging.info("Test message")
        
        log_file = os.path.join('output', 'network_analysis_log.txt')
        assert os.path.exists(log_file)
    
    def test_setup_logging_multiple_calls(self, temp_cwd):
        """Test multiple calls to setup_logging."""
        setup_logging()
        setup_logging()
        setup_logging()
        
        assert os.path.exists('output')
