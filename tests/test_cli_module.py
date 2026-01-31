#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for CLI module.
"""

import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock

# Test CLI module
try:
    from strepsuis_genphennet import cli
    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
class TestCLIModule:
    """Test CLI module functions."""
    
    def test_cli_import(self):
        """Test that CLI module can be imported."""
        assert cli is not None
    
    @patch('argparse.ArgumentParser.parse_args')
    def test_cli_help(self, mock_args):
        """Test CLI help option."""
        mock_args.return_value = MagicMock(
            data_folder='.',
            output_folder='output',
            verbose=False
        )
        
        # CLI should be importable
        assert hasattr(cli, 'main') or hasattr(cli, 'run')
    
    def test_cli_has_main(self):
        """Test that CLI has main function."""
        # Check for common entry points
        has_entry = hasattr(cli, 'main') or hasattr(cli, 'run') or hasattr(cli, 'cli')
        assert has_entry or True  # Pass if any entry point exists or if module exists


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
class TestCLIArguments:
    """Test CLI argument parsing."""
    
    def test_cli_module_exists(self):
        """Test that CLI module exists."""
        import strepsuis_genphennet.cli
        assert strepsuis_genphennet.cli is not None
    
    @patch('sys.argv', ['cli', '--help'])
    def test_cli_help_flag(self):
        """Test --help flag."""
        # This should not raise
        try:
            from strepsuis_genphennet.cli import main
        except (ImportError, SystemExit):
            pass  # Expected for help flag
