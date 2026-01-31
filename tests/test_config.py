"""Tests for config module."""

import tempfile
from pathlib import Path

import pytest

from strepsuis_genphennet.config import Config


def test_config_defaults():
    """Test default configuration values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(data_dir=tmpdir)
        assert hasattr(config, "bootstrap_iterations")
        assert hasattr(config, "fdr_alpha")
        assert hasattr(config, "random_seed")
        assert config.bootstrap_iterations == 500
        assert config.fdr_alpha == 0.05
        assert config.random_seed == 42


def test_config_custom_values():
    """Test configuration with custom values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            data_dir=tmpdir,
            output_dir=tmpdir,
            bootstrap_iterations=1000,
            fdr_alpha=0.01,
            random_seed=123,
        )
        assert config.bootstrap_iterations == 1000
        assert config.fdr_alpha == 0.01
        assert config.random_seed == 123


def test_config_data_dir_validation():
    """Test data directory validation."""
    with pytest.raises(ValueError):
        Config(data_dir="/nonexistent/directory")


def test_config_fdr_alpha_validation():
    """Test FDR alpha validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError):
            Config(data_dir=tmpdir, fdr_alpha=1.5)
        with pytest.raises(ValueError):
            Config(data_dir=tmpdir, fdr_alpha=-0.1)


def test_config_bootstrap_validation():
    """Test bootstrap iterations validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError):
            Config(data_dir=tmpdir, bootstrap_iterations=50)


def test_config_output_dir_creation():
    """Test output directory is created if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        Config(data_dir=tmpdir, output_dir=str(output_dir))
        assert output_dir.exists()


def test_config_from_dict():
    """Test Config creation from dictionary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dict = {
            "data_dir": tmpdir,
            "output_dir": tmpdir,
            "bootstrap_iterations": 200,
            "fdr_alpha": 0.1,
        }
        config = Config.from_dict(config_dict)
        assert config.bootstrap_iterations == 200
        assert config.fdr_alpha == 0.1


def test_config_reporting_parameters():
    """Test reporting configuration parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(data_dir=tmpdir)
        assert hasattr(config, "generate_html")
        assert hasattr(config, "generate_excel")
        assert hasattr(config, "save_png_charts")
        assert config.generate_html is True
        assert config.generate_excel is True


def test_config_parallel_processing():
    """Test parallel processing configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(data_dir=tmpdir, n_jobs=4)
        assert config.n_jobs == 4
