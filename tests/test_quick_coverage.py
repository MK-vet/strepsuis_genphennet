#!/usr/bin/env python3
"""Quick workflow coverage tests - optimized for speed and coverage increase"""

import pytest
import pandas as pd

import logging
logger = logging.getLogger(__name__)
from pathlib import Path


@pytest.fixture
def example_data():
    """Load example data if available"""
    base_dir = Path(__file__).parent.parent
    examples_dir = base_dir / "examples"
    
    # Try to find data files
    data_files = list(examples_dir.glob("*.csv"))
    if not data_files:
        pytest.skip("Example data not available")
    
    data = {}
    for f in data_files:
        try:
            data[f.stem] = pd.read_csv(f)
        except (ValueError, KeyError, AttributeError) as e:

            logger.warning(f"Operation failed: {e}")
            pass
    
    data["examples_dir"] = str(examples_dir)
    return data


class TestQuickWorkflow:
    """Quick workflow tests to increase coverage"""
    
    def test_config_and_analyzer_init(self, example_data, tmp_path):
        """Test configuration and analyzer initialization"""
        from strepsuis_genphennet.config import Config
        from strepsuis_genphennet.analyzer import get_analyzer
        
        config = Config(
            data_dir=example_data["examples_dir"],
            output_dir=str(tmp_path),
            bootstrap_iterations=100,
            random_seed=42
        )
        
        analyzer = get_analyzer(config)
        assert analyzer is not None
        assert analyzer.config == config
    
    def test_module_imports(self):
        """Test that core modules import successfully"""
        import strepsuis_genphennet
        from strepsuis_genphennet import config
        from strepsuis_genphennet import cli
        from strepsuis_genphennet import analyzer
        
        # Verify modules loaded
        assert config is not None
        assert cli is not None
        assert analyzer is not None
    
    def test_utilities_coverage(self, tmp_path):
        """Test utility functions to increase coverage"""
        from strepsuis_genphennet.config import Config
        
        # Test config creation
        config = Config(data_dir=str(tmp_path), output_dir=str(tmp_path / "output"))
        assert config.bootstrap_iterations > 0
        assert config.fdr_alpha > 0
        assert config.random_seed is not None
        
        # Test from_dict
        config_dict = {
            "data_dir": str(tmp_path),
            "output_dir": str(tmp_path / "output2")
        }
        config2 = Config.from_dict(config_dict)
        assert config2.data_dir == str(tmp_path)
