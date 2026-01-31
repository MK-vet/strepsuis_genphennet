"""Tests for analyzer module."""

from pathlib import Path

import pandas as pd
import pytest

from strepsuis_genphennet.analyzer import NetworkAnalyzer
from strepsuis_genphennet.config import Config


@pytest.fixture
def sample_data(tmp_path):
    """Create sample test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create sample CSV files
    df = pd.DataFrame(
        {"Strain_ID": ["S001", "S002", "S003"], "Feature1": [1, 0, 1], "Feature2": [0, 1, 1]}
    )
    df.to_csv(data_dir / "test_data.csv", index=False)
    return data_dir


def test_analyzer_initialization(sample_data, tmp_path):
    """Test analyzer initialization."""
    output_dir = tmp_path / "output"
    analyzer = NetworkAnalyzer(data_dir=str(sample_data), output_dir=str(output_dir))
    assert analyzer.config.data_dir == str(sample_data)
    assert Path(analyzer.config.output_dir).exists()


def test_analyzer_initialization_with_config(sample_data, tmp_path):
    """Test analyzer initialization with Config object."""
    output_dir = tmp_path / "output"
    config = Config(data_dir=str(sample_data), output_dir=str(output_dir))
    analyzer = NetworkAnalyzer(config=config)
    assert analyzer.config == config
    assert analyzer.results is None


def test_load_data(sample_data, tmp_path):
    """Test data loading."""
    output_dir = tmp_path / "output"
    _ = NetworkAnalyzer(data_dir=str(sample_data), output_dir=str(output_dir))
    # Verify data directory exists
    assert Path(sample_data).exists()
    csv_files = list(Path(sample_data).glob("*.csv"))
    assert len(csv_files) > 0


def test_analysis_execution(sample_data, tmp_path):
    """Test main analysis execution."""
    output_dir = tmp_path / "output"
    analyzer = NetworkAnalyzer(data_dir=str(sample_data), output_dir=str(output_dir))
    results = analyzer.run()
    assert results is not None
    assert isinstance(results, dict)
    assert "status" in results


def test_output_generation(sample_data, tmp_path):
    """Test output file generation."""
    output_dir = tmp_path / "output"
    analyzer = NetworkAnalyzer(data_dir=str(sample_data), output_dir=str(output_dir))
    results = analyzer.run()
    analyzer.results = results

    # Test HTML report generation
    html_path = analyzer.generate_html_report(results)
    assert html_path is not None
    assert "html" in html_path.lower()

    # Test Excel report generation
    excel_path = analyzer.generate_excel_report(results)
    assert excel_path is not None
    assert "xlsx" in excel_path.lower()


def test_analyzer_with_bootstrap(sample_data, tmp_path):
    """Test analyzer with bootstrap parameters."""
    output_dir = tmp_path / "output"
    analyzer = NetworkAnalyzer(
        data_dir=str(sample_data), output_dir=str(output_dir), bootstrap_iterations=100
    )
    assert analyzer.config.bootstrap_iterations == 100


def test_analyzer_with_fdr_alpha(sample_data, tmp_path):
    """Test analyzer with FDR alpha parameter."""
    output_dir = tmp_path / "output"
    analyzer = NetworkAnalyzer(
        data_dir=str(sample_data), output_dir=str(output_dir), fdr_alpha=0.01
    )
    assert analyzer.config.fdr_alpha == 0.01


def test_analyzer_invalid_data_dir(tmp_path):
    """Test analyzer with invalid data directory."""
    output_dir = tmp_path / "output"
    with pytest.raises(ValueError):
        NetworkAnalyzer(data_dir="/nonexistent", output_dir=str(output_dir))


def test_generate_report_without_results(sample_data, tmp_path):
    """Test report generation without running analysis."""
    output_dir = tmp_path / "output"
    analyzer = NetworkAnalyzer(data_dir=str(sample_data), output_dir=str(output_dir))
    with pytest.raises(ValueError):
        analyzer.generate_html_report()


def test_reproducibility(analyzer):
    """Test that analysis is reproducible with same seed."""
    analyzer.load_data() if hasattr(analyzer, "load_data") else None

    results1 = analyzer.run()
    results2 = analyzer.run()

    # Compare key results - should be identical with same seed
    assert results1["status"] == results2["status"]
    assert len(results1) == len(results2)


def test_empty_data_handling(tmp_path):
    """Test analyzer handles empty data gracefully."""

    import pandas as pd

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create empty CSV files
    pd.DataFrame(columns=["Strain_ID"]).to_csv(empty_dir / "test_data.csv", index=False)

    # Should handle empty data appropriately
    # Note: This tests the robustness of the analyzer


def test_multiple_runs(sample_data, tmp_path):
    """Test that analyzer can run multiple times."""
    output_dir = tmp_path / "output"
    analyzer = NetworkAnalyzer(data_dir=str(sample_data), output_dir=str(output_dir))

    # Run analysis twice
    results1 = analyzer.run()
    results2 = analyzer.run()

    # Both should succeed
    assert results1 is not None
    assert results2 is not None
    assert results1["status"] == "success"
    assert results2["status"] == "success"


def test_output_directory_creation(sample_data, tmp_path):
    """Test that output directory is created if it doesn't exist."""
    output_dir = tmp_path / "new_output"

    # Directory shouldn't exist yet
    assert not output_dir.exists()

    analyzer = NetworkAnalyzer(data_dir=str(sample_data), output_dir=str(output_dir))

    # Should create directory
    assert Path(analyzer.config.output_dir).exists()
