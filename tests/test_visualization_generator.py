import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timezone
from typing import Tuple, Dict, List, Optional, Generator, Mapping, Union, Any, cast
from unittest.mock import patch

# Import the module to test
import visualization_generator

@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def sample_correlation_data() -> Tuple[List[Dict[str, Any]], Dict[int, List[Optional[float]]]]:
    """Create sample correlation data for testing."""
    configs = [
        {'config_id': 1, 'indicator_name': 'RSI', 'params': {'period': 14}},
        {'config_id': 2, 'indicator_name': 'MACD', 'params': {'fast': 12, 'slow': 26}}
    ]
    
    # Create correlations with explicit Optional[float] type
    correlations: Dict[int, List[Optional[float]]] = {
        1: [0.5, 0.4, 0.3, 0.2, 0.1],  # 5 lags for config 1
        2: [0.6, 0.5, 0.4, 0.3, 0.2]   # 5 lags for config 2
    }
    
    return configs, correlations

def test_plot_correlation_lines(temp_dir: Path, sample_correlation_data: Tuple[List[Dict[str, Any]], Dict[int, List[Optional[float]]]]) -> None:
    """Test correlation line plot generation."""
    configs, correlations = sample_correlation_data
    max_lag = 5
    
    # Test basic plot generation
    output_file: Optional[Path] = visualization_generator.plot_correlation_lines(
        cast(Dict[int, List[Optional[float]]], correlations), configs, max_lag, temp_dir, "test_lines"
    )
    assert output_file is not None
    assert output_file.exists()
    assert output_file.suffix == '.png'
    
    # Test with empty correlations
    empty_corrs: Dict[int, List[Optional[float]]] = {}
    output_file = visualization_generator.plot_correlation_lines(
        empty_corrs, [], max_lag, temp_dir, "test_empty"
    )
    assert output_file is None  # Should not create file for empty data

def test_generate_combined_correlation_chart(temp_dir: Path, sample_correlation_data: Tuple[List[Dict[str, Any]], Dict[int, List[Optional[float]]]]) -> None:
    """Test combined correlation chart generation."""
    configs, correlations = sample_correlation_data
    max_lag = 5
    
    # Test basic chart generation
    output_file: Optional[Path] = visualization_generator.generate_combined_correlation_chart(
        cast(Dict[int, List[Optional[float]]], correlations), configs, max_lag, temp_dir, "test_combined"
    )
    assert output_file is not None
    assert output_file.exists()
    assert output_file.suffix == '.png'
    
    # Test with invalid data
    invalid_corrs: Dict[int, List[Optional[float]]] = {1: [None, 0.5, 0.3]}  # Contains None
    output_file = visualization_generator.generate_combined_correlation_chart(
        invalid_corrs, configs[:1], max_lag, temp_dir, "test_invalid"
    )
    assert output_file is None  # Should not create file for invalid data

def test_generate_enhanced_heatmap(temp_dir: Path, sample_correlation_data: Tuple[List[Dict[str, Any]], Dict[int, List[Optional[float]]]]) -> None:
    """Test enhanced heatmap generation."""
    configs, correlations = sample_correlation_data
    max_lag = 5
    
    # Test basic heatmap generation
    output_file: Optional[Path] = visualization_generator.generate_enhanced_heatmap(
        cast(Dict[int, List[Optional[float]]], correlations), configs, max_lag, temp_dir, "test_heatmap"
    )
    assert output_file is not None
    assert output_file.exists()
    assert output_file.suffix == '.png'
    
    # Test with single config
    single_config = configs[:1]
    single_corr: Dict[int, List[Optional[float]]] = {1: correlations[1]}
    output_file = visualization_generator.generate_enhanced_heatmap(
        single_corr, single_config, max_lag, temp_dir, "test_single"
    )
    assert output_file is not None
    assert output_file.exists()

def test_generate_correlation_envelope_chart(temp_dir: Path, sample_correlation_data: Tuple[List[Dict[str, Any]], Dict[int, List[Optional[float]]]]) -> None:
    """Test correlation envelope chart generation."""
    configs, correlations = sample_correlation_data
    max_lag = 5
    
    # Test basic envelope chart generation
    output_file: Optional[Path] = visualization_generator.generate_correlation_envelope_chart(
        cast(Dict[int, List[Optional[float]]], correlations), configs, max_lag, temp_dir, "test_envelope"
    )
    assert output_file is not None
    assert output_file.exists()
    assert output_file.suffix == '.png'
    
    # Test with varying correlation lengths
    varying_corrs: Dict[int, List[Optional[float]]] = {
        1: correlations[1],
        2: correlations[2][:3]  # Shorter correlation series
    }
    output_file = visualization_generator.generate_correlation_envelope_chart(
        varying_corrs, configs, max_lag, temp_dir, "test_varying"
    )
    assert output_file is not None
    assert output_file.exists()

def test_generate_peak_correlation_report(temp_dir: Path, sample_correlation_data: Tuple[List[Dict[str, Any]], Dict[int, List[Optional[float]]]]) -> None:
    """Test peak correlation report generation."""
    configs, correlations = sample_correlation_data
    max_lag = 5
    
    # Test basic report generation
    output_file: Optional[Path] = visualization_generator.generate_peak_correlation_report(
        cast(Dict[int, List[Optional[float]]], correlations), configs, max_lag, temp_dir, "test_peak"
    )
    assert output_file is not None
    assert output_file.exists()
    assert output_file.suffix == '.csv'
    
    # Verify report content (CSV format)
    content = output_file.read_text()
    assert "Config ID" in content
    assert "Indicator" in content
    assert "Parameters" in content
    assert "Peak Positive Corr" in content
    assert "Peak Negative Corr" in content
    assert "Peak Absolute Corr" in content
    
    # Test with empty correlations
    output_file = visualization_generator.generate_peak_correlation_report(
        {}, [], max_lag, temp_dir, "test_empty"
    )
    assert output_file is None  # Should not create file for empty data

def test_error_handling(temp_dir: Path, sample_correlation_data: Tuple[List[Dict[str, Any]], Dict[int, List[Optional[float]]]]) -> None:
    """Test error handling in visualization functions."""
    configs, correlations = sample_correlation_data
    max_lag = 5
    
    # Test with invalid directory (mocked)
    invalid_dir = Path("/nonexistent/path")
    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            visualization_generator.plot_correlation_lines(
                cast(Dict[int, List[Optional[float]]], correlations), configs, max_lag, invalid_dir, "test_error"
            )
    
    # Test with invalid correlation data
    invalid_corrs: Dict[int, str] = {1: "not a list"}  # Invalid correlation format
    result = visualization_generator.generate_enhanced_heatmap(
        cast(Dict[int, List[Optional[float]]], invalid_corrs), configs, max_lag, temp_dir, "test_error"
    )
    assert result is None  # Should not create file for invalid data
    
    # Test with invalid config format
    invalid_configs: List[Dict[str, str]] = [{"invalid": "format"}]
    result = visualization_generator.generate_combined_correlation_chart(
        cast(Dict[int, List[Optional[float]]], correlations), invalid_configs, max_lag, temp_dir, "test_error"
    )
    assert result is None  # Should not create file for invalid config format

def test_performance_with_large_data(temp_dir: Path) -> None:
    """Test visualization performance with larger datasets."""
    # Create larger dataset
    n_configs = 50
    max_lag = 20
    configs = [
        {
            'config_id': i,
            'indicator_name': f'IND_{i}',
            'params': {'param': i}
        }
        for i in range(1, n_configs + 1)
    ]
    
    correlations: Dict[int, List[Optional[float]]] = {
        i: [np.random.uniform(-1, 1) for _ in range(max_lag)]
        for i in range(1, n_configs + 1)
    }
    
    # Test each visualization type with larger dataset
    start_time = datetime.now(timezone.utc)
    
    # Test line plots
    output_file: Optional[Path] = visualization_generator.plot_correlation_lines(
        correlations, configs, max_lag, temp_dir, "test_large_lines"
    )
    assert output_file is not None
    assert output_file.exists()
    
    # Test combined chart
    output_file = visualization_generator.generate_combined_correlation_chart(
        correlations, configs, max_lag, temp_dir, "test_large_combined"
    )
    assert output_file is not None
    assert output_file.exists()
    
    # Test heatmap
    output_file = visualization_generator.generate_enhanced_heatmap(
        correlations, configs, max_lag, temp_dir, "test_large_heatmap"
    )
    assert output_file is not None
    assert output_file.exists()
    
    # Test envelope chart
    output_file = visualization_generator.generate_correlation_envelope_chart(
        correlations, configs, max_lag, temp_dir, "test_large_envelope"
    )
    assert output_file is not None
    assert output_file.exists()
    
    # Verify reasonable execution time (adjust threshold as needed)
    execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
    assert execution_time < 30  # Should complete within 30 seconds 