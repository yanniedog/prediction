import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timezone
from typing import Tuple, Dict, List, Optional, Generator, Mapping, Union, Any, cast
from unittest.mock import patch
import json
from visualization_generator import (
    generate_charts,
    plot_indicator_performance,
    plot_correlation_matrix,
    plot_optimization_results,
    plot_prediction_accuracy,
    _format_chart_data,
    _save_chart
)

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

@pytest.fixture(scope="function")
def sample_data() -> pd.DataFrame:
    """Create sample price data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
    np.random.seed(42)
    data = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.normal(100, 1, 100),
        "high": np.random.normal(101, 1, 100),
        "low": np.random.normal(99, 1, 100),
        "close": np.random.normal(100, 1, 100),
        "volume": np.random.normal(1000, 100, 100)
    })
    data["high"] = data[["open", "close"]].max(axis=1) + abs(np.random.normal(0, 0.1, 100))
    data["low"] = data[["open", "close"]].min(axis=1) - abs(np.random.normal(0, 0.1, 100))
    return data

@pytest.fixture(scope="function")
def sample_indicator_data() -> pd.DataFrame:
    """Create sample indicator data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
    np.random.seed(42)
    data = pd.DataFrame({
        "timestamp": dates,
        "RSI": np.random.uniform(0, 100, 100),
        "BB_upper": np.random.normal(102, 1, 100),
        "BB_middle": np.random.normal(100, 1, 100),
        "BB_lower": np.random.normal(98, 1, 100),
        "MACD": np.random.normal(0, 1, 100),
        "MACD_signal": np.random.normal(0, 1, 100),
        "MACD_hist": np.random.normal(0, 0.5, 100)
    })
    return data

@pytest.fixture(scope="function")
def sample_correlation_matrix() -> pd.DataFrame:
    """Create sample correlation matrix for testing."""
    # Create a correlation matrix
    data = {
        'RSI': [1.0, -0.15, -0.12, -0.08, 0.02, 0.17, 0.16],
        'BB_upper': [-0.15, 1.0, 0.95, 0.85, 0.75, 0.65, 0.55],
        'BB_middle': [-0.12, 0.95, 1.0, 0.90, 0.80, 0.70, 0.60],
        'BB_lower': [-0.08, 0.85, 0.90, 1.0, 0.85, 0.75, 0.65],
        'MACD': [0.02, 0.75, 0.80, 0.85, 1.0, 0.90, 0.80],
        'MACD_signal': [0.17, 0.65, 0.70, 0.75, 0.90, 1.0, 0.85],
        'MACD_hist': [0.16, 0.55, 0.60, 0.65, 0.80, 0.85, 1.0]
    }
    return pd.DataFrame(data, index=data.keys())

@pytest.fixture(scope="function")
def sample_optimization_data() -> Dict[str, Any]:
    """Create sample optimization data for testing."""
    return {
        "RSI": {
            "timeperiod": [14, 20, 30, 40, 50],
            "score": [0.8, 0.85, 0.75, 0.7, 0.65]
        },
        "BB": {
            "timeperiod": [20, 30, 40, 50, 60],
            "nbdevup": [2.0, 2.5, 3.0, 2.0, 2.5],
            "nbdevdn": [2.0, 2.5, 3.0, 2.0, 2.5],
            "score": [0.75, 0.8, 0.7, 0.65, 0.6]
        }
    }

@pytest.fixture(scope="function")
def sample_prediction_data() -> Tuple[pd.Series, pd.Series]:
    """Create sample prediction data for testing."""
    np.random.seed(42)
    actual = pd.Series(np.random.normal(0, 1, 100))
    predicted = actual + np.random.normal(0, 0.2, 100)  # Add some noise
    return actual, predicted

def test_format_chart_data(sample_data: pd.DataFrame, sample_indicator_data: pd.DataFrame):
    """Test chart data formatting."""
    # Test price data formatting
    formatted_data = _format_chart_data(sample_data, "price")
    assert isinstance(formatted_data, pd.DataFrame)
    assert all(col in formatted_data.columns for col in ["open", "high", "low", "close"])
    assert not formatted_data.empty
    
    # Test indicator data formatting
    formatted_data = _format_chart_data(sample_indicator_data, "indicator")
    assert isinstance(formatted_data, pd.DataFrame)
    assert all(col in formatted_data.columns for col in ["RSI", "BB_upper", "BB_middle", "BB_lower"])
    assert not formatted_data.empty
    
    # Test invalid data type
    with pytest.raises(ValueError):
        _format_chart_data(sample_data, "invalid_type")

def test_save_chart(temp_dir: Path):
    """Test chart saving functionality."""
    import matplotlib.pyplot as plt
    
    # Create a simple test chart
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    
    # Test saving chart
    chart_path = temp_dir / "test_chart.png"
    _save_chart(fig, chart_path)
    assert chart_path.exists()
    
    # Test saving with different formats
    for fmt in ["png", "jpg", "pdf", "svg"]:
        chart_path = temp_dir / f"test_chart.{fmt}"
        _save_chart(fig, chart_path)
        assert chart_path.exists()
    
    plt.close(fig)

def test_plot_indicator_performance(temp_dir: Path, sample_data: pd.DataFrame, sample_indicator_data: pd.DataFrame):
    """Test indicator performance plotting."""
    # Test basic plotting
    chart_path = temp_dir / "indicator_performance.png"
    plot_indicator_performance(sample_data, sample_indicator_data, chart_path)
    assert chart_path.exists()
    
    # Test with specific indicators
    chart_path = temp_dir / "rsi_performance.png"
    plot_indicator_performance(sample_data, sample_indicator_data[["RSI"]], chart_path)
    assert chart_path.exists()
    
    # Test with invalid data
    with pytest.raises(ValueError):
        plot_indicator_performance(pd.DataFrame(), sample_indicator_data, chart_path)
    
    # Test with missing required columns
    invalid_data = sample_data.drop(columns=["close"])
    with pytest.raises(ValueError):
        plot_indicator_performance(invalid_data, sample_indicator_data, chart_path)

def test_plot_correlation_matrix(temp_dir: Path, sample_correlation_matrix: pd.DataFrame):
    """Test correlation matrix plotting."""
    # Test basic plotting
    chart_path = temp_dir / "correlation_matrix.png"
    plot_correlation_matrix(sample_correlation_matrix, chart_path)
    assert chart_path.exists()
    
    # Test with custom title
    chart_path = temp_dir / "correlation_matrix_custom.png"
    plot_correlation_matrix(sample_correlation_matrix, chart_path, title="Custom Title")
    assert chart_path.exists()
    
    # Test with invalid data
    with pytest.raises(ValueError):
        plot_correlation_matrix(pd.DataFrame(), chart_path)
    
    # Test with non-square correlation matrix
    invalid_data = sample_correlation_matrix.iloc[:, :-1]  # Remove one column
    with pytest.raises(ValueError):
        plot_correlation_matrix(invalid_data, chart_path)

def test_plot_optimization_results(temp_dir: Path, sample_optimization_data: Dict[str, Any]):
    """Test optimization results plotting."""
    # Test basic plotting
    chart_path = temp_dir / "optimization_results.png"
    plot_optimization_results(sample_optimization_data, chart_path)
    assert chart_path.exists()
    
    # Test with single parameter
    single_param_data = {"RSI": sample_optimization_data["RSI"]}
    chart_path = temp_dir / "single_param_optimization.png"
    plot_optimization_results(single_param_data, chart_path)
    assert chart_path.exists()
    
    # Test with invalid data
    with pytest.raises(ValueError):
        plot_optimization_results({}, chart_path)
    
    # Test with missing required fields
    invalid_data = {"RSI": {"timeperiod": [14, 20]}}  # Missing score
    with pytest.raises(ValueError):
        plot_optimization_results(invalid_data, chart_path)

def test_plot_prediction_accuracy(temp_dir: Path, sample_prediction_data: Tuple[pd.Series, pd.Series]):
    """Test prediction accuracy plotting."""
    actual, predicted = sample_prediction_data
    
    # Test basic plotting
    chart_path = temp_dir / "prediction_accuracy.png"
    plot_prediction_accuracy(actual, predicted, chart_path)
    assert chart_path.exists()
    
    # Test with custom title
    chart_path = temp_dir / "prediction_accuracy_custom.png"
    plot_prediction_accuracy(actual, predicted, chart_path, title="Custom Title")
    assert chart_path.exists()
    
    # Test with invalid data
    with pytest.raises(ValueError):
        plot_prediction_accuracy(pd.Series(), predicted, chart_path)
    
    # Test with mismatched lengths
    with pytest.raises(ValueError):
        plot_prediction_accuracy(actual, predicted.iloc[:-1], chart_path)

def test_generate_charts(temp_dir: Path, sample_data: pd.DataFrame, sample_indicator_data: pd.DataFrame,
                        sample_correlation_matrix: pd.DataFrame, sample_optimization_data: Dict[str, Any],
                        sample_prediction_data: Tuple[pd.Series, pd.Series]):
    """Test main chart generation function."""
    actual, predicted = sample_prediction_data
    
    # Test generating all charts
    chart_paths = generate_charts(
        temp_dir,
        price_data=sample_data,
        indicator_data=sample_indicator_data,
        correlation_data=sample_correlation_matrix,
        optimization_data=sample_optimization_data,
        actual_predictions=actual,
        predicted_predictions=predicted
    )
    
    # Verify all charts were generated
    assert all(path.exists() for path in chart_paths.values())
    
    # Test generating subset of charts
    chart_paths = generate_charts(
        temp_dir,
        price_data=sample_data,
        indicator_data=sample_indicator_data
    )
    assert "indicator_performance" in chart_paths
    assert "correlation_matrix" not in chart_paths
    
    # Test with invalid data
    with pytest.raises(ValueError):
        generate_charts(temp_dir)  # No data provided

def test_chart_customization(temp_dir: Path, sample_data: pd.DataFrame, sample_indicator_data: pd.DataFrame):
    """Test chart customization options."""
    # Test custom figure size
    chart_path = temp_dir / "custom_size.png"
    plot_indicator_performance(sample_data, sample_indicator_data, chart_path, figsize=(12, 8))
    assert chart_path.exists()
    
    # Test custom colors
    chart_path = temp_dir / "custom_colors.png"
    plot_indicator_performance(sample_data, sample_indicator_data, chart_path, colors=["red", "blue", "green"])
    assert chart_path.exists()
    
    # Test custom style
    chart_path = temp_dir / "custom_style.png"
    plot_indicator_performance(sample_data, sample_indicator_data, chart_path, style="dark_background")
    assert chart_path.exists()
    
    # Test custom labels
    chart_path = temp_dir / "custom_labels.png"
    plot_indicator_performance(
        sample_data,
        sample_indicator_data,
        chart_path,
        title="Custom Title",
        xlabel="Custom X",
        ylabel="Custom Y"
    )
    assert chart_path.exists()

def test_error_handling(temp_dir: Path):
    """Test error handling in visualization."""
    # Test invalid file path
    with pytest.raises(ValueError):
        plot_indicator_performance(pd.DataFrame(), pd.DataFrame(), Path("/invalid/path/chart.png"))
    
    # Test invalid figure size
    with pytest.raises(ValueError):
        plot_indicator_performance(pd.DataFrame(), pd.DataFrame(), temp_dir / "chart.png", figsize=(0, 0))
    
    # Test invalid color list
    with pytest.raises(ValueError):
        plot_indicator_performance(pd.DataFrame(), pd.DataFrame(), temp_dir / "chart.png", colors=["invalid"])
    
    # Test invalid style
    with pytest.raises(ValueError):
        plot_indicator_performance(pd.DataFrame(), pd.DataFrame(), temp_dir / "chart.png", style="invalid_style") 