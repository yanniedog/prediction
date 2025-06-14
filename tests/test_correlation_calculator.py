import pytest
import pandas as pd
import numpy as np
from correlation_calculator import CorrelationCalculator
from datetime import datetime, timedelta
import time
from functools import wraps
import threading
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Any, Optional, Tuple, Callable
import sqlite_manager
from pathlib import Path
import tempfile
import shutil
import json

def timeout(seconds):
    """Decorator to add timeout to test functions using threading."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = []
            error = []
            
            def target():
                try:
                    result.append(func(*args, **kwargs))
                except Exception as e:
                    error.append(e)
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                raise TimeoutError(f"Test timed out after {seconds} seconds")
            if error:
                raise error[0]
            return result[0]
        return wrapper
    return decorator

@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Fixture to ensure matplotlib figures are cleaned up after each test."""
    yield
    try:
        plt.close('all')
    except (ImportError, NameError):
        pass

@pytest.fixture
def correlation_calculator():
    """Create a CorrelationCalculator instance."""
    return CorrelationCalculator()

@pytest.fixture
def test_data() -> pd.DataFrame:
    """Create test data for correlation calculations."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(200, 300, 100),
        'low': np.random.uniform(50, 100, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000, 5000, 100),
        'RSI_14': np.random.uniform(0, 100, 100),
        'MACD_12_26_9': np.random.uniform(-10, 10, 100),
        'returns': np.random.normal(0, 0.02, 100).cumsum(),
        'ma20': np.random.uniform(100, 200, 100),
        'ma50': np.random.uniform(100, 200, 100)
    })
    return data

def test_correlation_calculator_initialization():
    calculator = CorrelationCalculator()
    assert calculator is not None

@pytest.mark.timeout(30)  # Explicit timeout for this test
def test_correlation_report_generation(correlation_calculator, test_data):
    """Test correlation report generation with timeout and dependency checks."""
    # Use pytest.importorskip for optional dependencies
    pytest.importorskip('matplotlib')
    pytest.importorskip('seaborn')
    
    # Use a small subset of data for testing
    test_subset = test_data.iloc[:50].copy()
    
    # Generate report with timeout check (direct call, let pytest timeout handle it)
    report = correlation_calculator.generate_correlation_report(test_subset)
    
    # Basic structure validation
    assert isinstance(report, dict)
    assert 'summary' in report
    assert 'correlation_matrix' in report
    assert 'visualizations' in report
    assert 'analysis' in report
    
    # Validate summary statistics
    summary = report['summary']
    assert isinstance(summary, dict)
    assert 'total_correlations' in summary
    assert 'significant_correlations' in summary
    assert 'mean_correlation' in summary
    assert isinstance(summary['total_correlations'], int)
    assert isinstance(summary['significant_correlations'], int)
    assert isinstance(summary['mean_correlation'], float)
    
    # Validate correlation matrix
    corr_matrix = report['correlation_matrix']
    assert isinstance(corr_matrix, pd.DataFrame)
    assert not corr_matrix.empty
    assert corr_matrix.shape[0] == corr_matrix.shape[1]  # Square matrix
    assert corr_matrix.index.equals(corr_matrix.columns)  # Symmetric
    
    # Validate analysis section - can be None if sklearn is not available
    analysis = report['analysis']
    if correlation_calculator.SKLEARN_AVAILABLE:
        assert isinstance(analysis, dict)
        assert 'decomposition' in analysis
        assert 'stability' in analysis
        assert isinstance(analysis['decomposition'], dict)
        assert isinstance(analysis['stability'], dict)
    else:
        assert analysis is None
    
    # Validate visualizations if available
    visualizations = report['visualizations']
    assert isinstance(visualizations, dict)
    if 'heatmap' in visualizations:
        assert isinstance(visualizations['heatmap'], Figure)
        plt.close(visualizations['heatmap'])  # Clean up

@pytest.mark.timeout(20)
def test_correlation_calculation(correlation_calculator, test_data):
    """Test basic correlation calculation with timeout."""
    # Use the full test_data to satisfy min_data_points
    test_subset = test_data.copy()
    
    # Calculate correlation using the correct signature
    result = correlation_calculator.calculate_correlation(
        test_subset['close'],
        test_subset['volume'],
        method='pearson'
    )
    
    assert isinstance(result, (float, np.floating))
    assert -1 <= result <= 1

@pytest.mark.timeout(20)
def test_rolling_correlation(correlation_calculator, test_data):
    """Test rolling correlation calculation with timeout."""
    # Use a small subset of data
    test_subset = test_data.iloc[:40].copy()
    
    # Calculate rolling correlation (direct call, let pytest timeout handle it)
    result = correlation_calculator.calculate_rolling_correlation(
        test_subset['close'],
        test_subset['volume'],
        window=5
    )
    
    assert isinstance(result, pd.Series)
    assert not result.empty
    assert len(result) == len(test_subset)
    assert result.notna().any()  # Should have some non-NaN values
    assert all(-1 <= x <= 1 for x in result.dropna())

@pytest.mark.timeout(20)
def test_correlation_anomalies(correlation_calculator, test_data):
    """Test correlation anomaly detection with timeout."""
    # Use a small subset of data
    test_subset = test_data.iloc[:40].copy()
    
    # Detect anomalies (direct call, let pytest timeout handle it)
    result = correlation_calculator.detect_correlation_anomalies(
        test_subset,
        threshold=0.8
    )
    
    assert isinstance(result, dict)
    # Should contain keys for each column
    assert all(isinstance(k, str) for k in result.keys())

@pytest.mark.timeout(20)
def test_correlation_visualization(correlation_calculator, test_data):
    """Test correlation visualization with timeout and dependency check."""
    # Skip if visualization dependencies are not available
    pytest.importorskip('matplotlib')
    pytest.importorskip('seaborn')
    
    # Use a small subset of data
    test_subset = test_data.iloc[:30].copy()
    
    # Test both single correlation and matrix visualization
    # Single correlation visualization
    fig1 = correlation_calculator.visualize_correlation(
        test_subset['close'],
        test_subset['volume']
    )
    assert fig1 is not None
    
    # Matrix visualization
    fig2 = correlation_calculator.visualize_correlation_matrix(test_subset)
    assert fig2 is not None

def test_correlation_significance(correlation_calculator, test_data):
    """Test correlation significance calculation."""
    test_subset = test_data.iloc[:30].copy()
    result = correlation_calculator.test_correlation_significance(
        test_subset['close'],
        test_subset['volume'],
        alpha=0.05
    )
    assert isinstance(result, dict)
    assert 'correlation' in result
    assert 'p_value' in result
    assert 'significant' in result

def test_correlation_stability(correlation_calculator, test_data):
    """Test correlation stability analysis with timeout."""
    # Use a small subset of data
    test_subset = test_data.iloc[:40].copy()
    
    # Create a DataFrame with the two series
    test_df = pd.DataFrame({
        'close': test_subset['close'],
        'volume': test_subset['volume']
    })
    
    # Analyze stability (direct call, let pytest timeout handle it)
    result = correlation_calculator.analyze_correlation_stability(
        test_df,  # Pass DataFrame instead of separate series
        window_size=5
    )
    
    assert isinstance(result, dict)
    assert 'stability_score' in result
    assert 'trend' in result
    assert 'volatility' in result
    assert isinstance(result['stability_score'], float)
    assert isinstance(result['volatility'], float)
    assert isinstance(result['trend'], str)
    assert 0 <= result['stability_score'] <= 1
    assert result['volatility'] >= 0
    assert result['trend'] in ['increasing', 'decreasing', 'stable']

def test_correlation_decomposition(correlation_calculator, test_data):
    """Test correlation decomposition with timeout."""
    # Use a small subset of data
    test_subset = test_data.iloc[:30].copy()
    
    # Perform decomposition with timeout check
    result = assert_timeout(
        correlation_calculator.decompose_correlation,
        test_subset,
        n_components=2,
        timeout_seconds=5
    )
    
    assert isinstance(result, dict)
    assert 'components' in result
    assert 'explained_variance' in result
    assert 'loadings' in result
    assert isinstance(result['components'], pd.DataFrame)
    assert isinstance(result['explained_variance'], np.ndarray)
    assert isinstance(result['loadings'], pd.DataFrame)
    assert len(result['explained_variance']) == 2
    assert result['loadings'].shape[1] == 2

def test_correlation_clustering(correlation_calculator, test_data):
    """Test correlation clustering methods."""
    # Create sample data
    data = pd.DataFrame({
        'close': test_data['close'],
        'volume': test_data['volume'],
        'returns': test_data['close'].pct_change(),
        'ma20': test_data['close'].rolling(20).mean(),
        'ma50': test_data['close'].rolling(50).mean()
    }).dropna()
    
    # Perform correlation clustering
    clusters = correlation_calculator.cluster_correlations(data)
    assert isinstance(clusters, dict)
    assert 'labels' in clusters
    assert 'centers' in clusters
    assert 'silhouette_score' in clusters

@pytest.mark.timeout(20)
def test_correlation_network(correlation_calculator, test_data):
    """Test correlation network analysis with timeout."""
    # Use a small subset of data
    test_subset = test_data.iloc[:30].copy()
    
    # Analyze network with timeout check
    result = assert_timeout(
        correlation_calculator.analyze_correlation_network,
        test_subset,
        threshold=0.5,
        timeout_seconds=5
    )
    
    assert isinstance(result, dict)
    assert 'nodes' in result
    assert 'edges' in result
    assert 'centrality' in result
    assert isinstance(result['nodes'], list)
    assert isinstance(result['edges'], list)
    assert isinstance(result['centrality'], dict)
    if result['edges']:
        assert all(len(edge) == 3 for edge in result['edges'])  # (source, target, weight)
        assert all(isinstance(edge[2], float) for edge in result['edges'])

@pytest.mark.timeout(20)
def test_correlation_matrix_visualization(correlation_calculator, test_data):
    """Test correlation matrix visualization with timeout and dependency check."""
    # Skip if visualization dependencies are not available
    pytest.importorskip('matplotlib')
    pytest.importorskip('seaborn')
    
    # Use a small subset of data
    test_subset = test_data.iloc[:20].copy()
    
    # Generate matrix visualization with timeout check
    fig = assert_timeout(
        correlation_calculator.visualize_correlation_matrix,
        test_subset,
        timeout_seconds=5
    )
    
    assert isinstance(fig, Figure)
    plt.close(fig)  # Clean up

@pytest.mark.timeout(20)
def test_correlation_regimes(correlation_calculator, test_data):
    """Test correlation regime detection with timeout."""
    # Use a small subset of data
    test_subset = test_data.iloc[:40].copy()
    
    # Detect regimes with timeout check
    result = assert_timeout(
        correlation_calculator.detect_correlation_regimes,
        test_subset,
        n_regimes=2,
        timeout_seconds=5
    )
    
    assert isinstance(result, dict)
    assert 'regimes' in result
    assert 'transition_matrix' in result
    assert isinstance(result['regimes'], list)
    assert isinstance(result['transition_matrix'], np.ndarray)
    assert len(result['regimes']) == 2
    assert result['transition_matrix'].shape == (2, 2)

@pytest.fixture(scope="function")
def sample_data() -> pd.DataFrame:
    """Create sample data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
    np.random.seed(42)
    data = pd.DataFrame({
        "timestamp": dates,
        "price": np.random.normal(100, 1, 100).cumsum(),
        "volume": np.random.normal(1000, 100, 100),
        "indicator1": np.random.normal(0, 1, 100),
        "indicator2": np.random.normal(0, 1, 100)
    })
    # Add some correlation between indicators
    data["indicator2"] = 0.7 * data["indicator1"] + 0.3 * np.random.normal(0, 1, 100)
    return data

def assert_timeout(func: Callable, *args, timeout_seconds: int = 5, **kwargs) -> Any:
    """Utility function to assert function execution within timeout."""
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    assert execution_time < timeout_seconds, f"Function execution took {execution_time:.2f}s, exceeding timeout of {timeout_seconds}s"
    return result

def test_calculate_correlation(correlation_calculator, sample_data):
    """Test basic correlation calculation."""
    result = correlation_calculator.calculate_correlation(
        sample_data['indicator1'],
        sample_data['indicator2'],
        method='pearson'
    )
    assert isinstance(result, float)
    assert -1 <= result <= 1

def test_calculate_lag_correlation(correlation_calculator, sample_data):
    """Test lag correlation calculation."""
    # Test with different lags
    lags = range(-5, 6)
    correlations = correlation_calculator.calculate_lag_correlation(
        sample_data['indicator1'],
        sample_data['indicator2'],
        lags=lags
    )
    assert isinstance(correlations, pd.Series)
    assert len(correlations) == len(lags)
    assert all(-1 <= corr <= 1 for corr in correlations.dropna())

def test_calculate_rolling_correlation(correlation_calculator, sample_data):
    """Test rolling correlation calculation."""
    result = correlation_calculator.calculate_rolling_correlation(
        data1=sample_data['indicator1'],
        data2=sample_data['price'],
        window=20
    )
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_data)
    assert result.notna().any()

def test_calculate_correlation_matrix(correlation_calculator, sample_data):
    """Test correlation matrix calculation."""
    result = correlation_calculator.calculate_correlation_matrix(
        data=sample_data[['indicator1', 'indicator2', 'price']]
    )
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 3)
    assert result.index.equals(result.columns)

def test_test_correlation_significance(correlation_calculator, sample_data):
    """Test correlation significance testing."""
    result = correlation_calculator.test_correlation_significance(
        data1=sample_data['indicator1'],
        data2=sample_data['price']
    )
    assert isinstance(result, dict)
    assert 'correlation' in result
    assert 'p_value' in result
    assert 'significant' in result

def test_plot_correlation_heatmap(correlation_calculator, sample_data):
    """Test correlation heatmap plotting."""
    fig = correlation_calculator.plot_correlation_heatmap(
        data=sample_data[['indicator1', 'indicator2', 'price']]
    )
    assert isinstance(fig, Figure)

def test_plot_correlation_scatter(correlation_calculator, sample_data):
    """Test correlation scatter plot."""
    fig = correlation_calculator.plot_correlation_scatter(
        data1=sample_data['indicator1'],
        data2=sample_data['price']
    )
    assert isinstance(fig, Figure)

def test_decompose_correlation(correlation_calculator, sample_data):
    """Test correlation decomposition."""
    result = correlation_calculator.decompose_correlation(
        data=sample_data[['indicator1', 'indicator2', 'price']],
        n_components=2
    )
    assert isinstance(result, dict)
    assert 'components' in result
    assert 'explained_variance' in result

def test_cluster_correlations(correlation_calculator, sample_data):
    """Test correlation clustering."""
    result = correlation_calculator.cluster_correlations(
        data=sample_data[['indicator1', 'indicator2', 'price']],
        n_clusters=2
    )
    assert isinstance(result, dict)
    assert 'labels' in result
    assert 'centers' in result
    assert 'silhouette_score' in result

def test_analyze_correlation_stability(correlation_calculator, sample_data):
    """Test correlation stability analysis."""
    result = correlation_calculator.analyze_correlation_stability(
        data=sample_data[['indicator1', 'indicator2', 'price']],
        window_size=20
    )
    assert isinstance(result, dict)
    assert 'stability_score' in result
    assert 'volatility' in result
    assert 'trend' in result
    assert isinstance(result['stability_score'], float)
    assert isinstance(result['volatility'], float)
    assert isinstance(result['trend'], str)
    assert 0 <= result['stability_score'] <= 1
    assert result['volatility'] >= 0
    assert result['trend'] in ['increasing', 'decreasing', 'stable']

def test_detect_correlation_regimes(correlation_calculator, sample_data):
    """Test correlation regime detection."""
    result = correlation_calculator.detect_correlation_regimes(
        data=sample_data[['indicator1', 'indicator2', 'price']],
        n_regimes=2
    )
    assert isinstance(result, dict)
    assert 'regime_labels' in result
    assert 'regime_characteristics' in result

def test_analyze_correlation_network(correlation_calculator, sample_data):
    """Test correlation network analysis."""
    result = correlation_calculator.analyze_correlation_network(
        data=sample_data[['indicator1', 'indicator2', 'price']],
        threshold=0.5
    )
    assert isinstance(result, dict)
    assert 'network' in result
    assert 'centrality' in result

def test_detect_correlation_anomalies(correlation_calculator, sample_data):
    """Test correlation anomaly detection."""
    result = correlation_calculator.detect_correlation_anomalies(
        data=sample_data[['indicator1', 'indicator2', 'price']],
        window_size=20,
        threshold=2.0
    )
    assert isinstance(result, dict)
    assert 'anomalies' in result
    assert 'scores' in result

def test_generate_correlation_report(correlation_calculator, sample_data):
    """Test correlation report generation."""
    result = correlation_calculator.generate_correlation_report(
        data=sample_data[['indicator1', 'indicator2', 'price']]
    )
    assert isinstance(result, dict)
    assert 'summary' in result
    assert 'correlations' in result
    assert 'significance' in result

def test_visualize_correlation(correlation_calculator, sample_data):
    """Test correlation visualization."""
    fig = correlation_calculator.visualize_correlation(
        data1=sample_data['indicator1'],
        data2=sample_data['price']
    )
    assert isinstance(fig, Figure)

def test_visualize_correlation_matrix(correlation_calculator, sample_data):
    """Test correlation matrix visualization."""
    fig = correlation_calculator.visualize_correlation_matrix(
        data=sample_data[['indicator1', 'indicator2', 'price']]
    )
    assert isinstance(fig, Figure) 