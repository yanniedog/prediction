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
from conftest import assert_timeout, skip_if_missing_dependency

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

def test_correlation_calculator_initialization(correlation_calculator):
    """Test CorrelationCalculator initialization."""
    assert correlation_calculator is not None

@pytest.mark.timeout(30)  # Explicit timeout for this test
def test_correlation_report_generation(correlation_calculator, test_data):
    """Test correlation report generation with timeout and dependency checks."""
    # Skip if visualization dependencies are not available
    skip_if_missing_dependency('matplotlib')
    skip_if_missing_dependency('seaborn')
    
    # Use a small subset of data for testing
    test_subset = test_data.iloc[:50].copy()
    
    # Generate report with timeout check
    report = assert_timeout(
        correlation_calculator.generate_correlation_report,
        test_subset,
        timeout_seconds=10
    )
    
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
    
    # Validate analysis section
    analysis = report['analysis']
    assert isinstance(analysis, dict)
    assert 'decomposition' in analysis
    assert 'stability' in analysis
    assert isinstance(analysis['decomposition'], dict)
    assert isinstance(analysis['stability'], dict)
    
    # Validate visualizations if available
    visualizations = report['visualizations']
    assert isinstance(visualizations, dict)
    if 'correlation_matrix_plot' in visualizations:
        assert isinstance(visualizations['correlation_matrix_plot'], Figure)
    if 'correlation_heatmap' in visualizations:
        assert isinstance(visualizations['correlation_heatmap'], Figure)

@pytest.mark.timeout(20)
def test_correlation_calculation(correlation_calculator, test_data):
    """Test basic correlation calculation with timeout."""
    # Use a small subset of data
    test_subset = test_data.iloc[:30].copy()
    
    # Calculate correlation with timeout check
    result = assert_timeout(
        correlation_calculator.calculate_correlation,
        test_subset['close'],
        test_subset['volume'],
        timeout_seconds=5
    )
    
    assert isinstance(result, (float, np.floating))
    assert -1 <= result <= 1

@pytest.mark.timeout(20)
def test_rolling_correlation(correlation_calculator, test_data):
    """Test rolling correlation calculation with timeout."""
    # Use a small subset of data
    test_subset = test_data.iloc[:40].copy()
    
    # Calculate rolling correlation with timeout check
    result = assert_timeout(
        correlation_calculator.calculate_rolling_correlation,
        test_subset['close'],
        test_subset['volume'],
        window=5,
        timeout_seconds=5
    )
    
    assert isinstance(result, pd.Series)
    assert not result.empty
    assert len(result) == len(test_subset)
    assert result.notna().any()  # Should have some non-NaN values
    assert all(-1 <= x <= 1 for x in result.dropna())

@pytest.mark.timeout(20)
def test_correlation_matrix(correlation_calculator, test_data):
    """Test correlation matrix calculation with timeout."""
    # Use a small subset of data
    test_subset = test_data.iloc[:30].copy()
    
    # Calculate correlation matrix with timeout check
    result = assert_timeout(
        correlation_calculator.calculate_correlation_matrix,
        test_subset,
        timeout_seconds=5
    )
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert result.shape[0] == result.shape[1]  # Square matrix
    assert result.index.equals(result.columns)  # Symmetric
    assert all(-1 <= x <= 1 for x in result.values.flatten())

@pytest.mark.timeout(20)
def test_correlation_anomalies(correlation_calculator, test_data):
    """Test correlation anomaly detection with timeout."""
    # Use a small subset of data
    test_subset = test_data.iloc[:40].copy()
    
    # Detect anomalies with timeout check
    result = assert_timeout(
        correlation_calculator.detect_correlation_anomalies,
        test_subset,
        threshold=0.8,
        timeout_seconds=5
    )
    
    assert isinstance(result, dict)
    assert 'anomaly_correlations' in result
    assert isinstance(result['anomaly_correlations'], list)
    if result['anomaly_correlations']:
        assert all(isinstance(x, pd.DataFrame) for x in result['anomaly_correlations'])

@pytest.mark.timeout(20)
def test_correlation_visualization(correlation_calculator, test_data):
    """Test correlation visualization with timeout and dependency check."""
    # Skip if visualization dependencies are not available
    skip_if_missing_dependency('matplotlib')
    skip_if_missing_dependency('seaborn')
    
    # Use a small subset of data
    test_subset = test_data.iloc[:30].copy()
    
    # Test both single correlation and matrix visualization
    # Single correlation visualization
    fig1 = assert_timeout(
        correlation_calculator.visualize_correlation,
        test_subset['close'],
        test_subset['volume'],
        timeout_seconds=5
    )
    assert isinstance(fig1, Figure)
    plt.close(fig1)
    
    # Matrix visualization
    fig2 = assert_timeout(
        correlation_calculator.visualize_correlation_matrix,
        test_subset,
        timeout_seconds=5
    )
    assert isinstance(fig2, Figure)
    plt.close(fig2)

def test_correlation_significance(correlation_calculator, test_data):
    """Test correlation significance testing with timeout."""
    # Use a small subset of data
    test_subset = test_data.iloc[:30].copy()
    
    # Test significance with timeout check
    result = assert_timeout(
        correlation_calculator.test_correlation_significance,
        test_subset['close'],
        test_subset['volume'],
        timeout_seconds=5
    )
    
    assert isinstance(result, dict)
    assert 'significant' in result
    assert 'p_value' in result
    assert 'correlation' in result
    assert isinstance(result['significant'], bool)
    assert isinstance(result['p_value'], float)
    assert isinstance(result['correlation'], float)
    assert 0 <= result['p_value'] <= 1
    assert -1 <= result['correlation'] <= 1

def test_correlation_stability(correlation_calculator, test_data):
    """Test correlation stability analysis with timeout."""
    # Use a small subset of data
    test_subset = test_data.iloc[:40].copy()
    
    # Create a DataFrame with the two series
    test_df = pd.DataFrame({
        'close': test_subset['close'],
        'volume': test_subset['volume']
    })
    
    # Analyze stability with timeout check
    result = assert_timeout(
        correlation_calculator.analyze_correlation_stability,
        test_df,  # Pass DataFrame instead of separate series
        window_size=5,
        timeout_seconds=5
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

def test_correlation_forecasting(correlation_calculator, test_data):
    """Test correlation forecasting methods."""
    # Create sample data
    data = pd.DataFrame({
        'close': test_data['close'],
        'volume': test_data['volume'],
        'returns': test_data['close'].pct_change()
    })
    
    # Forecast correlations
    forecast = correlation_calculator.forecast_correlations(
        data,
        forecast_horizon=5
    )
    assert isinstance(forecast, dict)
    assert 'forecast_matrix' in forecast
    assert 'confidence_intervals' in forecast

def test_correlation_regime_detection(correlation_calculator, test_data):
    """Test correlation regime detection."""
    # Create sample data
    data = pd.DataFrame({
        'close': test_data['close'],
        'volume': test_data['volume'],
        'returns': test_data['close'].pct_change()
    })
    
    # Detect correlation regimes
    regimes = correlation_calculator.detect_correlation_regimes(
        data,
        n_regimes=2
    )
    assert isinstance(regimes, dict)
    assert 'regime_labels' in regimes
    assert 'regime_correlations' in regimes
    assert 'transition_matrix' in regimes

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
    skip_if_missing_dependency('matplotlib')
    skip_if_missing_dependency('seaborn')
    
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