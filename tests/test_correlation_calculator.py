import pytest
import pandas as pd
import numpy as np
from correlation_calculator import CorrelationCalculator, calculate_correlation_indicator_vs_future_price, _calculate_correlations_for_single_indicator, process_correlations
from datetime import datetime, timedelta
import time
from functools import wraps
import threading
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from conftest import assert_timeout, skip_if_missing_dependency
from typing import Dict, List, Any, Optional, Tuple
import sqlite_manager

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
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
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

@pytest.fixture
def sample_data() -> pd.DataFrame:
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(200, 300, 100),
        'low': np.random.uniform(50, 100, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000, 5000, 100),
        'RSI_14': np.random.uniform(0, 100, 100),  # Sample indicator
        'MACD_12_26_9': np.random.uniform(-10, 10, 100)  # Sample indicator
    })
    return data

@pytest.fixture
def sample_indicator_configs() -> List[Dict[str, Any]]:
    return [
        {
            'config_id': 1,
            'indicator_name': 'RSI',
            'params': {'period': 14}
        },
        {
            'config_id': 2,
            'indicator_name': 'MACD',
            'params': {'fast': 12, 'slow': 26, 'signal': 9}
        }
    ]

def test_calculate_correlation_valid(sample_data):
    """Test correlation calculation with valid data."""
    result = calculate_correlation_indicator_vs_future_price(
        data=sample_data,
        indicator_col='RSI_14',
        lag=1
    )
    assert isinstance(result, (float, type(None)))
    if result is not None:
        assert -1 <= result <= 1

def test_calculate_correlation_missing_column(sample_data):
    """Test correlation calculation with missing column."""
    result = calculate_correlation_indicator_vs_future_price(
        data=sample_data,
        indicator_col='nonexistent',
        lag=1
    )
    assert result is None

def test_calculate_correlation_invalid_lag(sample_data):
    """Test correlation calculation with invalid lag."""
    result = calculate_correlation_indicator_vs_future_price(
        data=sample_data,
        indicator_col='RSI_14',
        lag=0
    )
    assert result is None

def test_calculate_correlation_all_nan(sample_data):
    """Test correlation calculation with all NaN values."""
    data = sample_data.copy()
    data['RSI_14'] = np.nan
    result = calculate_correlation_indicator_vs_future_price(
        data=data,
        indicator_col='RSI_14',
        lag=1
    )
    assert pd.isna(result)

def test_calculate_correlation_constant_value(sample_data):
    """Test correlation calculation with constant value."""
    data = sample_data.copy()
    data['RSI_14'] = 50  # Constant value
    result = calculate_correlation_indicator_vs_future_price(
        data=data,
        indicator_col='RSI_14',
        lag=1
    )
    assert pd.isna(result)

def test_calculate_correlations_for_single_indicator(sample_data):
    """Test calculating correlations for single indicator."""
    indicator_col = 'RSI_14'
    indicator_series = sample_data[indicator_col]
    shifted_closes = {
        lag: sample_data['close'].shift(-lag)
        for lag in range(1, 4)
    }
    
    results = _calculate_correlations_for_single_indicator(
        indicator_col_name=indicator_col,
        indicator_series=indicator_series,
        shifted_closes_future=shifted_closes,
        max_lag=3,
        symbol_id=1,
        timeframe_id=1,
        config_id=1
    )
    
    assert isinstance(results, list)
    assert len(results) == 3  # One result per lag
    
    for result in results:
        assert len(result) == 5  # (symbol_id, timeframe_id, config_id, lag, correlation)
        assert isinstance(result[0], int)  # symbol_id
        assert isinstance(result[1], int)  # timeframe_id
        assert isinstance(result[2], int)  # config_id
        assert isinstance(result[3], int)  # lag
        assert isinstance(result[4], (float, type(None)))  # correlation

def test_calculate_correlations_all_nan(sample_data):
    """Test calculating correlations with all NaN values."""
    indicator_col = 'RSI_14'
    indicator_series = pd.Series(np.nan, index=sample_data.index)
    shifted_closes = {
        lag: sample_data['close'].shift(-lag)
        for lag in range(1, 4)
    }
    
    results = _calculate_correlations_for_single_indicator(
        indicator_col_name=indicator_col,
        indicator_series=indicator_series,
        shifted_closes_future=shifted_closes,
        max_lag=3,
        symbol_id=1,
        timeframe_id=1,
        config_id=1
    )
    
    assert len(results) == 3
    assert all(result[4] is None for result in results)

def test_process_correlations_valid(
    sample_indicator_configs,
    tmp_path
):
    """Test processing correlations with valid data."""
    # Generate larger sample data (at least 223 rows)
    n_rows = 250
    dates = pd.date_range(start='2024-01-01', periods=n_rows, freq='H')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(100, 200, n_rows),
        'high': np.random.uniform(200, 300, n_rows),
        'low': np.random.uniform(50, 100, n_rows),
        'close': np.random.uniform(100, 200, n_rows),
        'volume': np.random.uniform(1000, 5000, n_rows),
        'RSI_1': np.random.uniform(0, 100, n_rows),  # config_id=1
        'MACD_2': np.random.uniform(-10, 10, n_rows)  # config_id=2
    })
    # Create test database
    db_path = str(tmp_path / "test.db")
    # Initialize DB schema
    sqlite_manager.initialize_database(db_path)
    # Insert required symbol, timeframe, and indicator configs
    conn = sqlite_manager.create_connection(db_path)
    symbol_id = sqlite_manager._get_or_create_id(conn, 'symbols', 'symbol', 'BTCUSDT')
    timeframe_id = sqlite_manager._get_or_create_id(conn, 'timeframes', 'timeframe', '1h')
    for cfg in sample_indicator_configs:
        sqlite_manager.get_or_create_indicator_config_id(conn, cfg['indicator_name'], cfg['params'])
    conn.close()
    # Mock display progress function
    mock_display = lambda *args, **kwargs: None
    # Mock periodic report function
    mock_periodic_report = lambda *args, **kwargs: None
    result = process_correlations(
        data=sample_data,
        db_path=db_path,
        symbol_id=1,
        timeframe_id=1,
        indicator_configs_processed=sample_indicator_configs,
        max_lag=3,
        analysis_start_time_global=time.time(),
        total_analysis_steps_global=10,
        current_step_base=1,
        total_steps_in_phase=5,
        display_progress_func=mock_display,
        periodic_report_func=mock_periodic_report
    )
    assert isinstance(result, bool)
    assert result is True

def test_process_correlations_invalid_data(
    sample_indicator_configs,
    tmp_path
):
    """Test processing correlations with invalid data."""
    # Create empty DataFrame
    data = pd.DataFrame()
    
    # Create test database
    db_path = str(tmp_path / "test.db")
    
    # Mock display progress function
    mock_display = lambda *args, **kwargs: None
    
    # Mock periodic report function
    mock_periodic_report = lambda *args, **kwargs: None
    
    result = process_correlations(
        data=data,
        db_path=db_path,
        symbol_id=1,
        timeframe_id=1,
        indicator_configs_processed=sample_indicator_configs,
        max_lag=3,
        analysis_start_time_global=time.time(),
        total_analysis_steps_global=10,
        current_step_base=1,
        total_steps_in_phase=5,
        display_progress_func=mock_display,
        periodic_report_func=mock_periodic_report
    )
    
    assert isinstance(result, bool)
    assert result is False

def test_correlation_calculator_init(correlation_calculator):
    """Test CorrelationCalculator initialization."""
    assert isinstance(correlation_calculator, CorrelationCalculator)

def test_calculate_correlation(correlation_calculator, sample_data):
    """Test basic correlation calculation."""
    result = correlation_calculator.calculate_correlation(
        data1=sample_data['RSI_14'],
        data2=sample_data['close'],
        method='pearson'
    )
    assert isinstance(result, float)
    assert -1 <= result <= 1

def test_calculate_rolling_correlation(correlation_calculator, sample_data):
    """Test rolling correlation calculation."""
    result = correlation_calculator.calculate_rolling_correlation(
        data1=sample_data['RSI_14'],
        data2=sample_data['close'],
        window=20
    )
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_data)
    assert result.notna().any()

def test_calculate_correlation_matrix(correlation_calculator, sample_data):
    """Test correlation matrix calculation."""
    result = correlation_calculator.calculate_correlation_matrix(
        data=sample_data[['RSI_14', 'MACD_12_26_9', 'close']]
    )
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 3)
    assert result.index.equals(result.columns)

def test_test_correlation_significance(correlation_calculator, sample_data):
    """Test correlation significance testing."""
    result = correlation_calculator.test_correlation_significance(
        data1=sample_data['RSI_14'],
        data2=sample_data['close']
    )
    assert isinstance(result, dict)
    assert 'correlation' in result
    assert 'p_value' in result
    assert 'significant' in result

def test_plot_correlation_heatmap(correlation_calculator, sample_data):
    """Test correlation heatmap plotting."""
    fig = correlation_calculator.plot_correlation_heatmap(
        data=sample_data[['RSI_14', 'MACD_12_26_9', 'close']]
    )
    assert isinstance(fig, Figure)

def test_plot_correlation_scatter(correlation_calculator, sample_data):
    """Test correlation scatter plot."""
    fig = correlation_calculator.plot_correlation_scatter(
        data1=sample_data['RSI_14'],
        data2=sample_data['close']
    )
    assert isinstance(fig, Figure)

def test_decompose_correlation(correlation_calculator, sample_data):
    """Test correlation decomposition."""
    result = correlation_calculator.decompose_correlation(
        data=sample_data[['RSI_14', 'MACD_12_26_9', 'close']],
        n_components=2
    )
    assert isinstance(result, dict)
    assert 'components' in result
    assert 'explained_variance' in result

def test_cluster_correlations(correlation_calculator, sample_data):
    """Test correlation clustering."""
    result = correlation_calculator.cluster_correlations(
        data=sample_data[['RSI_14', 'MACD_12_26_9', 'close']],
        n_clusters=2
    )
    assert isinstance(result, dict)
    assert 'labels' in result
    assert 'centers' in result
    assert 'silhouette_score' in result

def test_analyze_correlation_stability(correlation_calculator, sample_data):
    """Test correlation stability analysis."""
    result = correlation_calculator.analyze_correlation_stability(
        data=sample_data[['RSI_14', 'MACD_12_26_9', 'close']],
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
        data=sample_data[['RSI_14', 'MACD_12_26_9', 'close']],
        n_regimes=2
    )
    assert isinstance(result, dict)
    assert 'regime_labels' in result
    assert 'regime_characteristics' in result

def test_analyze_correlation_network(correlation_calculator, sample_data):
    """Test correlation network analysis."""
    result = correlation_calculator.analyze_correlation_network(
        data=sample_data[['RSI_14', 'MACD_12_26_9', 'close']],
        threshold=0.5
    )
    assert isinstance(result, dict)
    assert 'network' in result
    assert 'centrality' in result

def test_detect_correlation_anomalies(correlation_calculator, sample_data):
    """Test correlation anomaly detection."""
    result = correlation_calculator.detect_correlation_anomalies(
        data=sample_data[['RSI_14', 'MACD_12_26_9', 'close']],
        window_size=20,
        threshold=2.0
    )
    assert isinstance(result, dict)
    assert 'anomalies' in result
    assert 'scores' in result

def test_generate_correlation_report(correlation_calculator, sample_data):
    """Test correlation report generation."""
    result = correlation_calculator.generate_correlation_report(
        data=sample_data[['RSI_14', 'MACD_12_26_9', 'close']]
    )
    assert isinstance(result, dict)
    assert 'summary' in result
    assert 'correlations' in result
    assert 'significance' in result

def test_visualize_correlation(correlation_calculator, sample_data):
    """Test correlation visualization."""
    fig = correlation_calculator.visualize_correlation(
        data1=sample_data['RSI_14'],
        data2=sample_data['close']
    )
    assert isinstance(fig, Figure)

def test_visualize_correlation_matrix(correlation_calculator, sample_data):
    """Test correlation matrix visualization."""
    fig = correlation_calculator.visualize_correlation_matrix(
        data=sample_data[['RSI_14', 'MACD_12_26_9', 'close']]
    )
    assert isinstance(fig, Figure) 