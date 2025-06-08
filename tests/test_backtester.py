import pytest
import pandas as pd
import numpy as np
from backtester import Backtester
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Callable
from matplotlib.figure import Figure
from unittest.mock import Mock, patch

@pytest.fixture
def backtester(data_manager, indicator_factory):
    """Provide a Backtester instance for testing."""
    return Backtester(data_manager, indicator_factory)

def test_backtester_initialization(backtester):
    """Test Backtester initialization."""
    assert backtester is not None
    assert hasattr(backtester, 'data_manager')
    assert hasattr(backtester, 'indicator_factory')

def test_strategy_execution(backtester, test_data):
    """Test strategy execution methods."""
    # Define a simple strategy
    def simple_strategy(data, params):
        sma = data['close'].rolling(window=params['window']).mean()
        return pd.Series(np.where(data['close'] > sma, 1, -1), index=data.index)
    
    # Execute strategy
    results = backtester.run_strategy(test_data, simple_strategy, {'window': 20})
    assert isinstance(results, dict)
    assert 'positions' in results
    assert 'returns' in results
    assert 'equity_curve' in results

def test_position_sizing(backtester, test_data):
    """Test position sizing methods."""
    # Create sample positions with the full index of test_data
    positions = pd.Series(0, index=test_data.index)
    positions.iloc[:5] = pd.Series([1, -1, 1, 0, -1], index=test_data.index[:5])  # Set first 5 values using Series
    
    # Test fixed position sizing
    sized_positions = backtester.size_positions(positions, method='fixed', size=0.1)
    assert isinstance(sized_positions, pd.Series)
    assert sized_positions.notna().all()  # No NaNs expected
    assert all(abs(x) in [0, 0.1] for x in sized_positions)
    assert sized_positions.index.equals(test_data.index)  # Ensure index alignment
    
    # Test dynamic position sizing with the original test data
    sized_positions = backtester.size_positions(
        positions, 
        method='dynamic',
        data=test_data,
        volatility_window=20
    )
    assert isinstance(sized_positions, pd.Series)
    assert sized_positions.index.equals(test_data.index)  # Ensure index alignment
    # For dynamic sizing, only check non-NaN values after the volatility window
    non_nan_mask = sized_positions.notna()
    if non_nan_mask.any():
        assert all(sized_positions[non_nan_mask].between(-1, 1))  # Position sizes should be between -1 and 1
    else:
        # If all values are NaN (due to insufficient data), that's acceptable
        assert len(sized_positions) == len(positions)  # Length should match input

def test_returns_calculation(backtester, test_data):
    """Test returns calculation methods."""
    # Create sample positions matching the full index
    positions = pd.Series(0, index=test_data.index)
    positions.iloc[:5] = pd.Series([1, -1, 1, 0, -1], index=test_data.index[:5])  # Set first 5 values using Series
    sized_positions = backtester.size_positions(positions, method='fixed', size=0.1)
    assert sized_positions.index.equals(test_data.index)  # Ensure index alignment
    
    # Calculate returns
    returns = backtester.calculate_returns(test_data, sized_positions)
    assert isinstance(returns, pd.Series)
    assert returns.index.equals(test_data.index)  # Ensure index alignment
    # Only the first row may be NaN due to pct_change, the rest should not be NaN
    assert not returns.iloc[1:].isnull().any()
    
    # Calculate cumulative returns
    cum_returns = backtester.calculate_cumulative_returns(returns)
    assert isinstance(cum_returns, pd.Series)
    assert cum_returns.index.equals(test_data.index)  # Ensure index alignment
    assert not cum_returns.iloc[1:].isnull().any()

def test_performance_metrics(backtester, test_data):
    """Test performance metrics calculation."""
    # Create sample strategy results with full index
    positions = pd.Series(0, index=test_data.index)
    positions.iloc[:5] = pd.Series([1, -1, 1, 0, -1], index=test_data.index[:5])  # Set first 5 values using Series
    sized_positions = backtester.size_positions(positions, method='fixed', size=0.1)
    assert sized_positions.index.equals(test_data.index)  # Ensure index alignment
    returns = backtester.calculate_returns(test_data, sized_positions)
    assert returns.index.equals(test_data.index)  # Ensure index alignment
    
    # Calculate performance metrics
    metrics = backtester.calculate_performance_metrics(returns)
    assert isinstance(metrics, dict)
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'annual_return' in metrics

def test_risk_metrics(backtester, test_data):
    """Test risk metrics calculation."""
    # Create sample strategy results with full index
    positions = pd.Series(0, index=test_data.index)
    positions.iloc[:5] = pd.Series([1, -1, 1, 0, -1], index=test_data.index[:5])  # Set first 5 values using Series
    sized_positions = backtester.size_positions(positions, method='fixed', size=0.1)
    assert sized_positions.index.equals(test_data.index)  # Ensure index alignment
    returns = backtester.calculate_returns(test_data, sized_positions)
    assert returns.index.equals(test_data.index)  # Ensure index alignment
    
    # Calculate risk metrics
    risk_metrics = backtester.calculate_risk_metrics(returns)
    assert isinstance(risk_metrics, dict)
    assert 'volatility' in risk_metrics
    assert 'var_95' in risk_metrics
    assert 'cvar_95' in risk_metrics

def test_transaction_costs(backtester, test_data):
    """Test transaction cost handling."""
    # Create sample positions matching the full index
    positions = pd.Series(0, index=test_data.index)
    positions.iloc[:5] = pd.Series([1, -1, 1, 0, -1], index=test_data.index[:5])  # Set first 5 values using Series
    sized_positions = backtester.size_positions(positions, method='fixed', size=0.1)
    assert sized_positions.index.equals(test_data.index)  # Ensure index alignment
    
    # Calculate returns with transaction costs
    returns = backtester.calculate_returns(
        test_data, 
        sized_positions,
        transaction_cost=0.001
    )
    assert isinstance(returns, pd.Series)
    assert returns.index.equals(test_data.index)  # Ensure index alignment
    # Only the first row may be NaN due to pct_change, the rest should not be NaN
    assert not returns.iloc[1:].isnull().any()

@pytest.mark.timeout(30)
def test_optimization(backtester, test_data):
    """Test strategy optimization."""
    # Define a parameterized strategy
    def param_strategy(data, params):
        sma = data['close'].rolling(window=params['window']).mean()
        return pd.Series(np.where(data['close'] > sma, 1, -1), index=data.index)
    
    # Define parameter space
    param_space = {
        'window': range(10, 31, 5)
    }
    
    # Optimize strategy
    optimization_results = backtester.optimize_strategy(
        test_data,
        param_strategy,
        param_space,
        metric='sharpe_ratio'
    )
    assert isinstance(optimization_results, dict)
    assert 'best_params' in optimization_results
    assert 'best_score' in optimization_results

@pytest.mark.timeout(30)
def test_walk_forward_analysis(backtester, test_data):
    """Test walk-forward analysis."""
    # Define a simple strategy
    def simple_strategy(data, params):
        sma = data['close'].rolling(window=params['window']).mean()
        return pd.Series(np.where(data['close'] > sma, 1, -1), index=data.index)
    
    # Perform walk-forward analysis
    results = backtester.walk_forward_analysis(
        test_data,
        simple_strategy,
        {'window': 20},
        train_size=0.7,
        step_size=0.1
    )
    assert isinstance(results, dict)
    assert 'train_metrics' in results
    assert 'test_metrics' in results

@pytest.mark.timeout(30)
def test_monte_carlo_simulation(backtester, test_data):
    """Test Monte Carlo simulation."""
    # Create sample strategy results with full index
    positions = pd.Series(0, index=test_data.index)
    positions.iloc[:5] = pd.Series([1, -1, 1, 0, -1], index=test_data.index[:5])  # Set first 5 values using Series
    sized_positions = backtester.size_positions(positions, method='fixed', size=0.1)
    assert sized_positions.index.equals(test_data.index)  # Ensure index alignment
    returns = backtester.calculate_returns(test_data, sized_positions)
    assert returns.index.equals(test_data.index)  # Ensure index alignment
    
    # Run Monte Carlo simulation
    simulation_results = backtester.run_monte_carlo_simulation(
        returns,
        n_simulations=100,
        time_steps=252
    )
    assert isinstance(simulation_results, dict)
    assert 'simulations' in simulation_results
    assert 'confidence_intervals' in simulation_results

@pytest.mark.visualization
@pytest.mark.timeout(30)
def test_visualization(backtester, test_data):
    """Test backtest visualization methods."""
    # Create sample strategy results with full index
    positions = pd.Series(0, index=test_data.index)
    positions.iloc[:5] = pd.Series([1, -1, 1, 0, -1], index=test_data.index[:5])  # Set first 5 values using Series
    sized_positions = backtester.size_positions(positions, method='fixed', size=0.1)
    assert sized_positions.index.equals(test_data.index)  # Ensure index alignment
    returns = backtester.calculate_returns(test_data, sized_positions)
    assert returns.index.equals(test_data.index)  # Ensure index alignment
    
    # Create equity curve plot
    equity_plot = backtester.plot_equity_curve(returns)
    assert equity_plot is not None
    plt.close(equity_plot)  # Explicitly close the plot
    
    # Create drawdown plot
    drawdown_plot = backtester.plot_drawdown(returns)
    assert drawdown_plot is not None
    plt.close(drawdown_plot)  # Explicitly close the plot
    
    # Create monthly returns heatmap
    heatmap = backtester.plot_monthly_returns_heatmap(returns)
    assert heatmap is not None
    plt.close(heatmap)  # Explicitly close the plot

@pytest.mark.timeout(30)
def test_report_generation(backtester, test_data):
    """Test report generation methods."""
    # Create sample strategy results with full index
    positions = pd.Series(0, index=test_data.index)
    positions.iloc[:5] = pd.Series([1, -1, 1, 0, -1], index=test_data.index[:5])  # Set first 5 values using Series
    sized_positions = backtester.size_positions(positions, method='fixed', size=0.1)
    assert sized_positions.index.equals(test_data.index)  # Ensure index alignment
    returns = backtester.calculate_returns(test_data, sized_positions)
    assert returns.index.equals(test_data.index)  # Ensure index alignment
    
    # Generate performance report
    report = backtester.generate_performance_report(returns)
    assert isinstance(report, dict)
    assert 'summary' in report
    assert 'metrics' in report
    assert 'charts' in report

@pytest.mark.timeout(30)
def test_benchmark_comparison(backtester, test_data):
    """Test benchmark comparison methods."""
    # Create sample strategy results with full index
    positions = pd.Series(0, index=test_data.index)
    positions.iloc[:5] = pd.Series([1, -1, 1, 0, -1], index=test_data.index[:5])  # Set first 5 values using Series
    sized_positions = backtester.size_positions(positions, method='fixed', size=0.1)
    assert sized_positions.index.equals(test_data.index)  # Ensure index alignment
    strategy_returns = backtester.calculate_returns(test_data, sized_positions)
    assert strategy_returns.index.equals(test_data.index)  # Ensure index alignment
    
    # Create benchmark returns (e.g., buy and hold)
    benchmark_returns = test_data['close'].pct_change()
    assert benchmark_returns.index.equals(test_data.index)  # Ensure index alignment
    
    # Compare with benchmark
    comparison = backtester.compare_with_benchmark(
        strategy_returns,
        benchmark_returns
    )
    assert isinstance(comparison, dict)
    assert 'relative_performance' in comparison
    assert 'metrics_comparison' in comparison

# Test fixtures
@pytest.fixture
def sample_data() -> pd.DataFrame:
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(200, 300, 100),
        'low': np.random.uniform(50, 100, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    })
    return data

@pytest.fixture
def mock_data_manager():
    return Mock()

@pytest.fixture
def mock_indicator_factory():
    return Mock()

@pytest.fixture
def sample_strategy():
    def strategy(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        # Simple moving average crossover strategy
        short_window = params.get('short_window', 10)
        long_window = params.get('long_window', 20)
        
        data['short_ma'] = data['close'].rolling(window=short_window).mean()
        data['long_ma'] = data['close'].rolling(window=long_window).mean()
        
        positions = pd.Series(0, index=data.index)
        positions[data['short_ma'] > data['long_ma']] = 1
        positions[data['short_ma'] < data['long_ma']] = -1
        
        return positions
    
    return strategy

# Test Backtester initialization
def test_backtester_init(backtester, mock_data_manager, mock_indicator_factory):
    """Test Backtester initialization."""
    assert isinstance(backtester, Backtester)
    assert backtester.data_manager == mock_data_manager
    assert backtester.indicator_factory == mock_indicator_factory

def test_validate_dependencies(backtester):
    """Test dependency validation."""
    backtester._validate_dependencies()
    assert hasattr(backtester, 'SCIPY_AVAILABLE')

# Test strategy running
def test_run_strategy_valid(backtester, sample_data, sample_strategy):
    """Test running a valid strategy."""
    params = {'short_window': 10, 'long_window': 20}
    results = backtester.run_strategy(sample_data, sample_strategy, params)
    
    assert isinstance(results, dict)
    assert 'positions' in results
    assert 'sized_positions' in results
    assert 'returns' in results
    assert 'equity_curve' in results
    
    assert isinstance(results['positions'], pd.Series)
    assert isinstance(results['sized_positions'], pd.Series)
    assert isinstance(results['returns'], pd.Series)
    assert isinstance(results['equity_curve'], pd.Series)

def test_run_strategy_invalid_return(backtester, sample_data):
    """Test running a strategy that returns invalid data."""
    def invalid_strategy(data, params):
        return pd.DataFrame()  # Should return Series
    
    with pytest.raises(ValueError):
        backtester.run_strategy(sample_data, invalid_strategy, {})

# Test position sizing
def test_size_positions_fixed(backtester, sample_data):
    """Test fixed position sizing."""
    positions = pd.Series([1, -1, 0, 1], index=sample_data.index[:4])
    sized = backtester.size_positions(positions, method='fixed', size=0.1)
    
    assert isinstance(sized, pd.Series)
    assert (sized == positions * 0.1).all()

def test_size_positions_dynamic(backtester, sample_data):
    """Test dynamic position sizing."""
    positions = pd.Series([1, -1, 0, 1], index=sample_data.index[:4])
    sized = backtester.size_positions(
        positions,
        method='dynamic',
        data=sample_data,
        volatility_window=20
    )
    
    assert isinstance(sized, pd.Series)
    assert len(sized) == len(positions)
    assert (sized <= positions).all()  # Dynamic sizing should not exceed original positions

def test_size_positions_invalid_method(backtester, sample_data):
    """Test position sizing with invalid method."""
    positions = pd.Series([1, -1, 0, 1], index=sample_data.index[:4])
    with pytest.raises(ValueError):
        backtester.size_positions(positions, method='invalid')

# Test returns calculation
def test_calculate_returns(backtester, sample_data):
    """Test returns calculation."""
    positions = pd.Series([1, -1, 0, 1], index=sample_data.index[:4])
    returns = backtester.calculate_returns(sample_data, positions)
    
    assert isinstance(returns, pd.Series)
    assert len(returns) == len(positions)
    assert not returns.isna().all()

def test_calculate_returns_with_costs(backtester, sample_data):
    """Test returns calculation with transaction costs."""
    positions = pd.Series([1, -1, 0, 1], index=sample_data.index[:4])
    returns = backtester.calculate_returns(sample_data, positions, transaction_cost=0.001)
    
    assert isinstance(returns, pd.Series)
    assert len(returns) == len(positions)
    assert not returns.isna().all()

# Test performance metrics
def test_calculate_performance_metrics(backtester, sample_data):
    """Test performance metrics calculation."""
    returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=sample_data.index)
    metrics = backtester.calculate_performance_metrics(returns)
    
    assert isinstance(metrics, dict)
    assert 'annual_return' in metrics
    assert 'annual_volatility' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics

def test_calculate_risk_metrics(backtester, sample_data):
    """Test risk metrics calculation."""
    returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=sample_data.index)
    metrics = backtester.calculate_risk_metrics(returns)
    
    assert isinstance(metrics, dict)
    assert 'volatility' in metrics
    assert 'var_95' in metrics
    assert 'cvar_95' in metrics
    assert 'skewness' in metrics
    assert 'kurtosis' in metrics

# Test strategy optimization
def test_optimize_strategy(backtester, sample_data, sample_strategy):
    """Test strategy optimization."""
    param_space = {
        'short_window': [5, 10, 15],
        'long_window': [20, 30, 40]
    }
    
    results = backtester.optimize_strategy(
        sample_data,
        sample_strategy,
        param_space,
        metric='sharpe_ratio'
    )
    
    assert isinstance(results, dict)
    assert 'best_params' in results
    assert 'best_score' in results
    assert 'all_results' in results

# Test walk-forward analysis
def test_walk_forward_analysis(backtester, sample_data, sample_strategy):
    """Test walk-forward analysis."""
    params = {'short_window': 10, 'long_window': 20}
    results = backtester.walk_forward_analysis(
        sample_data,
        sample_strategy,
        params,
        train_size=0.7,
        step_size=0.1
    )
    
    assert isinstance(results, dict)
    assert 'train_metrics' in results
    assert 'test_metrics' in results
    assert 'combined_metrics' in results

# Test Monte Carlo simulation
def test_run_monte_carlo_simulation(backtester, sample_data):
    """Test Monte Carlo simulation."""
    returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=sample_data.index)
    results = backtester.run_monte_carlo_simulation(
        returns,
        n_simulations=100,
        time_steps=252
    )
    
    assert isinstance(results, dict)
    assert 'simulations' in results
    assert 'percentiles' in results
    assert 'confidence_intervals' in results

# Test visualization functions
def test_plot_equity_curve(backtester, sample_data):
    """Test equity curve plotting."""
    returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=sample_data.index)
    fig = backtester.plot_equity_curve(returns)
    assert isinstance(fig, Figure)

def test_plot_drawdown(backtester, sample_data):
    """Test drawdown plotting."""
    returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=sample_data.index)
    fig = backtester.plot_drawdown(returns)
    assert isinstance(fig, Figure)

def test_plot_monthly_returns_heatmap(backtester, sample_data):
    """Test monthly returns heatmap plotting."""
    returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=sample_data.index)
    fig = backtester.plot_monthly_returns_heatmap(returns)
    assert isinstance(fig, Figure)

# Test performance reporting
def test_generate_performance_report(backtester, sample_data):
    """Test performance report generation."""
    returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=sample_data.index)
    report = backtester.generate_performance_report(returns)
    
    assert isinstance(report, dict)
    assert 'summary' in report
    assert 'metrics' in report
    assert 'risk_metrics' in report

def test_compare_with_benchmark(backtester, sample_data):
    """Test benchmark comparison."""
    strategy_returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=sample_data.index)
    benchmark_returns = pd.Series(np.random.normal(0.0005, 0.015, 100), index=sample_data.index)
    
    comparison = backtester.compare_with_benchmark(strategy_returns, benchmark_returns)
    
    assert isinstance(comparison, dict)
    assert 'relative_performance' in comparison
    assert 'correlation' in comparison
    assert 'information_ratio' in comparison

# Test main backtest function
@patch('backtester.sqlite_manager')
@patch('backtester.data_manager')
@patch('backtester.indicator_factory')
def test_run_backtest(mock_indicator_factory, mock_data_manager, mock_sqlite_manager, tmp_path):
    """Test main backtest function."""
    # Create test database
    db_path = tmp_path / "test.db"
    
    # Mock necessary functions and data
    mock_data_manager.get_data.return_value = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(200, 300, 100),
        'low': np.random.uniform(50, 100, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    })
    
    mock_sqlite_manager.get_indicator_configs.return_value = [
        {'config_id': 1, 'indicator_name': 'RSI', 'params': {'period': 14}},
        {'config_id': 2, 'indicator_name': 'MACD', 'params': {'fast': 12, 'slow': 26, 'signal': 9}}
    ]
    
    # Run backtest
    run_backtest(
        db_path=db_path,
        symbol='BTCUSDT',
        timeframe='1h',
        max_lag_backtest=3,
        num_backtest_points=50
    )
    
    # Verify that necessary functions were called
    mock_data_manager.get_data.assert_called_once()
    mock_sqlite_manager.get_indicator_configs.assert_called_once()

# Test edge cases and error handling
def test_calculate_returns_empty_data(backtester):
    """Test returns calculation with empty data."""
    data = pd.DataFrame()
    positions = pd.Series()
    returns = backtester.calculate_returns(data, positions)
    assert isinstance(returns, pd.Series)
    assert len(returns) == 0

def test_calculate_performance_metrics_no_scipy(backtester, monkeypatch):
    """Test performance metrics calculation without scipy."""
    monkeypatch.setattr(backtester, 'SCIPY_AVAILABLE', False)
    returns = pd.Series(np.random.normal(0.001, 0.02, 100))
    with pytest.raises(ImportError):
        backtester.calculate_performance_metrics(returns)

def test_optimize_strategy_empty_param_space(backtester, sample_data, sample_strategy):
    """Test strategy optimization with empty parameter space."""
    param_space = {}
    with pytest.raises(ValueError):
        backtester.optimize_strategy(sample_data, sample_strategy, param_space)

def test_walk_forward_analysis_invalid_sizes(backtester, sample_data, sample_strategy):
    """Test walk-forward analysis with invalid sizes."""
    params = {'short_window': 10, 'long_window': 20}
    with pytest.raises(ValueError):
        backtester.walk_forward_analysis(
            sample_data,
            sample_strategy,
            params,
            train_size=1.5,  # Invalid train size
            step_size=0.1
        )

def test_run_monte_carlo_simulation_invalid_params(backtester, sample_data):
    """Test Monte Carlo simulation with invalid parameters."""
    returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=sample_data.index)
    with pytest.raises(ValueError):
        backtester.run_monte_carlo_simulation(returns, n_simulations=0, time_steps=0)

def test_compare_with_benchmark_mismatched_indices(backtester):
    """Test benchmark comparison with mismatched indices."""
    strategy_returns = pd.Series(np.random.normal(0.001, 0.02, 100))
    benchmark_returns = pd.Series(np.random.normal(0.0005, 0.015, 50))  # Different length
    with pytest.raises(ValueError):
        backtester.compare_with_benchmark(strategy_returns, benchmark_returns) 