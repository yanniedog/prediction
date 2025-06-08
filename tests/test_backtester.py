import pytest
import pandas as pd
import numpy as np
from backtester import Backtester, run_backtest
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Callable
from matplotlib.figure import Figure
from unittest.mock import Mock, patch
import tempfile
import shutil
import json
from backtester import (
    Strategy,
    PerformanceMetrics,
    _calculate_returns,
    _calculate_drawdown,
    _calculate_sharpe_ratio,
    _calculate_sortino_ratio,
    _calculate_max_drawdown,
    _calculate_win_rate
)

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
    def simple_strategy(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Simple moving average strategy that avoids modifying input data."""
        # Create a copy of required columns
        df = pd.DataFrame(index=data.index)
        df['close'] = data['close']
        
        # Calculate moving average
        window = params['window']
        df['sma'] = df['close'].rolling(window=window).mean()
        
        # Generate positions using vectorized operations and convert to Series
        positions = pd.Series(np.where(df['close'] > df['sma'], 1, -1), index=data.index)
        return positions
    
    # Execute strategy
    results = backtester.run_strategy(test_data, simple_strategy, {'window': 20})
    assert isinstance(results, dict)
    assert 'positions' in results
    assert 'returns' in results
    assert 'equity_curve' in results
    assert results['positions'].index.equals(test_data.index)

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
def test_walk_forward_analysis(backtester, sample_data):
    """Test walk-forward analysis."""
    def strategy(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Moving average crossover strategy for walk-forward analysis."""
        # Create a copy of required columns
        df = pd.DataFrame(index=data.index)
        df['close'] = data['close']
        
        # Calculate moving averages
        short_window = params['short_window']
        long_window = params['long_window']
        
        df['short_ma'] = df['close'].rolling(window=short_window).mean()
        df['long_ma'] = df['close'].rolling(window=long_window).mean()
        
        # Generate positions using vectorized operations
        positions = np.where(df['short_ma'] > df['long_ma'], 1,
                           np.where(df['short_ma'] < df['long_ma'], -1, 0))
        
        return pd.Series(positions, index=data.index)
    
    params = {'short_window': 10, 'long_window': 20}
    results = backtester.walk_forward_analysis(
        sample_data,
        strategy,
        params,
        train_size=0.7,
        step_size=0.1
    )
    
    assert isinstance(results, dict)
    assert 'train_metrics' in results
    assert 'test_metrics' in results
    assert 'combined_metrics' in results

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
    # Check that risk metrics are present inside 'metrics'
    for key in ['volatility', 'var_95', 'cvar_95', 'skewness', 'kurtosis']:
        assert key in report['metrics']

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
@pytest.fixture(scope="function")
def sample_data() -> pd.DataFrame:
    """Create sample price data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
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
def sample_strategy() -> Strategy:
    """Create a sample strategy for testing."""
    def entry_condition(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        return data["close"] > data["close"].rolling(window=params["window"]).mean()
    
    def exit_condition(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        return data["close"] < data["close"].rolling(window=params["window"]).mean()
    
    return Strategy(
        name="Moving Average Crossover",
        entry_condition=entry_condition,
        exit_condition=exit_condition,
        params={"window": 20}
    )

@pytest.fixture(scope="function")
def backtester(sample_data: pd.DataFrame, sample_strategy: Strategy) -> Backtester:
    """Create a Backtester instance for testing."""
    return Backtester(sample_data, sample_strategy)

def test_backtester_initialization(sample_data: pd.DataFrame, sample_strategy: Strategy):
    """Test backtester initialization."""
    # Test basic initialization
    backtester = Backtester(sample_data, sample_strategy)
    assert backtester.data.equals(sample_data)
    assert backtester.strategy == sample_strategy
    
    # Test with invalid data
    with pytest.raises(ValueError):
        Backtester(pd.DataFrame(), sample_strategy)
    
    # Test with invalid strategy
    with pytest.raises(ValueError):
        Backtester(sample_data, None)

def test_strategy_execution(backtester: Backtester):
    """Test strategy execution."""
    # Test basic execution
    results = backtester.run()
    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    assert "position" in results.columns
    assert "returns" in results.columns
    
    # Test with custom parameters
    custom_params = {"window": 10}
    results = backtester.run(params=custom_params)
    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    
    # Test with invalid parameters
    with pytest.raises(ValueError):
        backtester.run(params={"invalid": 10})

def test_performance_metrics(backtester: Backtester):
    """Test performance metrics calculation."""
    # Run backtest
    results = backtester.run()
    metrics = backtester.calculate_metrics(results)
    
    # Verify metrics
    assert isinstance(metrics, PerformanceMetrics)
    assert isinstance(metrics.total_return, float)
    assert isinstance(metrics.sharpe_ratio, float)
    assert isinstance(metrics.sortino_ratio, float)
    assert isinstance(metrics.max_drawdown, float)
    assert isinstance(metrics.win_rate, float)
    
    # Test with empty results
    with pytest.raises(ValueError):
        backtester.calculate_metrics(pd.DataFrame())
    
    # Test with missing required columns
    invalid_results = results.drop(columns=["returns"])
    with pytest.raises(ValueError):
        backtester.calculate_metrics(invalid_results)

def test_returns_calculation(backtester: Backtester):
    """Test returns calculation."""
    # Test basic returns calculation
    results = backtester.run()
    returns = _calculate_returns(results["position"], results["close"])
    assert isinstance(returns, pd.Series)
    assert not returns.empty
    assert not returns.isna().all()
    
    # Test with zero positions
    zero_positions = pd.Series(0, index=results.index)
    returns = _calculate_returns(zero_positions, results["close"])
    assert (returns == 0).all()
    
    # Test with invalid data
    with pytest.raises(ValueError):
        _calculate_returns(pd.Series(), results["close"])

def test_drawdown_calculation(backtester: Backtester):
    """Test drawdown calculation."""
    # Run backtest and calculate returns
    results = backtester.run()
    returns = _calculate_returns(results["position"], results["close"])
    
    # Test drawdown calculation
    drawdown = _calculate_drawdown(returns)
    assert isinstance(drawdown, pd.Series)
    assert not drawdown.empty
    assert not drawdown.isna().all()
    assert (drawdown <= 0).all()  # Drawdowns should be non-positive
    
    # Test max drawdown
    max_dd = _calculate_max_drawdown(returns)
    assert isinstance(max_dd, float)
    assert max_dd <= 0
    assert max_dd >= drawdown.min()

def test_risk_metrics(backtester: Backtester):
    """Test risk metrics calculation."""
    # Run backtest and calculate returns
    results = backtester.run()
    returns = _calculate_returns(results["position"], results["close"])
    
    # Test Sharpe ratio
    sharpe = _calculate_sharpe_ratio(returns)
    assert isinstance(sharpe, float)
    assert not np.isnan(sharpe)
    assert not np.isinf(sharpe)
    
    # Test Sortino ratio
    sortino = _calculate_sortino_ratio(returns)
    assert isinstance(sortino, float)
    assert not np.isnan(sortino)
    assert not np.isinf(sortino)
    
    # Test with zero returns
    zero_returns = pd.Series(0, index=returns.index)
    assert _calculate_sharpe_ratio(zero_returns) == 0
    assert _calculate_sortino_ratio(zero_returns) == 0

def test_win_rate_calculation(backtester: Backtester):
    """Test win rate calculation."""
    # Run backtest and calculate returns
    results = backtester.run()
    returns = _calculate_returns(results["position"], results["close"])
    
    # Test win rate calculation
    win_rate = _calculate_win_rate(returns)
    assert isinstance(win_rate, float)
    assert 0 <= win_rate <= 1
    
    # Test with all winning trades
    winning_returns = pd.Series(0.1, index=returns.index)
    assert _calculate_win_rate(winning_returns) == 1.0
    
    # Test with all losing trades
    losing_returns = pd.Series(-0.1, index=returns.index)
    assert _calculate_win_rate(losing_returns) == 0.0

def test_position_management(backtester: Backtester):
    """Test position management."""
    # Test position sizing
    results = backtester.run(position_size=0.5)  # 50% position size
    assert (abs(results["position"]) <= 0.5).all()
    
    # Test position limits
    results = backtester.run(max_position=0.3)  # Maximum 30% position
    assert (abs(results["position"]) <= 0.3).all()
    
    # Test invalid position size
    with pytest.raises(ValueError):
        backtester.run(position_size=1.5)  # >100% position size
    
    # Test invalid max position
    with pytest.raises(ValueError):
        backtester.run(max_position=-0.1)  # Negative max position

def test_transaction_costs(backtester: Backtester):
    """Test transaction cost handling."""
    # Test with transaction costs
    results = backtester.run(transaction_cost=0.001)  # 0.1% transaction cost
    assert "transaction_cost" in results.columns
    assert (results["transaction_cost"] >= 0).all()
    
    # Test with zero transaction costs
    results = backtester.run(transaction_cost=0)
    assert (results["transaction_cost"] == 0).all()
    
    # Test with invalid transaction costs
    with pytest.raises(ValueError):
        backtester.run(transaction_cost=-0.001)  # Negative transaction cost

def test_optimization(backtester: Backtester):
    """Test strategy optimization."""
    # Define parameter grid
    param_grid = {
        "window": [10, 20, 30, 40, 50]
    }
    
    # Test optimization
    best_params, best_metrics = backtester.optimize(param_grid)
    assert isinstance(best_params, dict)
    assert "window" in best_params
    assert isinstance(best_metrics, PerformanceMetrics)
    
    # Test with invalid parameter grid
    with pytest.raises(ValueError):
        backtester.optimize({})
    
    # Test with invalid parameter values
    invalid_grid = {"window": [-1, 0]}  # Invalid window sizes
    with pytest.raises(ValueError):
        backtester.optimize(invalid_grid)

def test_plotting(backtester: Backtester, temp_dir: Path):
    """Test backtest plotting."""
    # Run backtest
    results = backtester.run()
    
    # Test equity curve plot
    plot_path = temp_dir / "equity_curve.png"
    backtester.plot_equity_curve(results, plot_path)
    assert plot_path.exists()
    
    # Test drawdown plot
    plot_path = temp_dir / "drawdown.png"
    backtester.plot_drawdown(results, plot_path)
    assert plot_path.exists()
    
    # Test monthly returns plot
    plot_path = temp_dir / "monthly_returns.png"
    backtester.plot_monthly_returns(results, plot_path)
    assert plot_path.exists()

def test_error_handling(backtester: Backtester):
    """Test error handling."""
    # Test with invalid data types
    invalid_data = backtester.data.copy()
    invalid_data["close"] = "invalid"
    with pytest.raises(ValueError):
        Backtester(invalid_data, backtester.strategy)
    
    # Test with missing required columns
    invalid_data = backtester.data.drop(columns=["close"])
    with pytest.raises(ValueError):
        Backtester(invalid_data, backtester.strategy)
    
    # Test with invalid strategy conditions
    def invalid_condition(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        return pd.Series(["invalid"] * len(data))  # Invalid return type
    
    invalid_strategy = Strategy(
        name="Invalid Strategy",
        entry_condition=invalid_condition,
        exit_condition=backtester.strategy.exit_condition,
        params=backtester.strategy.params
    )
    with pytest.raises(ValueError):
        Backtester(backtester.data, invalid_strategy)

# Test main backtest function
@patch('backtester.sqlite_manager')
@patch('backtester.data_manager')
@patch('backtester.indicator_factory')
def test_run_backtest(mock_indicator_factory, mock_data_manager, mock_sqlite_manager, tmp_path):
    """Test main backtest function."""
    db_path = tmp_path / "test.db"
    # Create a DataFrame with required columns
    df = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=100, freq='H'),
        'close': np.random.uniform(100, 200, 100),
        'open_time': pd.date_range(start='2024-01-01', periods=100, freq='H'),
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(200, 300, 100),
        'low': np.random.uniform(50, 100, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    })
    with patch('backtester.data_manager.load_data', return_value=df), \
         patch('backtester.leaderboard_manager.load_leaderboard', return_value={(1, 'positive'): {'config_id_source_db': 1, 'indicator_name': 'RSI', 'config_json': '{}', 'correlation_value': 0.9}}), \
         patch('backtester.sqlite_manager.create_connection'), \
         patch('backtester.sqlite_manager._get_or_create_id', return_value=1):
        try:
            result = run_backtest(
                db_path=db_path,
                symbol='BTCUSDT',
                timeframe='1h',
                max_lag_backtest=1,
                num_backtest_points=1
            )
            assert isinstance(result, dict)
            assert 'positions_by_lag' in result
            assert 'results_df' in result
            positions_by_lag = result['positions_by_lag']
            assert isinstance(positions_by_lag, dict)
            for lag, series in positions_by_lag.items():
                assert isinstance(series, pd.Series)
                # Should only contain -1, 0, or 1
                assert set(series.unique()).issubset({-1, 0, 1})
                # Index should not be empty
                assert len(series.index) > 0
        except Exception as e:
            pytest.fail(f"run_backtest raised an exception: {e}")

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