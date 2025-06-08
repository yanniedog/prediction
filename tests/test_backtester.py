import pytest
import pandas as pd
import numpy as np
from backtester import Backtester
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

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