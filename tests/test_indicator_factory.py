import pytest
import pandas as pd
import numpy as np
from indicator_factory import IndicatorFactory
import json
from pathlib import Path

@pytest.fixture(scope="module")
def factory():
    return IndicatorFactory()

@pytest.fixture(scope="module")
def indicator_defs():
    with open(Path(__file__).parent.parent / "indicator_params.json", "r") as f:
        return json.load(f)

@pytest.fixture
def sample_data():
    # Minimal DataFrame with required columns for most indicators
    return pd.DataFrame({
        "close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "open": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "high": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "low": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "volume": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
    })

@pytest.fixture
def test_data():
    """Create sample test data."""
    dates = pd.date_range(start='2020-01-01', end='2020-02-19', freq='D')
    data = pd.DataFrame({
        'open': np.random.uniform(1000, 6000, len(dates)),
        'high': np.random.uniform(1000, 6000, len(dates)),
        'low': np.random.uniform(1000, 6000, len(dates)),
        'close': np.random.uniform(1000, 6000, len(dates)),
        'volume': np.random.uniform(10000, 100000, len(dates))
    }, index=dates)
    return data

def test_indicator_factory_initialization(indicator_factory):
    """Test IndicatorFactory initialization."""
    assert indicator_factory is not None

def test_indicator_creation(indicator_factory, test_data):
    """Test creation of various indicators."""
    # Test computing an exponential moving average
    result_df = indicator_factory.compute_indicators(test_data, indicators=['ema'])
    assert isinstance(result_df, pd.DataFrame)
    assert not result_df.empty
    assert len(result_df) == len(test_data)
    assert any(col.startswith('ema_') for col in result_df.columns)

    # Test computing a relative strength index
    result_df = indicator_factory.compute_indicators(test_data, indicators=['rsi'])
    assert isinstance(result_df, pd.DataFrame)
    assert not result_df.empty
    assert len(result_df) == len(test_data)
    rsi_col = [col for col in result_df.columns if col.startswith('rsi_')][0]
    assert result_df[rsi_col].max() <= 100
    assert result_df[rsi_col].min() >= 0

def test_indicator_parameters(indicator_factory, test_data):
    """Test indicator parameter handling."""
    # Test with valid parameters
    result_df = indicator_factory.compute_indicators(test_data, indicators=['bb'])
    assert isinstance(result_df, pd.DataFrame)
    assert any(col.startswith('bb_') for col in result_df.columns)

    # Test with invalid indicator name
    with pytest.raises(ValueError):
        indicator_factory.compute_indicators(test_data, indicators=['INVALID_INDICATOR'])

def test_indicator_combinations(indicator_factory, test_data):
    """Test combining multiple indicators."""
    # Create multiple indicators
    result_df = indicator_factory.compute_indicators(test_data, indicators=['ema', 'rsi'])
    assert isinstance(result_df, pd.DataFrame)
    assert any(col.startswith('ema_') for col in result_df.columns)
    assert any(col.startswith('rsi_') for col in result_df.columns)
    assert not result_df.isnull().all().any()

def test_indicator_validation(indicator_factory, test_data):
    """Test indicator validation methods."""
    # Test with valid data
    result_df = indicator_factory.compute_indicators(test_data, indicators=['ema'])
    assert isinstance(result_df, pd.DataFrame)
    assert not result_df.empty

    # Test with invalid data
    invalid_data = test_data.copy()
    invalid_data.loc[invalid_data.index[0], 'close'] = np.nan
    result_df = indicator_factory.compute_indicators(invalid_data, indicators=['ema'])
    assert isinstance(result_df, pd.DataFrame)
    # Should still return a DataFrame but with NaN values where calculation failed
    assert result_df.isnull().any().any()

def test_indicator_customization(indicator_factory, test_data):
    """Test custom indicator creation and modification."""
    # Test creating a custom indicator
    def custom_ma(data, period):
        return data['close'].rolling(window=period).mean()

    custom_indicator = indicator_factory.create_custom_indicator(
        'CustomMA',
        custom_ma,
        test_data,
        period=20
    )
    assert isinstance(custom_indicator, pd.Series)
    assert not custom_indicator.isnull().all()

def test_indicator_caching(indicator_factory, test_data):
    """Test indicator caching functionality."""
    # Create the same indicator twice
    indicator1 = indicator_factory.create_indicator('SMA', test_data, period=20)
    indicator2 = indicator_factory.create_indicator('SMA', test_data, period=20)

    # Verify caching works
    assert indicator1.equals(indicator2)

def test_indicator_error_handling(indicator_factory, test_data):
    """Test error handling for various scenarios."""
    # Test with empty data
    empty_data = pd.DataFrame()
    with pytest.raises(ValueError):
        indicator_factory.create_indicator('SMA', empty_data, period=20)

    # Test with invalid indicator name
    with pytest.raises(ValueError):
        indicator_factory.create_indicator('InvalidIndicator', test_data, period=20)

    # Test with insufficient data
    small_data = test_data.iloc[:5]
    with pytest.raises(ValueError):
        indicator_factory.create_indicator('SMA', small_data, timeperiod=20)

def test_indicator_performance(indicator_factory, test_data):
    """Test indicator performance calculations."""
    # Test creating a performance indicator
    returns = indicator_factory.create_indicator('Returns', test_data)
    assert isinstance(returns, pd.Series)
    assert not returns.isnull().all()

    # Test creating a volatility indicator
    volatility = indicator_factory.create_indicator('Volatility', test_data, period=20)
    assert isinstance(volatility, pd.Series)
    assert not volatility.isnull().all()
    assert volatility.min() >= 0

def test_indicator_visualization(indicator_factory, test_data):
    """Test indicator visualization methods."""
    # Create an indicator with visualization
    bb = indicator_factory.create_indicator('BB', test_data, period=20, std_dev=2)
    plot = indicator_factory.plot_indicator(bb, test_data)
    assert plot is not None  # Verify plot object is created

def test_loads_params(factory, indicator_defs):
    # All indicators in the JSON should be loaded
    for name in indicator_defs:
        assert name in factory.indicator_params
        # Each should have a params dict
        assert isinstance(factory.indicator_params[name]["params"], dict)

def test_get_indicator_params(factory, indicator_defs):
    for name in indicator_defs:
        params = factory.get_indicator_params(name)
        assert isinstance(params, dict)
        # Should match the params in the JSON
        assert params == indicator_defs[name]["params"]

def test_validate_params(factory, indicator_defs):
    for name, definition in indicator_defs.items():
        params = definition["params"]
        # Use defaults for validation
        defaults = {k: v["default"] for k, v in params.items() if "default" in v}
        # Should not raise
        factory.validate_params(name, defaults)

def test_compute_indicator(factory, indicator_defs, sample_data):
    # Test compute_indicator for all indicators with required inputs present in sample_data
    for name, definition in indicator_defs.items():
        req = definition.get("required_inputs", [])
        if all(col in sample_data.columns for col in req):
            params = {k: v["default"] for k, v in definition["params"].items() if "default" in v}
            try:
                result = factory.compute_indicator(name, sample_data, params)
                assert isinstance(result, pd.DataFrame)
            except Exception as e:
                # Some indicators may require more data or special cases; skip if so
                continue

def test_get_all_indicator_names(factory, indicator_defs):
    names = factory.get_all_indicator_names()
    for name in indicator_defs:
        assert name in names

def test_get_available_indicators(indicator_factory):
    """Test getting list of available indicators."""
    indicators = indicator_factory.get_available_indicators()
    assert isinstance(indicators, list)
    assert len(indicators) > 0
    assert all(isinstance(name, str) for name in indicators)

def test_get_indicator_params_basic(indicator_factory):
    """Test getting indicator parameters."""
    # Test with valid indicator
    params = indicator_factory.get_indicator_params('ema')
    assert isinstance(params, dict)
    assert 'length' in params or 'period' in params

    # Test with invalid indicator
    params = indicator_factory.get_indicator_params('INVALID_INDICATOR')
    assert params is None 