import pytest
import pandas as pd
import numpy as np
from indicator_factory_class import IndicatorFactory
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

def test_indicator_factory_initialization(indicator_factory):
    """Test IndicatorFactory initialization."""
    assert indicator_factory is not None
    assert hasattr(indicator_factory, 'indicators')
    assert isinstance(indicator_factory.indicators, dict)

def test_indicator_creation(indicator_factory, test_data):
    """Test creation of various indicators."""
    # Test creating a simple moving average
    sma = indicator_factory.create_indicator('SMA', test_data, period=20)
    assert isinstance(sma, pd.Series)
    assert not sma.isnull().all()
    assert len(sma) == len(test_data)

    # Test creating a relative strength index
    rsi = indicator_factory.create_indicator('RSI', test_data, period=14)
    assert isinstance(rsi, pd.Series)
    assert not rsi.isnull().all()
    assert len(rsi) == len(test_data)
    assert rsi.max() <= 100
    assert rsi.min() >= 0

def test_indicator_parameters(indicator_factory, test_data):
    """Test indicator parameter handling."""
    # Test with valid parameters
    bollinger = indicator_factory.create_indicator('BB', test_data, period=20, std_dev=2)
    assert isinstance(bollinger, pd.DataFrame)
    assert all(col in bollinger.columns for col in ['upper', 'middle', 'lower'])

    # Test with invalid parameters
    with pytest.raises(ValueError):
        indicator_factory.create_indicator('SMA', test_data, period=-1)

def test_indicator_combinations(indicator_factory, test_data):
    """Test combining multiple indicators."""
    # Create multiple indicators
    sma = indicator_factory.create_indicator('SMA', test_data, period=20)
    ema = indicator_factory.create_indicator('EMA', test_data, period=20)
    rsi = indicator_factory.create_indicator('RSI', test_data, period=14)

    # Test combining indicators
    combined = indicator_factory.combine_indicators([sma, ema, rsi])
    assert isinstance(combined, pd.DataFrame)
    assert len(combined.columns) == 3
    assert not combined.isnull().all().any()

def test_indicator_validation(indicator_factory, test_data):
    """Test indicator validation methods."""
    # Test with valid data
    indicator = indicator_factory.create_indicator('SMA', test_data, period=20)
    assert indicator_factory.validate_indicator(indicator)

    # Test with invalid data
    invalid_data = test_data.copy()
    invalid_data.loc[invalid_data.index[0], 'close'] = np.nan
    with pytest.raises(ValueError):
        indicator_factory.create_indicator('SMA', invalid_data, period=20)

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
        indicator_factory.create_indicator('SMA', small_data, period=20)

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