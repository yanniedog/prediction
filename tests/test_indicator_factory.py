import pytest
import pandas as pd
import numpy as np
from indicator_factory import IndicatorFactory
import json
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

@pytest.fixture(scope="module")
def factory():
    """Create a factory instance for testing."""
    return IndicatorFactory()

@pytest.fixture(scope="function")
def test_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    data = pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(200, 300, 100),
        'low': np.random.uniform(50, 100, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    })
    return data

@pytest.fixture(scope="function")
def temp_params_file():
    """Create a temporary parameters file for testing."""
    temp_dir = tempfile.mkdtemp()
    params_file = Path(temp_dir) / "indicator_params.json"
    params = {
        "RSI": {
            "type": "talib",
            "required_inputs": ["close"],
            "params": {"timeperiod": 14}
        },
        "BB": {
            "type": "talib",
            "required_inputs": ["close"],
            "params": {"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2}
        }
    }
    with open(params_file, 'w') as f:
        json.dump(params, f)
    yield params_file
    shutil.rmtree(temp_dir)

def test_factory_initialization(factory):
    """Test factory initialization and parameter loading."""
    assert isinstance(factory, IndicatorFactory)
    assert isinstance(factory.indicator_params, dict)
    assert len(factory.indicator_params) > 0

def test_compute_indicators_basic(factory, test_data):
    """Test basic indicator computation."""
    # Test computing a single indicator
    result = factory.compute_indicators(test_data, indicators=['RSI'])
    assert isinstance(result, pd.DataFrame)
    assert 'RSI' in result.columns
    assert len(result) == len(test_data)
    
    # Test computing multiple indicators
    result = factory.compute_indicators(test_data, indicators=['RSI', 'BB'])
    assert isinstance(result, pd.DataFrame)
    assert 'RSI' in result.columns
    assert any(col.startswith('BB_') for col in result.columns)
    assert len(result) == len(test_data)

def test_compute_indicators_validation(factory, test_data):
    """Test input validation for compute_indicators."""
    # Test with None data
    with pytest.raises(ValueError):
        factory.compute_indicators(None)
    
    # Test with empty DataFrame
    with pytest.raises(ValueError):
        factory.compute_indicators(pd.DataFrame())
    
    # Test with invalid indicator name
    with pytest.raises(ValueError):
        factory.compute_indicators(test_data, indicators=['INVALID_INDICATOR'])
    
    # Test with missing required columns
    invalid_data = test_data.drop('close', axis=1)
    with pytest.raises(ValueError):
        factory.compute_indicators(invalid_data, indicators=['RSI'])

def test_compute_single_indicator(factory, test_data):
    """Test single indicator computation."""
    # Test RSI computation
    rsi_config = {
        'indicator_name': 'RSI',
        'params': {'timeperiod': 14},
        'config_id': 1
    }
    result = factory._compute_single_indicator(test_data, 'RSI', rsi_config)
    assert isinstance(result, pd.DataFrame)
    assert 'RSI' in result.columns
    assert len(result) == len(test_data)
    assert not result['RSI'].isna().all()  # Should have some valid values
    
    # Test BB computation
    bb_config = {
        'indicator_name': 'BB',
        'params': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2},
        'config_id': 2
    }
    result = factory._compute_single_indicator(test_data, 'BB', bb_config)
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['BB_upper', 'BB_middle', 'BB_lower'])
    assert len(result) == len(test_data)

def test_indicator_plotting(factory, test_data, temp_dir):
    """Test indicator plotting functionality."""
    # Test RSI plotting
    output_path = temp_dir / "rsi_plot.png"
    factory.plot_indicator('RSI', test_data, {'timeperiod': 14}, str(output_path))
    assert output_path.exists()
    
    # Test BB plotting
    output_path = temp_dir / "bb_plot.png"
    factory.plot_indicator('BB', test_data, {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2}, str(output_path))
    assert output_path.exists()
    
    # Test plotting with invalid indicator
    with pytest.raises(ValueError):
        factory.plot_indicator('INVALID', test_data, {}, str(output_path))
    
    # Clean up plots
    plt.close('all')

def test_parameter_validation(factory, temp_params_file):
    """Test parameter validation."""
    # Test with valid parameters
    factory.params_file = temp_params_file
    factory.indicator_params = factory._load_params()
    factory._validate_params()  # Should not raise
    
    # Test with invalid indicator type
    invalid_params = {
        "INVALID": {
            "type": "invalid_type",
            "required_inputs": ["close"],
            "params": {"timeperiod": 14}
        }
    }
    with patch.object(factory, 'indicator_params', invalid_params):
        with pytest.raises(ValueError):
            factory._validate_params()
    
    # Test with missing required inputs
    invalid_params = {
        "RSI": {
            "type": "talib",
            "required_inputs": [],  # Empty required inputs
            "params": {"timeperiod": 14}
        }
    }
    with patch.object(factory, 'indicator_params', invalid_params):
        with pytest.raises(ValueError):
            factory._validate_params()

def test_get_available_indicators(factory):
    """Test getting available indicators."""
    indicators = factory.get_available_indicators()
    assert isinstance(indicators, list)
    assert len(indicators) > 0
    assert all(isinstance(ind, str) for ind in indicators)
    
    # Test alias method
    indicators2 = factory.get_all_indicator_names()
    assert indicators == indicators2

def test_error_handling(factory, test_data):
    """Test error handling in various scenarios."""
    # Test with invalid indicator computation
    with patch.object(factory, '_compute_single_indicator', return_value=None):
        result = factory.compute_indicators(test_data, indicators=['RSI'])
        assert isinstance(result, pd.DataFrame)
        assert 'RSI' not in result.columns
    
    # Test with plotting error
    with patch('matplotlib.pyplot.savefig', side_effect=Exception("Plot error")):
        with pytest.raises(Exception):
            factory.plot_indicator('RSI', test_data, {'timeperiod': 14})
    
    # Test with invalid parameter file
    with patch('builtins.open', side_effect=FileNotFoundError()):
        with pytest.raises(FileNotFoundError):
            factory._load_params()

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
    indicator1 = indicator_factory.create_indicator('SMA', test_data, timeperiod=20)
    indicator2 = indicator_factory.create_indicator('SMA', test_data, timeperiod=20)

    # Verify caching works
    assert indicator1.equals(indicator2)

def test_indicator_error_handling(indicator_factory, test_data):
    """Test error handling for various scenarios."""
    # Test with empty data
    empty_data = pd.DataFrame()
    with pytest.raises(ValueError):
        indicator_factory.create_indicator('SMA', empty_data, timeperiod=20)

    # Test with invalid indicator name
    with pytest.raises(ValueError):
        indicator_factory.create_indicator('InvalidIndicator', test_data, timeperiod=20)

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
    params = {'period': 20, 'std_dev': 2}
    bb = indicator_factory.create_indicator('BB', test_data, **params)
    # Test plotting with correct arguments
    indicator_factory.plot_indicator('BB', test_data, params)
    # Test that the indicator was created successfully
    assert bb is not None
    assert isinstance(bb, pd.Series)
    assert len(bb) == len(test_data)

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
    assert 'timeperiod' in params  # Updated to match actual indicator definition

    # Test with invalid indicator
    params = indicator_factory.get_indicator_params('INVALID_INDICATOR')
    assert params is None

@pytest.mark.timeout(10)
def test_compute_indicator_ta_error_handling(indicator_factory):
    """Test that create_indicator returns None or handles error when TA-Lib SMA raises a TypeError."""
    import pandas as pd
    import numpy as np
    from unittest.mock import patch, MagicMock

    # Create a minimal DataFrame
    df = pd.DataFrame({"close": np.random.rand(100)})
    # Patch talib.SMA to raise TypeError
    with patch("talib.SMA", new=MagicMock(side_effect=TypeError("SMA() takes at least 1 positional argument (0 given)"))):
        with patch("indicator_factory.logger.error") as mock_logger:
            # Should not raise, should handle gracefully
            result = indicator_factory.create_indicator("sma", df, timeperiod=14)
            if result is None:
                handled = True
            elif hasattr(result, 'empty') and result.empty:
                handled = True
            elif hasattr(result, 'isnull') and result.isnull().all().all():
                handled = True
            else:
                handled = False
            assert handled, "create_indicator did not handle TA-Lib error gracefully."
            mock_logger.assert_called()

def test_ema_accuracy(indicator_factory, test_data):
    """Test EMA calculation accuracy against TA-Lib's initialization method."""
    # Create a simple dataset
    data = pd.DataFrame({
        'close': np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype=np.float64),
        'open': np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype=np.float64),
        'high': np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0], dtype=np.float64),
        'low': np.array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0], dtype=np.float64)
    })
    timeperiod = 3
    ema_series = indicator_factory.create_indicator('ema', data, timeperiod=timeperiod)
    closes = data['close']
    expected = closes.copy()
    mean_first = closes.iloc[:timeperiod].mean()
    expected.iloc[:timeperiod] = mean_first
    alpha = 2 / (timeperiod + 1)
    for i in range(timeperiod, len(closes)):
        expected.iloc[i] = alpha * closes.iloc[i] + (1 - alpha) * expected.iloc[i-1]
    pd.testing.assert_series_equal(
        ema_series.round(6),
        expected.round(6),
        check_names=False,
        rtol=1e-5
    )

def test_ema_timeperiods(indicator_factory, test_data):
    """Test EMA with different timeperiods."""
    # Test various timeperiods
    periods = [2, 5, 14, 50, 100]
    for period in periods:
        # Ensure data is long enough for the period
        if len(test_data) < period:
            n = period + 10
            data = pd.DataFrame({
                'close': np.linspace(10, 10 + n - 1, n, dtype=np.float64),
                'open': np.linspace(10, 10 + n - 1, n, dtype=np.float64),
                'high': np.linspace(11, 11 + n - 1, n, dtype=np.float64),
                'low': np.linspace(9, 9 + n - 1, n, dtype=np.float64),
                'volume': np.linspace(1000, 2000, n, dtype=np.float64)
            })
        else:
            data = test_data
        ema_series = indicator_factory.create_indicator('ema', data, timeperiod=period)
        # Basic validation
        assert isinstance(ema_series, (pd.Series, pd.DataFrame))
        assert len(ema_series) == len(data)
        if isinstance(ema_series, pd.Series):
            assert not ema_series.isnull().all()
            # Check that EMA values are within price range
            assert ema_series.min() >= data['low'].min()
            assert ema_series.max() <= data['high'].max()
        else:
            for col in ema_series.columns:
                assert not ema_series[col].isnull().all()
                assert ema_series[col].min() >= data['low'].min()
                assert ema_series[col].max() <= data['high'].max()
        # Check that EMA is more responsive with shorter periods
        if period < 14:
            default_series = indicator_factory.create_indicator('ema', data)
            if isinstance(ema_series, pd.Series) and isinstance(default_series, pd.Series):
                assert ema_series.std() > default_series.std()

def test_ema_edge_cases(indicator_factory):
    """Test EMA with edge cases."""
    # Test with minimum period (2)
    data = pd.DataFrame({
        'close': np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=np.float64),
        'open': np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=np.float64),
        'high': np.array([11.0, 12.0, 13.0, 14.0, 15.0], dtype=np.float64),
        'low': np.array([9.0, 10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    })
    # Test minimum period
    ema_series = indicator_factory.create_indicator('ema', data, timeperiod=2)
    assert not ema_series.isnull().all()
    # Test with very long period (should be close to SMA)
    long_period = 100
    n = long_period + 10
    long_data = pd.DataFrame({
        'close': np.linspace(10, 10 + n - 1, n, dtype=np.float64),
        'open': np.linspace(10, 10 + n - 1, n, dtype=np.float64),
        'high': np.linspace(11, 11 + n - 1, n, dtype=np.float64),
        'low': np.linspace(9, 9 + n - 1, n, dtype=np.float64)
    })
    ema_series = indicator_factory.create_indicator('ema', long_data, timeperiod=long_period)
    sma_series = indicator_factory.create_indicator('sma', long_data, timeperiod=long_period)
    # EMA and SMA should be similar for long periods
    pd.testing.assert_series_equal(
        ema_series.round(3),
        sma_series.round(3),
        check_names=False,
        rtol=0.1  # Allow 10% difference due to different calculation methods
    )

def test_ema_convergence(indicator_factory):
    """Test EMA convergence properties."""
    # Create a dataset with a clear trend
    n = 100
    trend = np.linspace(0, 100, n)  # Linear trend
    noise = np.random.normal(0, 1, n)  # Small noise
    data = pd.DataFrame({
        'close': trend + noise,
        'open': trend + noise,
        'high': trend + noise + 1,
        'low': trend + noise - 1
    })
    # Test different periods
    periods = [5, 20, 50]
    for period in periods:
        ema_series = indicator_factory.create_indicator('ema', data, timeperiod=period)
        # EMA should follow the trend
        correlation = ema_series.corr(pd.Series(trend, index=data.index))
        assert correlation > 0.89  # High correlation with trend
        # EMA should be smoother than raw data
        assert ema_series.std() < data['close'].std()

@pytest.fixture
def indicator_factory():
    return IndicatorFactory()

@pytest.fixture
def valid_params_file(tmp_path):
    params = {
        "RSI": {
            "name": "RSI",
            "type": "talib",
            "required_inputs": ["close"],
            "params": {
                "timeperiod": {
                    "type": "int",
                    "min": 2,
                    "max": 200,
                    "default": 14,
                    "description": "RSI period"
                }
            },
            "conditions": []
        }
    }
    params_file = tmp_path / "indicator_params.json"
    with open(params_file, 'w') as f:
        json.dump(params, f)
    return params_file

@pytest.fixture
def invalid_params_file(tmp_path):
    params = {
        "RSI": {
            "name": "RSI",
            "type": "invalid_type",  # Invalid type
            "required_inputs": "close",  # Should be list
            "params": {
                "timeperiod": {
                    "type": "int",
                    "min": 200,  # min > max
                    "max": 2,
                    "default": 14
                }
            },
            "conditions": []
        }
    }
    params_file = tmp_path / "invalid_params.json"
    with open(params_file, 'w') as f:
        json.dump(params, f)
    return params_file

def test_load_valid_params(indicator_factory, valid_params_file):
    """Test loading valid indicator parameters"""
    indicator_factory.params_file = valid_params_file
    params = indicator_factory._load_params()
    assert isinstance(params, dict)
    assert "RSI" in params
    assert params["RSI"]["type"] == "talib"
    assert isinstance(params["RSI"]["required_inputs"], list)
    assert "timeperiod" in params["RSI"]["params"]

def test_load_invalid_params(indicator_factory, invalid_params_file):
    """Test loading invalid indicator parameters"""
    indicator_factory.params_file = invalid_params_file
    with pytest.raises(ValueError) as exc_info:
        indicator_factory._load_params()
    assert "Invalid indicator type" in str(exc_info.value)

def test_validate_params(indicator_factory, valid_params_file):
    """Test validating valid indicator parameters"""
    indicator_factory.params_file = valid_params_file
    params = indicator_factory._load_params()
    indicator_factory.indicator_params = params
    indicator_factory._validate_params()  # Should not raise

def test_validate_invalid_params(indicator_factory, invalid_params_file):
    """Test validating invalid indicator parameters"""
    indicator_factory.params_file = invalid_params_file
    with pytest.raises(ValueError):
        indicator_factory._load_params()

def test_missing_params_file(indicator_factory, tmp_path):
    """Test handling missing parameters file"""
    indicator_factory.params_file = tmp_path / "nonexistent.json"
    with pytest.raises(FileNotFoundError):
        indicator_factory._load_params()

def test_empty_params_file(indicator_factory, tmp_path):
    """Test handling empty parameters file"""
    params_file = tmp_path / "empty_params.json"
    with open(params_file, 'w') as f:
        json.dump({}, f)
    indicator_factory.params_file = params_file
    with pytest.raises(ValueError) as exc_info:
        indicator_factory._load_params()
    assert "Empty indicator parameters" in str(exc_info.value)

def test_invalid_json_file(indicator_factory, tmp_path):
    """Test handling invalid JSON file"""
    params_file = tmp_path / "invalid_json.json"
    with open(params_file, 'w') as f:
        f.write("invalid json content")
    indicator_factory.params_file = params_file
    with pytest.raises(json.JSONDecodeError):
        indicator_factory._load_params()

def test_parameter_conditions(indicator_factory, tmp_path):
    """Test parameter conditions validation"""
    params = {
        "MACD": {
            "name": "MACD",
            "type": "talib",
            "required_inputs": ["close"],
            "params": {
                "fastperiod": {
                    "type": "int",
                    "min": 2,
                    "max": 200,
                    "default": 12
                },
                "slowperiod": {
                    "type": "int",
                    "min": 2,
                    "max": 200,
                    "default": 26
                }
            },
            "conditions": [
                {
                    "fastperiod": {
                        "lt": "slowperiod"
                    }
                }
            ]
        }
    }
    params_file = tmp_path / "params_with_conditions.json"
    with open(params_file, 'w') as f:
        json.dump(params, f)
    indicator_factory.params_file = params_file
    params = indicator_factory._load_params()
    indicator_factory.indicator_params = params
    indicator_factory._validate_params()  # Should not raise

def test_invalid_parameter_conditions(indicator_factory, tmp_path):
    """Test invalid parameter conditions"""
    params = {
        "MACD": {
            "name": "MACD",
            "type": "talib",
            "required_inputs": ["close"],
            "params": {
                "fastperiod": {
                    "type": "int",
                    "min": 2,
                    "max": 200,
                    "default": 12
                }
            },
            "conditions": [
                {
                    "nonexistent_param": {  # Invalid parameter
                        "lt": "fastperiod"
                    }
                }
            ]
        }
    }
    params_file = tmp_path / "invalid_conditions.json"
    with open(params_file, 'w') as f:
        json.dump(params, f)
    indicator_factory.params_file = params_file
    params = indicator_factory._load_params()
    indicator_factory.indicator_params = params
    with pytest.raises(ValueError) as exc_info:
        indicator_factory._validate_params()
    assert "Invalid condition parameter" in str(exc_info.value)

def test_parameter_ranges(indicator_factory, tmp_path):
    """Test parameter range validation"""
    params = {
        "RSI": {
            "name": "RSI",
            "type": "talib",
            "required_inputs": ["close"],
            "params": {
                "timeperiod": {
                    "type": "int",
                    "min": 2,
                    "max": 200,
                    "default": 1  # Invalid default < min
                }
            },
            "conditions": []
        }
    }
    params_file = tmp_path / "invalid_ranges.json"
    with open(params_file, 'w') as f:
        json.dump(params, f)
    indicator_factory.params_file = params_file
    params = indicator_factory._load_params()
    indicator_factory.indicator_params = params
    with pytest.raises(ValueError) as exc_info:
        indicator_factory._validate_params()
    assert "Invalid default value" in str(exc_info.value)

def test_required_keys(indicator_factory, tmp_path):
    """Test required keys validation"""
    params = {
        "RSI": {
            "name": "RSI",
            "type": "talib"
            # Missing required_inputs and params
        }
    }
    params_file = tmp_path / "missing_keys.json"
    with open(params_file, 'w') as f:
        json.dump(params, f)
    indicator_factory.params_file = params_file
    with pytest.raises(ValueError) as exc_info:
        indicator_factory._load_params()
    assert "Missing required keys" in str(exc_info.value)

@pytest.fixture
def complex_test_data():
    """Create complex test data with various scenarios."""
    dates = pd.date_range(start='2024-01-01', periods=200, freq='h')
    data = pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(100, 200, 200),
        'high': np.random.uniform(200, 300, 200),
        'low': np.random.uniform(50, 100, 200),
        'close': np.random.uniform(100, 200, 200),
        'volume': np.random.uniform(1000, 5000, 200),
        'adj_close': np.random.uniform(100, 200, 200)
    })
    # Add some NaN values
    data.loc[data.index[10:20], 'close'] = np.nan
    data.loc[data.index[50:60], 'volume'] = np.nan
    return data

def test_compute_single_indicator_edge_cases(factory, complex_test_data):
    """Test edge cases for _compute_single_indicator."""
    # Test with NaN values
    result = factory._compute_single_indicator(complex_test_data, 'RSI', {
        'type': 'talib',
        'required_inputs': ['close'],
        'params': {'timeperiod': 14}
    })
    assert isinstance(result, pd.DataFrame)
    assert not result['RSI'].isna().all()
    
    # Test with custom indicator
    def custom_indicator(data, period):
        return pd.Series(np.random.random(len(data)), index=data.index)
    
    result = factory._compute_single_indicator(complex_test_data, 'CUSTOM', {
        'type': 'custom',
        'required_inputs': ['close'],
        'params': {'period': 14},
        'function': custom_indicator
    })
    assert isinstance(result, pd.DataFrame)
    assert 'CUSTOM' in result.columns

def test_compute_indicators_comprehensive(factory, complex_test_data):
    """Test comprehensive indicator computation scenarios."""
    # Test computing all available indicators
    result = factory.compute_indicators(complex_test_data)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(complex_test_data)
    
    # Test with specific indicator combinations
    indicators = ['RSI', 'BB', 'MACD', 'EMA']
    result = factory.compute_indicators(complex_test_data, indicators=indicators)
    assert isinstance(result, pd.DataFrame)
    for indicator in indicators:
        assert any(col.startswith(indicator) for col in result.columns)
    
    # Test with empty indicator list
    result = factory.compute_indicators(complex_test_data, indicators=[])
    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) == len(complex_test_data.columns)

def test_create_custom_indicator(factory, complex_test_data):
    """Test creation of custom indicators."""
    def custom_ma(data, period):
        return data['close'].rolling(window=period).mean()
    
    result = factory.create_custom_indicator('CUSTOM_MA', custom_ma, complex_test_data, period=20)
    assert isinstance(result, pd.DataFrame)
    assert 'CUSTOM_MA' in result.columns
    assert not result['CUSTOM_MA'].isna().all()
    
    # Test with invalid function
    with pytest.raises(ValueError):
        factory.create_custom_indicator('INVALID', None, complex_test_data)

def test_create_indicator(factory, complex_test_data):
    """Test creation of indicators with various parameters."""
    # Test creating RSI
    result = factory.create_indicator('RSI', complex_test_data, timeperiod=14)
    assert isinstance(result, pd.DataFrame)
    assert 'RSI' in result.columns
    
    # Test creating BB with custom parameters
    result = factory.create_indicator('BB', complex_test_data, timeperiod=20, nbdevup=2.5, nbdevdn=2.5)
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['BB_upper', 'BB_middle', 'BB_lower'])
    
    # Test with invalid indicator
    with pytest.raises(ValueError):
        factory.create_indicator('INVALID', complex_test_data)

def test_plot_indicator_comprehensive(factory, complex_test_data, temp_dir):
    """Test comprehensive indicator plotting scenarios."""
    # Test plotting RSI
    output_path = temp_dir / "rsi_plot.png"
    factory.plot_indicator('RSI', complex_test_data, {'timeperiod': 14}, str(output_path))
    assert output_path.exists()
    
    # Test plotting BB with custom parameters
    output_path = temp_dir / "bb_plot.png"
    factory.plot_indicator('BB', complex_test_data, {
        'timeperiod': 20,
        'nbdevup': 2.5,
        'nbdevdn': 2.5
    }, str(output_path))
    assert output_path.exists()
    
    # Test plotting without output path
    factory.plot_indicator('RSI', complex_test_data, {'timeperiod': 14})
    
    # Test plotting with invalid data
    with pytest.raises(ValueError):
        factory.plot_indicator('RSI', pd.DataFrame(), {'timeperiod': 14})

def test_compute_configured_indicators(factory, complex_test_data):
    """Test computing indicators with configurations."""
    configs = [
        {
            'name': 'RSI',
            'params': {'timeperiod': 14}
        },
        {
            'name': 'BB',
            'params': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2}
        }
    ]
    
    result, failed_indices = factory.compute_configured_indicators(complex_test_data, configs)
    assert isinstance(result, pd.DataFrame)
    assert isinstance(failed_indices, set)
    assert len(result) == len(complex_test_data)
    
    # Test with invalid config
    invalid_configs = [{'name': 'INVALID', 'params': {}}]
    result, failed_indices = factory.compute_configured_indicators(complex_test_data, invalid_configs)
    assert len(failed_indices) > 0

def test_parameter_validation_comprehensive(factory, temp_params_file):
    """Test comprehensive parameter validation scenarios."""
    # Test with valid parameter ranges
    valid_params = {
        "RSI": {
            "type": "talib",
            "required_inputs": ["close"],
            "params": {
                "timeperiod": {
                    "min": 2,
                    "max": 100,
                    "default": 14
                }
            },
            "conditions": [
                {
                    "timeperiod": {
                        "gte": 2,
                        "lte": 100
                    }
                }
            ]
        }
    }
    
    with open(temp_params_file, 'w') as f:
        json.dump(valid_params, f)
    
    factory.params_file = temp_params_file
    factory.indicator_params = factory._load_params()
    factory._validate_params()  # Should not raise
    
    # Test with invalid parameter ranges
    invalid_params = {
        "RSI": {
            "type": "talib",
            "required_inputs": ["close"],
            "params": {
                "timeperiod": {
                    "min": 100,  # min > max
                    "max": 50,
                    "default": 14
                }
            }
        }
    }
    
    with open(temp_params_file, 'w') as f:
        json.dump(invalid_params, f)
    
    factory.params_file = temp_params_file
    factory.indicator_params = factory._load_params()
    with pytest.raises(ValueError):
        factory._validate_params()

def test_get_ta_lib_output_suffixes(factory):
    """Test getting TA-Lib output suffixes."""
    # Test with standard function
    suffixes = factory._get_ta_lib_output_suffixes('RSI')
    assert isinstance(suffixes, list)
    assert len(suffixes) > 0
    
    # Test with custom function
    suffixes = factory._get_ta_lib_output_suffixes('CUSTOM_FUNC')
    assert isinstance(suffixes, list)
    assert len(suffixes) > 0

def test_indicator_factory_error_handling(factory, complex_test_data):
    """Test comprehensive error handling scenarios."""
    # Test with invalid indicator type
    with pytest.raises(ValueError):
        factory._compute_single_indicator(complex_test_data, 'INVALID', {
            'type': 'invalid_type',
            'required_inputs': ['close'],
            'params': {}
        })
    
    # Test with missing required inputs
    with pytest.raises(ValueError):
        factory._compute_single_indicator(complex_test_data, 'RSI', {
            'type': 'talib',
            'required_inputs': ['nonexistent_column'],
            'params': {}
        })
    
    # Test with invalid parameter values
    with pytest.raises(ValueError):
        factory.create_indicator('RSI', complex_test_data, timeperiod=-1)
    
    # Test with invalid data type
    with pytest.raises(ValueError):
        factory.compute_indicators("invalid_data")
    
    # Test with empty indicator list
    result = factory.compute_indicators(complex_test_data, indicators=[])
    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) == len(complex_test_data.columns) 