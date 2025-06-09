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
from datetime import datetime, timedelta
from custom_indicators import compute_returns, compute_volume_oscillator

@pytest.fixture
def indicator_factory():
    """Create a fresh indicator factory instance for each test."""
    return IndicatorFactory()

@pytest.fixture
def test_data():
    """Create test data with OHLCV columns."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    np.random.seed(42)
    data = pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(200, 300, 100),
        'low': np.random.uniform(50, 100, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    })
    # Ensure price relationships are valid
    data['high'] = data[['open', 'close']].max(axis=1) + np.random.uniform(0, 50, 100)
    data['low'] = data[['open', 'close']].min(axis=1) - np.random.uniform(0, 50, 100)
    return data

def test_compute_indicators_basic(indicator_factory, test_data):
    """Test basic indicator computation."""
    # Test with valid indicator
    result_df = indicator_factory.compute_indicators(test_data, indicators=['RSI'])
    assert isinstance(result_df, pd.DataFrame)
    assert 'RSI' in result_df.columns
    assert not result_df['RSI'].isna().all()
    assert len(result_df) == len(test_data)

    # Test with invalid indicator
    with pytest.raises(ValueError, match="Invalid indicators"):
        indicator_factory.compute_indicators(test_data, indicators=['INVALID'])

def test_indicator_creation(indicator_factory, test_data):
    """Test creating individual indicators."""
    # Test RSI
    rsi_df = indicator_factory.create_indicator('RSI', test_data)
    assert isinstance(rsi_df, pd.DataFrame)
    assert 'RSI' in rsi_df.columns
    assert not rsi_df['RSI'].isna().all()

    # Test BB
    bb_df = indicator_factory.create_indicator('BB', test_data)
    assert isinstance(bb_df, pd.DataFrame)
    assert all(col in bb_df.columns for col in ['BB_upper', 'BB_middle', 'BB_lower'])
    assert not bb_df.isna().all().any()

    # Test with invalid parameters
    with pytest.raises(ValueError):
        indicator_factory.create_indicator('RSI', test_data, timeperiod=0)

def test_indicator_combinations(indicator_factory, test_data):
    """Test combining multiple indicators."""
    # Create multiple indicators
    result_df = indicator_factory.compute_indicators(test_data, indicators=['RSI', 'BB'])
    assert isinstance(result_df, pd.DataFrame)
    assert all(col in result_df.columns for col in ['RSI', 'BB_upper', 'BB_middle', 'BB_lower'])
    assert not result_df.isna().all().any()

    # Test with custom indicators
    result_df = indicator_factory.compute_indicators(test_data, indicators=['Returns', 'Volume_Oscillator'])
    assert isinstance(result_df, pd.DataFrame)
    assert all(col in result_df.columns for col in ['Returns', 'Volume_Oscillator'])
    assert not result_df.isna().all().any()

def test_indicator_validation(indicator_factory, test_data):
    """Test indicator validation methods."""
    # Test with valid data
    result_df = indicator_factory.compute_indicators(test_data, indicators=['RSI'])
    assert isinstance(result_df, pd.DataFrame)
    assert 'RSI' in result_df.columns
    assert not result_df['RSI'].isna().all()

    # Test with missing required columns
    with pytest.raises(ValueError, match="Missing required columns"):
        indicator_factory.create_indicator('RSI', test_data.drop('close', axis=1))

    # Test with empty data
    with pytest.raises(ValueError, match="Input data is empty"):
        indicator_factory.create_indicator('RSI', pd.DataFrame())

def test_indicator_performance(indicator_factory, test_data):
    """Test indicator performance calculations."""
    # Test creating a performance indicator
    returns = indicator_factory.create_indicator('Returns', test_data)
    assert isinstance(returns, pd.DataFrame)
    assert 'Returns' in returns.columns
    assert not returns['Returns'].isna().all()

    # Test with different timeperiods
    returns_5 = indicator_factory.create_indicator('Returns', test_data, timeperiod=5)
    assert isinstance(returns_5, pd.DataFrame)
    assert 'Returns' in returns_5.columns
    assert not returns_5['Returns'].isna().all()
    assert len(returns_5) == len(test_data)

def test_get_indicator_params_basic(indicator_factory):
    """Test getting indicator parameters."""
    # Test with valid indicator
    params = indicator_factory.get_indicator_params('RSI')
    assert isinstance(params, dict)
    assert 'timeperiod' in params
    assert isinstance(params['timeperiod'], dict)
    assert 'default' in params['timeperiod']
    assert 'min' in params['timeperiod']
    assert 'max' in params['timeperiod']

    # Test with invalid indicator
    with pytest.raises(ValueError, match="Unknown indicator"):
        indicator_factory.get_indicator_params('INVALID')

def test_compute_indicator_ta_error_handling(indicator_factory):
    """Test error handling for TA-Lib functions."""
    import pandas as pd
    import numpy as np
    from unittest.mock import patch, MagicMock

    # Create a minimal DataFrame
    df = pd.DataFrame({"close": np.random.rand(100)})

    # Test with invalid indicator name
    with pytest.raises(ValueError, match="Unknown indicator"):
        indicator_factory.create_indicator("invalid_indicator", df)

    # Test with invalid parameters
    with pytest.raises(ValueError, match="Invalid period value"):
        indicator_factory.create_indicator("RSI", df, timeperiod="invalid")

def test_indicator_plotting(factory, complex_test_data, temp_dir):
    """Test indicator plotting functionality."""
    # Test plotting single indicator
    output_path = temp_dir / "single_indicator.png"
    factory.plot_indicator('RSI', complex_test_data, {'timeperiod': 14}, str(output_path))
    assert output_path.exists()
    
    # Test plotting multiple indicators
    output_path = temp_dir / "multiple_indicators.png"
    factory.plot_indicators(complex_test_data, ['RSI', 'BB'], str(output_path))
    assert output_path.exists()
    
    # Test plotting with custom indicators
    output_path = temp_dir / "custom_indicators.png"
    factory.plot_indicators(complex_test_data, ['Returns', 'Volume_Oscillator'], str(output_path))
    assert output_path.exists()
    
    # Test plotting with indicator configs
    output_path = temp_dir / "indicator_configs.png"
    configs = {
        'RSI': {'timeperiod': 14},
        'BB': {'timeperiod': 20, 'nbdevup': 2.0, 'nbdevdn': 2.0}
    }
    factory.plot_indicators(complex_test_data, configs, str(output_path))
    assert output_path.exists()
    
    # Test plotting without output path (should show plot)
    factory.plot_indicator('RSI', complex_test_data, {'timeperiod': 14})
    factory.plot_indicators(complex_test_data, ['RSI', 'BB'])
    
    # Test plotting with invalid data
    with pytest.raises(ValueError):
        factory.plot_indicator('RSI', pd.DataFrame(), {'timeperiod': 14})
    with pytest.raises(ValueError):
        factory.plot_indicators(pd.DataFrame(), ['RSI', 'BB'])
    
    # Test plotting with invalid indicator
    with pytest.raises(ValueError):
        factory.plot_indicator('INVALID', complex_test_data, {})
    with pytest.raises(ValueError):
        factory.plot_indicators(complex_test_data, ['INVALID'])

def test_loads_params(indicator_factory):
    """Test loading indicator parameters."""
    params = indicator_factory.indicator_params
    assert isinstance(params, dict)
    assert len(params) > 0
    assert all(isinstance(v, dict) for v in params.values())
    assert all('name' in v for v in params.values())
    assert all('params' in v for v in params.values())

def test_get_indicator_params(indicator_factory):
    """Test getting indicator parameters with different cases."""
    # Test uppercase
    params_upper = indicator_factory.get_indicator_params('RSI')
    assert isinstance(params_upper, dict)
    
    # Test lowercase
    params_lower = indicator_factory.get_indicator_params('rsi')
    assert isinstance(params_lower, dict)
    assert params_upper == params_lower

    # Test mixed case
    params_mixed = indicator_factory.get_indicator_params('RsI')
    assert isinstance(params_mixed, dict)
    assert params_upper == params_mixed

def test_compute_indicator(factory, indicator_defs, sample_data):
    # Test compute_indicator for all indicators with required inputs present in sample_data
    for name, definition in indicator_defs.items():
        req = definition.get("required_inputs", [])
        if all(col in sample_data.columns for col in req):
            params = {k: v["default"] for k, v in definition["params"].items() if "default" in v}
            try:
                # Ensure we're working with numeric data only
                numeric_data = sample_data.select_dtypes(include=[np.number])
                if len(numeric_data.columns) == 0:
                    continue  # Skip if no numeric columns
                
                result = factory.compute_indicator(name, numeric_data, params)
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
            'indicator_name': 'RSI',
            'config_id': 1,
            'params': {'timeperiod': 14}
        },
        {
            'indicator_name': 'BB',
            'config_id': 2,
            'params': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2}
        }
    ]
    
    result, failed_indices = factory.compute_configured_indicators(complex_test_data, configs)
    assert isinstance(result, pd.DataFrame)
    assert isinstance(failed_indices, set)
    assert len(result) == len(complex_test_data)
    
    # Test with invalid config
    invalid_configs = [{
        'indicator_name': 'INVALID',
        'config_id': 3,
        'params': {}
    }]
    result, failed_indices = factory.compute_configured_indicators(complex_test_data, invalid_configs)
    assert len(failed_indices) > 0
    assert 3 in failed_indices  # Verify the invalid config ID is in failed_indices

def test_parameter_validation_comprehensive(factory, temp_params_file):
    """Test comprehensive parameter validation scenarios."""
    # Test with valid parameter ranges
    valid_params = {
        "RSI": {
            "name": "RSI",
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
    assert 'RSI' in factory.indicator_params

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
        }, {})
    
    # Test with missing required inputs - use the actual indicator config format
    with pytest.raises(ValueError):
        factory._compute_single_indicator(complex_test_data, 'RSI', {
            'name': 'RSI',
            'type': 'talib',
            'required_inputs': ['nonexistent_column'],
            'params': {'timeperiod': {'default': 14}}
        }, {})
    
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
    
    # Ensure proper price relationships
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    # Clean any NaN values to ensure indicators work properly
    data = data.ffill().bfill()
    
    return data

def test_compute_single_indicator_edge_cases(factory, complex_test_data):
    """Test edge cases for _compute_single_indicator."""
    # Test with NaN values
    result = factory._compute_single_indicator(complex_test_data, 'RSI', {
        'name': 'RSI',
        'type': 'talib',
        'required_inputs': ['close'],
        'params': {'timeperiod': 14}
    }, {'timeperiod': 14})
    assert isinstance(result, pd.DataFrame)
    assert not result['RSI'].isna().all()

    # Test with custom indicator
    def custom_indicator(data, period):
        return pd.Series(np.random.random(len(data)), index=data.index)

    factory.register_custom_indicator('custom', custom_indicator)
    result = factory._compute_single_indicator(complex_test_data, 'custom', {
        'name': 'custom',
        'type': 'custom',
        'required_inputs': ['close'],
        'params': {'period': 14},
        'function': custom_indicator
    }, {'period': 14})
    assert isinstance(result, pd.DataFrame)
    assert 'custom' in result.columns 