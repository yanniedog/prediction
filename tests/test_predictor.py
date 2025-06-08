import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from predictor import Predictor
import json
from unittest.mock import patch
from sqlite_manager import initialize_database
from typing import Dict, List, Any, Optional, Tuple
from predictor import (
    predict_price_movement,
    calculate_indicators,
    _cache_indicators,
    _get_cached_indicators,
    _calculate_correlation,
    _prepare_prediction_data
)

@pytest.fixture
def predictor():
    """Provide a Predictor instance for testing."""
    return Predictor()

@pytest.fixture
def test_data():
    """Provide test data for prediction operations."""
    # Create sample historical data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1h')
    data = pd.DataFrame({
        'open_time': [int(d.timestamp() * 1000) for d in dates],
        'open': np.random.normal(100, 1, len(dates)),
        'high': np.random.normal(101, 1, len(dates)),
        'low': np.random.normal(99, 1, len(dates)),
        'close': np.random.normal(100, 1, len(dates)),
        'volume': np.random.normal(1000, 100, len(dates))
    })
    return data

@pytest.fixture
def test_db_path(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test.db"
    return db_path

@pytest.fixture(scope="function")
def sample_data() -> pd.DataFrame:
    """Create sample price data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
    np.random.seed(42)
    data = pd.DataFrame({
        "open": np.random.normal(100, 1, 100),
        "high": np.random.normal(101, 1, 100),
        "low": np.random.normal(99, 1, 100),
        "close": np.random.normal(100, 1, 100),
        "volume": np.random.normal(1000, 100, 100)
    }, index=dates)
    data["high"] = data[["open", "close"]].max(axis=1) + abs(np.random.normal(0, 0.1, 100))
    data["low"] = data[["open", "close"]].min(axis=1) - abs(np.random.normal(0, 0.1, 100))
    return data

@pytest.fixture(scope="function")
def sample_indicator_definition() -> Dict[str, Any]:
    """Create sample indicator definition for testing."""
    return {
        "RSI": {
            "type": "talib",
            "required_inputs": ["close"],
            "params": {
                "timeperiod": {
                    "default": 14,
                    "min": 2,
                    "max": 100
                }
            }
        },
        "BB": {
            "type": "talib",
            "required_inputs": ["close"],
            "params": {
                "timeperiod": {
                    "default": 20,
                    "min": 5,
                    "max": 200
                },
                "nbdevup": {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 5.0
                },
                "nbdevdn": {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 5.0
                }
            }
        }
    }

@pytest.fixture(scope="function")
def prediction_data(sample_data: pd.DataFrame, sample_indicator_definition: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Prepare prediction data for testing."""
    return _prepare_prediction_data(sample_data, sample_indicator_definition)

def test_predictor_initialization():
    """Test Predictor initialization."""
    # Test successful initialization
    predictor = Predictor()
    assert predictor is not None
    
    # Test initialization without statsmodels
    with patch('predictor.STATSMODELS_AVAILABLE', False), \
         patch('predictor.sm', None):
        with pytest.raises(ImportError):
            Predictor()

def test_get_latest_data_point(predictor, test_data, test_db_path):
    """Test getting the latest data point."""
    # Save test data to database
    conn = sqlite3.connect(str(test_db_path))
    test_data.to_sql('historical_data', conn, if_exists='replace', index=False)
    conn.close()
    
    # Get latest data point
    latest_data = predictor.get_latest_data_point(test_db_path)
    assert isinstance(latest_data, pd.DataFrame)
    assert not latest_data.empty
    assert 'open_time' in latest_data.columns
    assert 'close' in latest_data.columns
    
    # Test with non-existent database, suppress error logs and print output
    with patch('predictor.logger.error'), patch('builtins.print'):
        latest_data = predictor.get_latest_data_point(Path('nonexistent.db'))
    assert latest_data is None

def test_get_historical_indicator_price_pairs(predictor, test_data, test_db_path):
    """Test getting historical indicator and price pairs."""
    # Create test indicator config
    indicator_config = {
        'config_id': 1,
        'indicator_name': 'Test Indicator',
        'params': {'window': 20}
    }
    
    # Test getting pairs
    pairs = predictor.get_historical_indicator_price_pairs(
        db_path=test_db_path,
        symbol_id=1,
        timeframe_id=1,
        indicator_config=indicator_config,
        lag=5,
        full_historical_data=test_data,
        indicator_series_cache={}
    )
    assert pairs is None or isinstance(pairs, pd.DataFrame)
    
    # Test with invalid indicator config
    invalid_config = {'indicator_name': 'Invalid'}
    pairs = predictor.get_historical_indicator_price_pairs(
        db_path=test_db_path,
        symbol_id=1,
        timeframe_id=1,
        indicator_config=invalid_config,
        lag=5,
        full_historical_data=test_data,
        indicator_series_cache={}
    )
    assert pairs is None

def test_perform_prediction_regression(predictor, test_data):
    """Test performing prediction regression."""
    # Create sample historical pairs
    hist_pairs = pd.DataFrame({
        'Indicator_t': np.random.normal(0, 1, 100),
        'Close_t_plus_lag': np.random.normal(100, 1, 100)
    })
    
    # Test regression
    result = predictor.perform_prediction_regression(
        hist_pairs=hist_pairs,
        current_indicator_value=0.5,
        current_lag=5
    )
    assert result is None or isinstance(result, dict)
    
    # Test with insufficient data
    small_pairs = pd.DataFrame({
        'Indicator_t': [1, 2],
        'Close_t_plus_lag': [100, 101]
    })
    result = predictor.perform_prediction_regression(
        hist_pairs=small_pairs,
        current_indicator_value=0.5,
        current_lag=5
    )
    assert result is None

def test_predict_price(predictor, test_data, test_db_path):
    """Test predicting price."""
    # Initialize database schema
    if not initialize_database(str(test_db_path)):
        pytest.fail("Failed to initialize database schema")
    
    # Save test data to database
    conn = sqlite3.connect(str(test_db_path))
    try:
        # Get symbol and timeframe IDs
        conn.execute("BEGIN;")
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO symbols (symbol) VALUES (?)", ('BTCUSDT',))
        cursor.execute("INSERT OR IGNORE INTO timeframes (timeframe) VALUES (?)", ('1h',))
        cursor.execute("SELECT id FROM symbols WHERE symbol = ?", ('BTCUSDT',))
        symbol_id = cursor.fetchone()[0]
        cursor.execute("SELECT id FROM timeframes WHERE timeframe = ?", ('1h',))
        timeframe_id = cursor.fetchone()[0]
        conn.commit()
        
        # Add symbol_id and timeframe_id to test data
        test_data['symbol_id'] = symbol_id
        test_data['timeframe_id'] = timeframe_id
        test_data['close_time'] = test_data['open_time'] + 3600000  # Add 1 hour in milliseconds
        
        # Save to historical_data table
        test_data.to_sql('historical_data', conn, if_exists='replace', index=False)
    finally:
        conn.close()
    
    # Test prediction
    predictor.predict_price(
        db_path=test_db_path,
        symbol='BTCUSDT',
        timeframe='1h',
        final_target_lag=5
    )
    
    # Test with non-existent database, suppress error logs and print output
    with patch('predictor.logger.error'), patch('builtins.print'):
        predictor.predict_price(
            db_path=Path('nonexistent.db'),
            symbol='BTCUSDT',
            timeframe='1h',
            final_target_lag=5
        )

def test_plot_predicted_path(predictor, tmp_path):
    """Test plotting predicted path."""
    # Create sample data
    dates = [datetime.now() + timedelta(hours=i) for i in range(10)]
    prices = [100.0 + i for i in range(10)]  # Ensure float values
    ci_lower = [p - 1.0 for p in prices]
    ci_upper = [p + 1.0 for p in prices]
    prediction_results = [
        {
            'lag': 5,
            'indicator_name': 'Test Indicator',
            'correlation': 0.8,
            'predicted_price': 105.0,
            'confidence_interval': (104.0, 106.0)
        }
    ]
    
    # Test plotting
    predictor.plot_predicted_path(
        dates=dates,
        prices=prices,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        timeframe='1h',
        symbol='BTCUSDT',
        file_prefix=str(tmp_path / 'test_prediction'),
        final_target_lag=5,
        prediction_results=prediction_results
    )
    
    # Verify plot file was created
    plot_files = list(tmp_path.glob('test_prediction*.png'))
    assert len(plot_files) > 0

def test_prepare_prediction_data(sample_data: pd.DataFrame, sample_indicator_definition: Dict[str, Any]):
    """Test preparation of prediction data."""
    data, indicator_def = _prepare_prediction_data(sample_data, sample_indicator_definition)
    
    # Verify data preparation
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert all(col in data.columns for col in ["open", "high", "low", "close", "volume"])
    
    # Verify indicator definition
    assert isinstance(indicator_def, dict)
    assert "RSI" in indicator_def
    assert "BB" in indicator_def
    assert "params" in indicator_def["RSI"]
    assert "params" in indicator_def["BB"]

def test_calculate_indicators(prediction_data: Tuple[pd.DataFrame, Dict[str, Any]]):
    """Test indicator calculation."""
    data, indicator_def = prediction_data
    
    # Test RSI calculation
    indicators = calculate_indicators(data, indicator_def["RSI"], {"timeperiod": 14})
    assert isinstance(indicators, pd.Series)
    assert not indicators.empty
    assert not indicators.isna().all()
    
    # Test BB calculation
    indicators = calculate_indicators(data, indicator_def["BB"], {
        "timeperiod": 20,
        "nbdevup": 2.0,
        "nbdevdn": 2.0
    })
    assert isinstance(indicators, pd.Series)
    assert not indicators.empty
    assert not indicators.isna().all()
    
    # Test invalid parameters
    with pytest.raises(ValueError):
        calculate_indicators(data, indicator_def["RSI"], {"timeperiod": 1})  # Below min
    
    # Test missing required parameter
    with pytest.raises(ValueError):
        calculate_indicators(data, indicator_def["RSI"], {})  # Missing timeperiod

def test_indicator_caching(prediction_data: Tuple[pd.DataFrame, Dict[str, Any]]):
    """Test indicator caching functionality."""
    data, indicator_def = prediction_data
    
    # Calculate and cache indicators
    config = {"timeperiod": 14}
    indicators = calculate_indicators(data, indicator_def["RSI"], config)
    _cache_indicators(data, indicator_def["RSI"], config, indicators)
    
    # Retrieve cached indicators
    cached_indicators = _get_cached_indicators(data, indicator_def["RSI"], config)
    assert isinstance(cached_indicators, pd.Series)
    assert not cached_indicators.empty
    assert cached_indicators.equals(indicators)
    
    # Test cache miss
    different_config = {"timeperiod": 20}
    cached_indicators = _get_cached_indicators(data, indicator_def["RSI"], different_config)
    assert cached_indicators is None

def test_calculate_correlation(prediction_data: Tuple[pd.DataFrame, Dict[str, Any]]):
    """Test correlation calculation."""
    data, indicator_def = prediction_data
    
    # Calculate indicators
    indicators = calculate_indicators(data, indicator_def["RSI"], {"timeperiod": 14})
    
    # Test correlation calculation
    correlation = _calculate_correlation(indicators, data["close"], lag=1)
    assert isinstance(correlation, float)
    assert not np.isnan(correlation)
    assert not np.isinf(correlation)
    assert -1 <= correlation <= 1
    
    # Test with different lags
    correlations = [_calculate_correlation(indicators, data["close"], lag=i) for i in range(1, 6)]
    assert len(correlations) == 5
    assert all(isinstance(c, float) for c in correlations)
    assert all(not np.isnan(c) for c in correlations)
    assert all(not np.isinf(c) for c in correlations)
    assert all(-1 <= c <= 1 for c in correlations)

def test_predict_price_movement(prediction_data: Tuple[pd.DataFrame, Dict[str, Any]]):
    """Test price movement prediction."""
    data, indicator_def = prediction_data
    
    # Test prediction with RSI
    predictions = predict_price_movement(data, indicator_def["RSI"], {"timeperiod": 14}, lag=1)
    assert isinstance(predictions, pd.Series)
    assert not predictions.empty
    assert not predictions.isna().all()
    
    # Test prediction with BB
    predictions = predict_price_movement(data, indicator_def["BB"], {
        "timeperiod": 20,
        "nbdevup": 2.0,
        "nbdevdn": 2.0
    }, lag=1)
    assert isinstance(predictions, pd.Series)
    assert not predictions.empty
    assert not predictions.isna().all()
    
    # Test with different lags
    for lag in range(1, 6):
        predictions = predict_price_movement(data, indicator_def["RSI"], {"timeperiod": 14}, lag=lag)
        assert isinstance(predictions, pd.Series)
        assert not predictions.empty
        assert not predictions.isna().all()

def test_prediction_accuracy(prediction_data: Tuple[pd.DataFrame, Dict[str, Any]]):
    """Test prediction accuracy."""
    data, indicator_def = prediction_data
    
    # Generate predictions
    predictions = predict_price_movement(data, indicator_def["RSI"], {"timeperiod": 14}, lag=1)
    
    # Calculate actual price movements
    actual_movements = data["close"].pct_change().shift(-1)  # Next period's return
    
    # Calculate accuracy metrics
    correct_predictions = (predictions > 0) == (actual_movements > 0)
    accuracy = correct_predictions.mean()
    
    assert isinstance(accuracy, float)
    assert not np.isnan(accuracy)
    assert 0 <= accuracy <= 1

def test_error_handling(prediction_data: Tuple[pd.DataFrame, Dict[str, Any]]):
    """Test error handling in prediction."""
    data, indicator_def = prediction_data
    
    # Test with empty data
    with pytest.raises(ValueError):
        predict_price_movement(pd.DataFrame(), indicator_def["RSI"], {"timeperiod": 14})
    
    # Test with invalid indicator definition
    with pytest.raises(ValueError):
        predict_price_movement(data, {}, {"timeperiod": 14})
    
    # Test with invalid parameters
    with pytest.raises(ValueError):
        predict_price_movement(data, indicator_def["RSI"], {"timeperiod": 1})  # Below min
    
    # Test with missing required parameter
    with pytest.raises(ValueError):
        predict_price_movement(data, indicator_def["RSI"], {})  # Missing timeperiod
    
    # Test with invalid lag
    with pytest.raises(ValueError):
        predict_price_movement(data, indicator_def["RSI"], {"timeperiod": 14}, lag=0)  # Invalid lag

def test_data_validation(prediction_data: Tuple[pd.DataFrame, Dict[str, Any]]):
    """Test data validation in prediction."""
    data, indicator_def = prediction_data
    
    # Test with missing required columns
    invalid_data = data.drop(columns=["close"])
    with pytest.raises(ValueError):
        predict_price_movement(invalid_data, indicator_def["RSI"], {"timeperiod": 14})
    
    # Test with non-numeric data
    invalid_data = data.copy()
    invalid_data["close"] = "invalid"
    with pytest.raises(ValueError):
        predict_price_movement(invalid_data, indicator_def["RSI"], {"timeperiod": 14})
    
    # Test with all NaN data
    invalid_data = data.copy()
    invalid_data["close"] = np.nan
    with pytest.raises(ValueError):
        predict_price_movement(invalid_data, indicator_def["RSI"], {"timeperiod": 14})

def test_indicator_validation(prediction_data: Tuple[pd.DataFrame, Dict[str, Any]]):
    """Test indicator validation in prediction."""
    data, indicator_def = prediction_data
    
    # Test with invalid indicator type
    invalid_def = indicator_def["RSI"].copy()
    invalid_def["type"] = "invalid"
    with pytest.raises(ValueError):
        predict_price_movement(data, invalid_def, {"timeperiod": 14})
    
    # Test with missing required inputs
    invalid_def = indicator_def["RSI"].copy()
    invalid_def["required_inputs"] = ["invalid"]
    with pytest.raises(ValueError):
        predict_price_movement(data, invalid_def, {"timeperiod": 14})
    
    # Test with invalid parameter definition
    invalid_def = indicator_def["RSI"].copy()
    invalid_def["params"]["timeperiod"]["min"] = 100
    invalid_def["params"]["timeperiod"]["max"] = 50
    with pytest.raises(ValueError):
        predict_price_movement(data, invalid_def, {"timeperiod": 14}) 