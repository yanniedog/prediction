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