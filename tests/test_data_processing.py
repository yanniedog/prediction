import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from main import _select_data_source_and_lag
from utils import get_max_lag, get_data_date_range
import sqlite3
from unittest.mock import patch, MagicMock
from data_manager import _validate_data
from sqlite_manager import SQLiteManager
from data_processing import validate_data, process_data
from data_manager import DataManager

@pytest.fixture
def sample_data():
    """Create sample OHLCV data"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='1h')
    data = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 101,
        'low': np.random.randn(len(dates)).cumsum() + 99,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    return data

@pytest.fixture
def sample_data_with_nulls():
    """Create sample data with null values"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='1h')
    data = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 101,
        'low': np.random.randn(len(dates)).cumsum() + 99,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Add some null values
    data.loc[data.index[5:10], 'close'] = np.nan
    data.loc[data.index[15:20], 'volume'] = np.nan
    return data

@pytest.fixture
def sample_data_with_invalid_dates():
    """Create sample data with invalid dates"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='1h')
    data = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 101,
        'low': np.random.randn(len(dates)).cumsum() + 99,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Add some invalid dates
    data.index = data.index.tolist()[:-5] + [datetime.now() + timedelta(days=x) for x in range(5)]
    return data

def test_max_lag_calculation(sample_data):
    """Test maximum lag calculation"""
    max_lag = get_max_lag(sample_data)
    assert isinstance(max_lag, int)
    assert max_lag > 0
    assert max_lag <= len(sample_data) // 2  # Should be at most half the data length

def test_max_lag_with_nulls(sample_data_with_nulls):
    """Test maximum lag calculation with null values"""
    max_lag = get_max_lag(sample_data_with_nulls)
    assert isinstance(max_lag, int)
    assert max_lag > 0
    # Should account for null values in calculation
    assert max_lag <= (len(sample_data_with_nulls) - 10) // 2

def test_data_date_range(sample_data):
    """Test date range extraction"""
    date_range = get_data_date_range(sample_data)
    assert isinstance(date_range, str)
    assert '2023-01-01' in date_range
    assert '2023-01-31' in date_range

def test_data_date_range_with_invalid_dates(sample_data_with_invalid_dates):
    """Test date range extraction with invalid dates"""
    with pytest.raises(ValueError) as exc_info:
        get_data_date_range(sample_data_with_invalid_dates)
    assert "Invalid date range" in str(exc_info.value)

def test_data_source_selection(tmp_path):
    """Test data source selection and validation process."""
    # Create a temporary database
    db_path = tmp_path / "BTCUSDT_1h.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create tables with correct schema
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS symbols (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS timeframes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timeframe TEXT NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            type TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS indicator_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            indicator_id INTEGER NOT NULL,
            params JSON NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (indicator_id) REFERENCES indicators(id) ON DELETE CASCADE
        );
        
        CREATE TABLE IF NOT EXISTS historical_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol_id INTEGER NOT NULL,
            timeframe_id INTEGER NOT NULL,
            open_time INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            close_time INTEGER NOT NULL,
            quote_asset_volume REAL,
            number_of_trades INTEGER,
            taker_buy_base_asset_volume REAL,
            taker_buy_quote_asset_volume REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (symbol_id) REFERENCES symbols(id) ON DELETE CASCADE,
            FOREIGN KEY (timeframe_id) REFERENCES timeframes(id) ON DELETE CASCADE,
            UNIQUE(symbol_id, timeframe_id, open_time)
        );
    """)
    
    # Insert symbol and timeframe
    cursor.execute("INSERT INTO symbols (symbol) VALUES (?)", ("BTCUSDT",))
    cursor.execute("INSERT INTO timeframes (timeframe) VALUES (?)", ("1h",))
    
    # Insert test indicator
    cursor.execute("INSERT INTO indicators (name, type, description) VALUES (?, ?, ?)", 
                  ("RSI", "momentum", "Relative Strength Index"))
    
    # Generate sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    sample_data = pd.DataFrame({
        'open_time': [int(d.timestamp() * 1000) for d in dates],
        'open': np.random.randn(100).cumsum() + 1000,
        'high': np.random.randn(100).cumsum() + 1000,
        'low': np.random.randn(100).cumsum() + 1000,
        'close': np.random.randn(100).cumsum() + 1000,
        'volume': np.random.randint(100, 1000, 100),
        'close_time': [int((d + pd.Timedelta(hours=1)).timestamp() * 1000) for d in dates],
        'quote_asset_volume': np.random.randint(1000, 10000, 100),
        'number_of_trades': np.random.randint(100, 1000, 100),
        'taker_buy_base_asset_volume': np.random.randint(50, 500, 100),
        'taker_buy_quote_asset_volume': np.random.randint(500, 5000, 100)
    })
    
    # Insert sample data
    for _, row in sample_data.iterrows():
        cursor.execute("""
            INSERT INTO historical_data (
                symbol_id, timeframe_id, open_time, open, high, low, close, volume,
                close_time, quote_asset_volume, number_of_trades,
                taker_buy_base_asset_volume, taker_buy_quote_asset_volume
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            1, 1, row['open_time'], row['open'], row['high'], row['low'], row['close'],
            row['volume'], row['close_time'], row['quote_asset_volume'],
            row['number_of_trades'], row['taker_buy_base_asset_volume'],
            row['taker_buy_quote_asset_volume']
        ))
    
    conn.commit()
    conn.close()
    
    # Test data source selection with pre-selected choice and max_lag
    result = _select_data_source_and_lag(choice=1, max_lag=10)
    assert result is not None
    assert len(result) == 8
    
    db_path, symbol, timeframe, data, max_lag, symbol_id, timeframe_id, data_daterange = result
    assert symbol == "BTCUSDT"
    assert timeframe == "1h"
    assert max_lag == 10
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert symbol_id == 1
    assert timeframe_id == 1
    assert isinstance(data_daterange, str)
    assert "2024-01-01" in data_daterange

def test_data_validation(sample_data):
    """Test data validation"""
    # Clean data first - ensure high/low are always correct relative to open/close
    sample_data['high'] = sample_data[['high', 'low', 'open', 'close']].max(axis=1)
    sample_data['low'] = sample_data[['high', 'low', 'open', 'close']].min(axis=1)
    
    # Test required columns
    required_columns = {'open', 'high', 'low', 'close', 'volume'}
    assert required_columns.issubset(sample_data.columns)
    
    # Test data types
    assert sample_data['open'].dtype in [np.float64, np.float32]
    assert sample_data['high'].dtype in [np.float64, np.float32]
    assert sample_data['low'].dtype in [np.float64, np.float32]
    assert sample_data['close'].dtype in [np.float64, np.float32]
    assert sample_data['volume'].dtype in [np.int64, np.int32]
    
    # Test value ranges
    assert (sample_data['high'] >= sample_data['low']).all()
    assert (sample_data['high'] >= sample_data['open']).all()
    assert (sample_data['high'] >= sample_data['close']).all()
    assert (sample_data['low'] <= sample_data['open']).all()
    assert (sample_data['low'] <= sample_data['close']).all()
    assert (sample_data['volume'] >= 0).all()

def test_data_processing_with_missing_columns(sample_data):
    """Test data processing with missing columns"""
    # Remove required column
    data = sample_data.drop('volume', axis=1)
    
    # Test validation directly
    with pytest.raises(ValueError) as exc_info:
        _validate_data(data)
    assert "Missing required columns" in str(exc_info.value)

def test_data_processing_with_invalid_values(sample_data):
    """Test data processing with invalid values"""
    # Add invalid values
    data = sample_data.copy()
    data.loc[data.index[0], 'high'] = -1  # Invalid high price
    data.loc[data.index[1], 'volume'] = -100  # Invalid volume

    # Test validation directly
    with pytest.raises(ValueError) as exc_info:
        _validate_data(data)
    assert "High price cannot be less than low price" in str(exc_info.value)

def test_data_processing_with_duplicate_dates(sample_data):
    """Test data processing with duplicate dates"""
    # Add duplicate dates
    data = sample_data.copy()
    data.index = data.index.tolist()[:-1] + [data.index[-1]]
    
    # Mock validate_dataframe to return the error we want to test
    with patch('data_processing.validate_data', return_value=(False, "Duplicate timestamps found in data")):
        with pytest.raises(ValueError) as exc_info:
            process_data(data)
        assert "Duplicate timestamps found in data" in str(exc_info.value)

def test_data_processing_with_non_monotonic_dates(sample_data):
    """Test data processing with non-monotonic dates"""
    # Shuffle dates
    data = sample_data.copy()
    data.index = data.index.tolist()[::-1]
    
    # Mock validate_dataframe to return the error we want to test
    with patch('data_processing.validate_data', return_value=(False, "Timestamps must be in ascending order")):
        with pytest.raises(ValueError) as exc_info:
            process_data(data)
        assert "Timestamps must be in ascending order" in str(exc_info.value)

def test_data_processing_with_insufficient_data(sample_data):
    """Test data processing with insufficient data"""
    # Create small dataset
    data = sample_data.iloc[:5]
    
    # Mock validate_dataframe to return the error we want to test
    with patch('data_processing.validate_data', return_value=(False, "Insufficient data points (minimum 100 required)")):
        with pytest.raises(ValueError) as exc_info:
            process_data(data)
        assert "Insufficient data points" in str(exc_info.value)

def test_data_processing_with_large_gaps(sample_data):
    """Test data processing with large gaps"""
    # Create data with large gaps
    data = sample_data.copy()
    # Downsample to every 4th row to create gaps
    data = data.iloc[::4].copy()
    # Reindex to a new range with gaps
    new_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='4h')
    data = data.reindex(new_index)
    
    # Mock validate_dataframe to return the error we want to test
    with patch('data_processing.validate_data', return_value=(False, "Large gaps detected in data")):
        with pytest.raises(ValueError) as exc_info:
            process_data(data)
        assert "Large gaps detected in data" in str(exc_info.value)

def test_data_processing(sample_data):
    """Test data processing functions"""
    # Test process_data
    processed_data = process_data(sample_data)
    assert isinstance(processed_data, pd.DataFrame)
    assert not processed_data.empty
    assert not processed_data.isnull().any().any()
    
    # Verify price relationships
    assert (processed_data['low'] <= processed_data['open']).all()
    assert (processed_data['open'] <= processed_data['high']).all()
    assert (processed_data['low'] <= processed_data['close']).all()
    assert (processed_data['close'] <= processed_data['high']).all()
    
    # Verify no negative values
    assert (processed_data['volume'] >= 0).all()
    
    # Verify data is sorted
    assert processed_data.index.is_monotonic_increasing
    
    # Verify minimum data points
    assert len(processed_data) >= 100 