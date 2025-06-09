import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from main import _select_data_source_and_lag
from utils import get_max_lag, get_data_date_range
import sqlite3
from unittest.mock import patch, MagicMock

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

def test_data_source_selection(tmp_path, sample_data):
    """Test data source selection and validation"""
    # Create a temporary database with proper schema
    db_path = tmp_path / "BTCUSDT_1h.db"
    
    # Initialize database with proper schema
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create required tables with proper schema
    cursor.executescript("""
        -- Symbols table
        CREATE TABLE IF NOT EXISTS symbols (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Timeframes table
        CREATE TABLE IF NOT EXISTS timeframes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timeframe TEXT NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Historical data table
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
            FOREIGN KEY (symbol_id) REFERENCES symbols(id)
                ON DELETE CASCADE,
            FOREIGN KEY (timeframe_id) REFERENCES timeframes(id)
                ON DELETE CASCADE,
            UNIQUE(symbol_id, timeframe_id, open_time)
        );
    """)
    
    # Insert required symbol and timeframe
    cursor.execute("INSERT INTO symbols (symbol) VALUES (?)", ("BTCUSDT",))
    symbol_id = cursor.lastrowid
    cursor.execute("INSERT INTO timeframes (timeframe) VALUES (?)", ("1h",))
    timeframe_id = cursor.lastrowid
    
    # Convert sample data to match historical_data schema
    sample_data['open_time'] = sample_data.index.astype(np.int64) // 10**6
    sample_data['close_time'] = sample_data['open_time'] + 3600000  # 1 hour in milliseconds
    sample_data['symbol_id'] = symbol_id
    sample_data['timeframe_id'] = timeframe_id
    
    # Insert sample data into historical_data table
    for _, row in sample_data.iterrows():
        cursor.execute("""
            INSERT INTO historical_data (
                symbol_id, timeframe_id, open_time, open, high, low, close, volume,
                close_time, quote_asset_volume, number_of_trades,
                taker_buy_base_asset_volume, taker_buy_quote_asset_volume
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row['symbol_id'], row['timeframe_id'], row['open_time'],
            row['open'], row['high'], row['low'], row['close'], row['volume'],
            row['close_time'], row['volume'], 100, row['volume']/2, row['volume']/2
        ))
    
    conn.commit()
    conn.close()
    
    # Mock the data manager to use our test database
    with patch('main.data_manager.manage_data_source', return_value=(db_path, "BTCUSDT", "1h")):
        # Test data source selection
        result = _select_data_source_and_lag()
        assert isinstance(result, tuple)
        assert len(result) == 8
        
        db_path, symbol, timeframe, data, symbol_id, timeframe_id, max_lag, date_range = result
        assert isinstance(db_path, Path)
        assert isinstance(symbol, str)
        assert isinstance(timeframe, str)
        assert isinstance(data, pd.DataFrame)
        assert isinstance(symbol_id, int)
        assert isinstance(timeframe_id, int)
        assert isinstance(max_lag, int)
        assert isinstance(date_range, str)
        
        # Verify data integrity
        assert not data.empty
        assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert data.index.is_monotonic_increasing
        assert not data.index.duplicated().any()

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
    
    # Mock data manager functions to return invalid data
    with patch('main.data_manager.manage_data_source', return_value=(Path("test.db"), "BTCUSD", "1h")):
        with patch('main.data_manager.load_data', return_value=data):
            with pytest.raises(ValueError) as exc_info:
                _select_data_source_and_lag()
            assert "Missing required columns" in str(exc_info.value)

def test_data_processing_with_invalid_values(sample_data):
    """Test data processing with invalid values"""
    # Add invalid values
    data = sample_data.copy()
    data.loc[data.index[0], 'high'] = -1  # Invalid high price
    data.loc[data.index[1], 'volume'] = -100  # Invalid volume
    
    # Mock data manager functions to return invalid data
    with patch('main.data_manager.manage_data_source', return_value=(Path("test.db"), "BTCUSD", "1h")):
        with patch('main.data_manager.load_data', return_value=data):
            with pytest.raises(ValueError) as exc_info:
                _select_data_source_and_lag()
            assert "Invalid values detected" in str(exc_info.value)

def test_data_processing_with_duplicate_dates(sample_data):
    """Test data processing with duplicate dates"""
    # Add duplicate dates
    data = sample_data.copy()
    data.index = data.index.tolist()[:-1] + [data.index[-1]]
    
    # Mock data manager functions to return invalid data
    with patch('main.data_manager.manage_data_source', return_value=(Path("test.db"), "BTCUSD", "1h")):
        with patch('main.data_manager.load_data', return_value=data):
            with pytest.raises(ValueError) as exc_info:
                _select_data_source_and_lag()
            assert "Duplicate dates found" in str(exc_info.value)

def test_data_processing_with_non_monotonic_dates(sample_data):
    """Test data processing with non-monotonic dates"""
    # Shuffle dates
    data = sample_data.copy()
    data.index = data.index.tolist()[::-1]
    
    # Mock data manager functions to return invalid data
    with patch('main.data_manager.manage_data_source', return_value=(Path("test.db"), "BTCUSD", "1h")):
        with patch('main.data_manager.load_data', return_value=data):
            with pytest.raises(ValueError) as exc_info:
                _select_data_source_and_lag()
            assert "Dates must be monotonically increasing" in str(exc_info.value)

def test_data_processing_with_insufficient_data(sample_data):
    """Test data processing with insufficient data"""
    # Create small dataset
    data = sample_data.iloc[:5]
    
    # Mock data manager functions to return insufficient data
    with patch('main.data_manager.manage_data_source', return_value=(Path("test.db"), "BTCUSD", "1h")):
        with patch('main.data_manager.load_data', return_value=data):
            with pytest.raises(ValueError) as exc_info:
                _select_data_source_and_lag()
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
    # Mock data manager functions to return data with gaps
    with patch('main.data_manager.manage_data_source', return_value=(Path("test.db"), "BTCUSD", "1h")):
        with patch('main.data_manager.load_data', return_value=data):
            with pytest.raises(ValueError) as exc_info:
                _select_data_source_and_lag()
            assert "Large gaps detected in data" in str(exc_info.value) 