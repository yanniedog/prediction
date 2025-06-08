import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_manager import _fetch_klines, _process_klines, _save_to_sqlite
from pathlib import Path
import sqlite3
import os

def test_data_manager_initialization(data_manager):
    """Test DataManager initialization."""
    assert data_manager is not None
    assert hasattr(data_manager, 'config')

def test_load_data(data_manager, test_data, tmp_path):
    """Test loading data from different sources."""
    # Test loading from DataFrame
    loaded_data = data_manager.load_data(test_data)
    assert isinstance(loaded_data, pd.DataFrame)
    assert not loaded_data.empty
    assert all(col in loaded_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    # Test loading from CSV
    csv_path = tmp_path / "test_data.csv"
    test_data.to_csv(csv_path)
    loaded_from_csv = data_manager.load_data(str(csv_path))
    assert isinstance(loaded_from_csv, pd.DataFrame)
    assert not loaded_from_csv.empty

def test_data_validation(data_manager, test_data):
    """Test data validation methods."""
    # Test valid data
    assert data_manager.validate_data(test_data)

    # Test invalid data
    invalid_data = test_data.copy()
    invalid_data.loc[invalid_data.index[0], 'close'] = np.nan
    assert not data_manager.validate_data(invalid_data)

def test_data_preprocessing(data_manager, test_data):
    """Test data preprocessing methods."""
    processed_data = data_manager.preprocess_data(test_data)
    assert isinstance(processed_data, pd.DataFrame)
    assert not processed_data.empty
    assert not processed_data.isnull().any().any()

def test_data_splitting(data_manager, test_data):
    """Test train/test data splitting."""
    original_length = len(test_data)
    train_data, test_data = data_manager.split_data(test_data, test_size=0.2)
    assert isinstance(train_data, pd.DataFrame)
    assert isinstance(test_data, pd.DataFrame)
    assert len(train_data) + len(test_data) == original_length  # Total length should equal original
    assert len(test_data) == pytest.approx(original_length * 0.2, rel=0.1)  # Test size should be ~20%

def test_feature_engineering(data_manager, test_data):
    """Test feature engineering methods."""
    features = data_manager.engineer_features(test_data)
    assert isinstance(features, pd.DataFrame)
    assert not features.empty
    assert not features.isnull().any().any()

def test_data_normalization(data_manager, test_data):
    """Test data normalization methods."""
    normalized_data = data_manager.normalize_data(test_data)
    assert isinstance(normalized_data, pd.DataFrame)
    assert not normalized_data.empty
    assert normalized_data.select_dtypes(include=[np.number]).max().max() <= 1
    assert normalized_data.select_dtypes(include=[np.number]).min().min() >= -1

def test_data_aggregation(data_manager, test_data):
    """Test data aggregation methods."""
    # Test daily to weekly aggregation
    weekly_data = data_manager.aggregate_data(test_data, 'W')
    assert isinstance(weekly_data, pd.DataFrame)
    assert not weekly_data.empty
    assert len(weekly_data) < len(test_data)

    # Test daily to monthly aggregation
    monthly_data = data_manager.aggregate_data(test_data, 'M')
    assert isinstance(monthly_data, pd.DataFrame)
    assert not monthly_data.empty
    assert len(monthly_data) < len(weekly_data)

def test_data_cleaning(data_manager, test_data):
    """Test data cleaning methods."""
    # Add some outliers
    dirty_data = test_data.copy()
    dirty_data.loc[dirty_data.index[0], 'close'] = dirty_data['close'].mean() * 10
    
    cleaned_data = data_manager.clean_data(dirty_data)
    assert isinstance(cleaned_data, pd.DataFrame)
    assert not cleaned_data.empty
    assert not cleaned_data.isnull().any().any()
    
    # Verify outliers were handled
    assert cleaned_data['close'].max() < dirty_data['close'].max()

def test_data_sampling(data_manager, test_data):
    """Test data sampling methods."""
    # Test random sampling
    sampled_data = data_manager.sample_data(test_data, n_samples=10)
    assert isinstance(sampled_data, pd.DataFrame)
    assert len(sampled_data) == 10

    # Test systematic sampling
    systematic_sample = data_manager.sample_data(test_data, method='systematic', step=5)
    assert isinstance(systematic_sample, pd.DataFrame)
    assert len(systematic_sample) == len(test_data) // 5 

def test_fetch_klines_handles_invalid_symbol(monkeypatch):
    # Simulate API returning empty for invalid symbol
    def mock_get(*args, **kwargs):
        class MockResponse:
            def raise_for_status(self): pass
            def json(self): return []
        return MockResponse()
    monkeypatch.setattr("requests.get", mock_get)
    result = _fetch_klines("INVALID", "1d", 0, 1000)
    assert result == []

def test_process_klines_handles_empty():
    df = _process_klines([])
    assert isinstance(df, pd.DataFrame)
    assert df.empty

def test_process_klines_handles_invalid_data():
    # Data with invalid open_time
    klines = [[None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    df = _process_klines(klines)
    assert df.empty

def test_save_to_sqlite_handles_empty(tmp_path):
    db_path = tmp_path / "test.db"
    df = pd.DataFrame()
    # Should return True (nothing to save, but not an error)
    assert _save_to_sqlite(df, str(db_path), "BTCUSDT", "1d")

def test_save_to_sqlite_creates_db(tmp_path):
    db_path = tmp_path / "test.db"
    # Use a valid open_time (after 2015-01-01)
    valid_open_time = int(datetime(2016, 1, 1).timestamp() * 1000)
    df = pd.DataFrame({
        "open_time": [valid_open_time], "open": [1], "high": [1], "low": [1], "close": [1], "volume": [1],
        "close_time": [valid_open_time], "quote_asset_volume": [1], "number_of_trades": [1],
        "taker_buy_base_asset_volume": [1], "taker_buy_quote_asset_volume": [1]
    })
    # Create required tables
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE symbols (id INTEGER PRIMARY KEY, symbol TEXT)")
    cur.execute("INSERT INTO symbols (symbol) VALUES ('BTCUSDT')")
    cur.execute("CREATE TABLE timeframes (id INTEGER PRIMARY KEY, timeframe TEXT)")
    cur.execute("INSERT INTO timeframes (timeframe) VALUES ('1d')")
    conn.commit()
    conn.close()
    assert _save_to_sqlite(df, str(db_path), "BTCUSDT", "1d")
    assert os.path.exists(db_path)
    # Optionally, check that at least one table exists
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cur.fetchall()]
    assert len(tables) > 0
    conn.close()

def test_fetch_klines_handles_api_error(monkeypatch):
    class MockResponse:
        def raise_for_status(self): raise Exception("API error")
        def json(self): return []
    def mock_get(*args, **kwargs): return MockResponse()
    monkeypatch.setattr("requests.get", mock_get)
    result = _fetch_klines("BTCUSDT", "1d", 0, 1000)
    assert result == []

def test_process_klines_filters_old_timestamps():
    # Timestamp before 2015
    klines = [[1000000000, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    df = _process_klines(klines)
    assert df.empty 