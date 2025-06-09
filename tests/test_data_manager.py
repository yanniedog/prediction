import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_manager import _fetch_klines, _process_klines, _save_to_sqlite
from pathlib import Path
import sqlite3
import os
import tempfile
import shutil
import json
from data_manager import (
    DataManager,
    _load_csv,
    _save_csv,
    _validate_data,
    _merge_data,
    _filter_data,
    _resample_data,
    _fill_missing_data,
    _split_data
)

@pytest.fixture(scope="function")
def temp_dir() -> Path:
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def sample_data() -> pd.DataFrame:
    dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
    np.random.seed(42)
    data = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.normal(100, 1, 100),
        "high": np.random.normal(101, 1, 100),
        "low": np.random.normal(99, 1, 100),
        "close": np.random.normal(100, 1, 100),
        "volume": np.random.normal(1000, 100, 100)
    })
    data["high"] = data[["open", "close"]].max(axis=1) + abs(np.random.normal(0, 0.1, 100))
    data["low"] = data[["open", "close"]].min(axis=1) - abs(np.random.normal(0, 0.1, 100))
    return data

@pytest.fixture(scope="function")
def data_manager(temp_dir: Path):
    # DataManager does not accept data_dir, so use config or default
    import config as app_config
    return DataManager(app_config)

@pytest.fixture
def test_data():
    """Create test data with proper time intervals."""
    dates = pd.date_range(start='2020-01-01', end='2020-03-01', freq='1h')
    data = pd.DataFrame({
        'open': np.random.uniform(1000, 6000, len(dates)),
        'high': np.random.uniform(1000, 6000, len(dates)),
        'low': np.random.uniform(1000, 6000, len(dates)),
        'close': np.random.uniform(1000, 6000, len(dates)),
        'volume': np.random.uniform(1000, 10000, len(dates))
    }, index=dates)
    
    # Ensure high is always >= low
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    return data

def test_data_manager_initialization(temp_dir: Path):
    manager = DataManager(data_dir=temp_dir)
    assert manager.data_dir == temp_dir
    assert temp_dir.exists()
    with pytest.raises(ValueError):
        DataManager(data_dir="/invalid/path")

def test_load_and_save_csv(data_manager: DataManager, sample_data: pd.DataFrame, temp_dir: Path):
    csv_path = temp_dir / "test.csv"
    _save_csv(sample_data, csv_path)
    assert csv_path.exists()
    loaded = _load_csv(csv_path)
    pd.testing.assert_frame_equal(loaded, sample_data)
    with pytest.raises(ValueError):
        _load_csv(temp_dir / "nonexistent.csv")

def test_validate_data(sample_data: pd.DataFrame):
    assert _validate_data(sample_data)
    invalid = sample_data.drop(columns=["close"])
    with pytest.raises(ValueError):
        _validate_data(invalid)
    nan_data = sample_data.copy()
    nan_data.loc[0, "close"] = np.nan
    with pytest.raises(ValueError):
        _validate_data(nan_data)

def test_merge_data(sample_data: pd.DataFrame):
    merged = _merge_data([sample_data, sample_data])
    assert isinstance(merged, pd.DataFrame)
    assert len(merged) == 2 * len(sample_data)
    with pytest.raises(ValueError):
        _merge_data([])

def test_filter_data(sample_data: pd.DataFrame):
    filtered = _filter_data(sample_data, start=sample_data["timestamp"][10], end=sample_data["timestamp"][20])
    assert isinstance(filtered, pd.DataFrame)
    assert filtered["timestamp"].min() >= sample_data["timestamp"][10]
    assert filtered["timestamp"].max() <= sample_data["timestamp"][20]
    with pytest.raises(ValueError):
        _filter_data(sample_data, start="2100-01-01", end="2100-01-02")

def test_resample_data(sample_data: pd.DataFrame):
    resampled = _resample_data(sample_data, rule="D")
    assert isinstance(resampled, pd.DataFrame)
    assert len(resampled) < len(sample_data)
    with pytest.raises(ValueError):
        _resample_data(sample_data, rule="invalid")

def test_fill_missing_data(sample_data: pd.DataFrame):
    data = sample_data.copy()
    data.loc[0, "close"] = np.nan
    filled = _fill_missing_data(data)
    assert not filled["close"].isna().any()
    with pytest.raises(ValueError):
        _fill_missing_data(pd.DataFrame())

def test_normalize_data(data_manager: DataManager, sample_data: pd.DataFrame):
    normalized = data_manager.normalize_data(sample_data, columns=["open", "close"])
    assert np.allclose(normalized[["open", "close"]].mean(), 0, atol=1e-1)
    assert np.allclose(normalized[["open", "close"]].std(), 1, atol=1e-1)
    with pytest.raises(ValueError):
        data_manager.normalize_data(sample_data, columns=["nonexistent"])

def test_split_data(sample_data: pd.DataFrame):
    train, test = _split_data(sample_data, test_size=0.2)
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert len(train) + len(test) == len(sample_data)
    with pytest.raises(ValueError):
        _split_data(sample_data, test_size=1.5)

def test_data_manager_methods(data_manager: DataManager, sample_data: pd.DataFrame, temp_dir: Path):
    # Save and load
    path = temp_dir / "data.csv"
    data_manager.save_data(sample_data, path)
    loaded = data_manager.load_data(path)
    pd.testing.assert_frame_equal(loaded, sample_data)
    # Validate
    assert data_manager.validate_data(sample_data)
    # Merge
    merged = data_manager.merge_data([sample_data, sample_data])
    assert len(merged) == 2 * len(sample_data)
    # Filter
    filtered = data_manager.filter_data(sample_data, start=sample_data["timestamp"][10], end=sample_data["timestamp"][20])
    assert filtered["timestamp"].min() >= sample_data["timestamp"][10]
    # Resample
    resampled = data_manager.resample_data(sample_data, rule="D")
    assert len(resampled) < len(sample_data)
    # Fill missing
    data = sample_data.copy()
    data.loc[0, "close"] = np.nan
    filled = data_manager.fill_missing_data(data)
    assert not filled["close"].isna().any()
    # Normalize
    normalized = data_manager.normalize_data(sample_data, columns=["open", "close"])
    assert np.allclose(normalized[["open", "close"]].mean(), 0, atol=1e-1)
    # Split
    train, test = data_manager.split_data(sample_data, test_size=0.2)
    assert len(train) + len(test) == len(sample_data)

def test_error_handling(data_manager: DataManager, temp_dir: Path):
    with pytest.raises(ValueError):
        data_manager.load_data(temp_dir / "nonexistent.csv")
    with pytest.raises(ValueError):
        data_manager.save_data(pd.DataFrame(), temp_dir / "empty.csv")
    with pytest.raises(ValueError):
        data_manager.validate_data(pd.DataFrame())
    with pytest.raises(ValueError):
        data_manager.merge_data([])
    with pytest.raises(ValueError):
        data_manager.filter_data(pd.DataFrame(), start="2100-01-01", end="2100-01-02")
    with pytest.raises(ValueError):
        data_manager.resample_data(pd.DataFrame(), rule="D")
    with pytest.raises(ValueError):
        data_manager.fill_missing_data(pd.DataFrame())
    with pytest.raises(ValueError):
        data_manager.normalize_data(pd.DataFrame(), columns=["open"])
    with pytest.raises(ValueError):
        data_manager.split_data(pd.DataFrame(), test_size=0.2)

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
    with pytest.raises(ValueError) as exc_info:
        data_manager.validate_data(invalid_data)
    assert "NaN values found in column: close" in str(exc_info.value)

    # Test missing columns
    invalid_data = test_data.drop('volume', axis=1)
    with pytest.raises(ValueError) as exc_info:
        data_manager.validate_data(invalid_data)
    assert "Missing required columns: ['volume']" in str(exc_info.value)

    # Test invalid high/low relationship
    invalid_data = test_data.copy()
    invalid_data.loc[invalid_data.index[0], 'high'] = invalid_data.loc[invalid_data.index[0], 'low'] - 1
    with pytest.raises(ValueError) as exc_info:
        data_manager.validate_data(invalid_data)
    assert "High price cannot be less than low price" in str(exc_info.value)

    # Test negative volume
    invalid_data = test_data.copy()
    invalid_data.loc[invalid_data.index[0], 'volume'] = -1
    with pytest.raises(ValueError) as exc_info:
        data_manager.validate_data(invalid_data)
    assert "Volume cannot be negative" in str(exc_info.value)

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
    
    # Check that numeric columns have mean=0 and std=1
    numeric_cols = normalized_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mean = normalized_data[col].mean()
        std = normalized_data[col].std()
        assert abs(mean) < 1e-10, f"Column {col} mean should be 0, got {mean}"
        assert abs(std - 1) < 1e-10, f"Column {col} std should be 1, got {std}"
    
    # Check that non-numeric columns are preserved
    non_numeric_cols = normalized_data.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        assert col in test_data.columns, f"Non-numeric column {col} should be preserved"
        pd.testing.assert_series_equal(normalized_data[col], test_data[col])

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