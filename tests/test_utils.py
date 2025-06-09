import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Generator, cast
import json
import os
import time

# Import the module to test
import utils
import config as app_config

@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def sample_param_dicts() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Create sample parameter dictionaries for testing."""
    dict1 = {
        'period': 14,
        'fast': 12.0,
        'slow': 26.0,
        'signal': 9,
        'threshold': 0.5
    }
    dict2 = {
        'period': 14,
        'fast': 12.0000001,  # Slightly different float
        'slow': 26.0,
        'signal': 9,
        'threshold': 0.5
    }
    return dict1, dict2

@pytest.fixture(scope="function")
def mock_config_paths(temp_dir: Path, monkeypatch) -> None:
    """Mock config paths to use temp directory for testing."""
    # Create subdirectories in temp_dir
    reports_dir = temp_dir / "reports"
    logs_dir = temp_dir / "logs"
    db_dir = temp_dir / "database"
    for dir_path in [reports_dir, logs_dir, db_dir]:
        dir_path.mkdir()
    
    # Mock the config paths
    monkeypatch.setattr(app_config, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(app_config, "LOG_DIR", logs_dir)
    monkeypatch.setattr(app_config, "DB_DIR", db_dir)
    monkeypatch.setattr(app_config, "HEATMAPS_DIR", reports_dir / "heatmaps")
    monkeypatch.setattr(app_config, "LINE_CHARTS_DIR", reports_dir / "line_charts")
    monkeypatch.setattr(app_config, "COMBINED_CHARTS_DIR", reports_dir / "combined_charts")
    monkeypatch.setattr(app_config, "PROJECT_ROOT", temp_dir)
    monkeypatch.setattr(app_config, "LEADERBOARD_DB_PATH", db_dir / "correlation_leaderboard.db")
    monkeypatch.setattr(app_config, "INDICATOR_PARAMS_PATH", temp_dir / "indicator_params.json")

def test_cleanup_previous_content(temp_dir: Path, mock_config_paths: None) -> None:
    """Test cleanup of previous content."""
    # Create test files inside the subdirectories that will be cleaned
    test_files = [
        app_config.REPORTS_DIR / "test1.txt",
        app_config.LOG_DIR / "test2.log",
        app_config.DB_DIR / "test3.db"
    ]
    test_dirs = [
        app_config.REPORTS_DIR / "subdir1",
        app_config.LOG_DIR / "subdir2",
        app_config.DB_DIR / "subdir3"
    ]
    
    # Create files and directories
    for file in test_files:
        file.write_text("test content")
    for dir_path in test_dirs:
        dir_path.mkdir()
        (dir_path / "test.txt").write_text("test content")
    
    # Test cleanup with different options
    utils.cleanup_previous_content(
        clean_reports=True,
        clean_logs=True,
        clean_db=True,
        exclude_files=["test3.db"]  # Exclude this file
    )
    
    # Verify cleanup results
    assert not (app_config.REPORTS_DIR / "test1.txt").exists()
    assert not (app_config.LOG_DIR / "test2.log").exists()
    assert (app_config.DB_DIR / "test3.db").exists()  # Should be excluded
    assert not (app_config.REPORTS_DIR / "subdir1").exists()
    assert not (app_config.LOG_DIR / "subdir2").exists()
    assert not (app_config.DB_DIR / "subdir3").exists()

def test_parse_indicator_column_name() -> None:
    """Test parsing of indicator column names."""
    # Test valid column names
    assert utils.parse_indicator_column_name("RSI_123") == ("RSI", 123, None)
    assert utils.parse_indicator_column_name("MACD_456_FASTK") == ("MACD", 456, "FASTK")
    assert utils.parse_indicator_column_name("ADX_789_SIGNAL") == ("ADX", 789, "SIGNAL")
    
    # Test invalid column names
    assert utils.parse_indicator_column_name("RSI") is None
    assert utils.parse_indicator_column_name("RSI_ABC") is None
    assert utils.parse_indicator_column_name("RSI_123_") is None
    assert utils.parse_indicator_column_name("_123_FASTK") is None

def test_get_config_identifier() -> None:
    """Test generation of config identifiers."""
    # Test basic identifier
    assert utils.get_config_identifier("RSI", 123, None) == "RSI_123"
    
    # Test with suffix
    assert utils.get_config_identifier("MACD", 456, "FASTK") == "MACD_456_FASTK"
    
    # Test with spaces and special characters
    assert utils.get_config_identifier("RSI ", 123, " FAST K ") == "RSI_123_FAST_K"
    
    # Test with numeric suffix
    assert utils.get_config_identifier("ADX", 789, 1) == "ADX_789_1"

def test_round_floats_for_hashing() -> None:
    """Test float rounding for consistent hashing."""
    # Test simple float
    assert utils.round_floats_for_hashing(3.14159265359) == 3.14159265
    
    # Test nested structures
    test_data = {
        'float1': 3.14159265359,
        'list1': [1.23456789, 2.34567890],
        'dict1': {
            'float2': 4.56789012,
            'tuple1': (5.67890123, 6.78901234)
        }
    }
    expected = {
        'float1': 3.14159265,
        'list1': [1.23456789, 2.34567890],
        'dict1': {
            'float2': 4.56789012,
            'tuple1': (5.67890123, 6.78901234)
        }
    }
    assert utils.round_floats_for_hashing(test_data) == expected
    
    # Test non-float values
    assert utils.round_floats_for_hashing("string") == "string"
    assert utils.round_floats_for_hashing(42) == 42
    assert utils.round_floats_for_hashing(None) is None

def test_compare_param_dicts(sample_param_dicts: Tuple[Dict[str, Any], Dict[str, Any]]) -> None:
    """Test parameter dictionary comparison."""
    dict1, dict2 = sample_param_dicts
    
    # Test identical dictionaries
    assert utils.compare_param_dicts(dict1, dict1) is True
    
    # Test dictionaries with slightly different floats
    assert utils.compare_param_dicts(dict1, dict2) is True  # Should be equal due to float tolerance
    
    # Test with None values
    assert utils.compare_param_dicts(None, None) is True
    assert utils.compare_param_dicts(dict1, None) is False
    assert utils.compare_param_dicts(None, dict2) is False
    
    # Test with invalid types
    invalid_dict = cast(Dict[str, Any], "not a dict")
    assert utils.compare_param_dicts(invalid_dict, dict1) is False
    assert utils.compare_param_dicts(dict1, invalid_dict) is False
    
    # Test with different keys
    dict3 = dict1.copy()
    dict3['new_key'] = 42
    assert utils.compare_param_dicts(dict1, dict3) is False
    
    # Test with numpy types
    dict4 = dict1.copy()
    dict4['period'] = np.int32(14)
    assert utils.compare_param_dicts(dict1, dict4) is True

def test_estimate_price_precision() -> None:
    """Test price precision estimation."""
    # Test various price ranges
    assert utils.estimate_price_precision(12345.67) == 1  # >= 10000
    assert utils.estimate_price_precision(1234.56) == 2   # >= 1000
    assert utils.estimate_price_precision(123.45) == 3    # >= 10
    assert utils.estimate_price_precision(1.2345) == 4    # >= 1
    assert utils.estimate_price_precision(0.01234) == 6   # >= 0.01
    assert utils.estimate_price_precision(0.000123) == 7  # >= 0.0001
    assert utils.estimate_price_precision(0.00000123) == 8  # < 0.0001
    
    # Test edge cases
    assert utils.estimate_price_precision(0) == 4
    assert utils.estimate_price_precision(-123.45) == 3
    assert utils.estimate_price_precision(float('inf')) == 4
    assert utils.estimate_price_precision(float('nan')) == 4

def test_timeframe_calculations() -> None:
    """Test timeframe-related calculations."""
    # Test _get_seconds_per_period
    assert utils._get_seconds_per_period('1m') == 60.0
    assert utils._get_seconds_per_period('5m') == 300.0
    assert utils._get_seconds_per_period('1h') == 3600.0
    assert utils._get_seconds_per_period('1d') == 86400.0
    assert utils._get_seconds_per_period('1w') == 604800.0
    assert utils._get_seconds_per_period('1y') == 31536000.0
    assert utils._get_seconds_per_period('1M') is None  # Variable length
    assert utils._get_seconds_per_period('invalid') is None
    
    # Test calculate_periods_between_dates
    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2024, 1, 2, tzinfo=timezone.utc)
    
    assert utils.calculate_periods_between_dates(start_date, end_date, '1h') == 24
    assert utils.calculate_periods_between_dates(start_date, end_date, '4h') == 6
    assert utils.calculate_periods_between_dates(start_date, end_date, '1d') == 1
    
    # Test estimate_days_in_periods
    assert utils.estimate_days_in_periods(24, '1h') == 1.0
    assert utils.estimate_days_in_periods(7, '1d') == 7.0
    assert utils.estimate_days_in_periods(1, '1M') == 30.436875  # Average days per month
    
    # Test estimate_future_date
    future_date = utils.estimate_future_date(start_date, 24, '1h')
    assert future_date == datetime(2024, 1, 2, tzinfo=timezone.utc)
    
    future_date = utils.estimate_future_date(start_date, 7, '1d')
    assert future_date == datetime(2024, 1, 8, tzinfo=timezone.utc)
    
    # Test invalid inputs
    assert utils.calculate_periods_between_dates(end_date, start_date, '1h') == 0  # End before start
    assert utils.estimate_days_in_periods(-1, '1h') is None  # Negative periods
    assert utils.estimate_future_date(start_date, -1, '1h') is None  # Negative periods

def test_estimate_duration() -> None:
    """Test duration estimation for different analysis paths."""
    # Test tweak path (Bayesian Optimization)
    duration = utils.estimate_duration(
        num_configs_or_indicators=5,  # 5 indicators
        max_lag=10,
        path_type='tweak'
    )
    assert isinstance(duration, timedelta)
    assert duration.total_seconds() > 0
    
    # Test classical path
    duration = utils.estimate_duration(
        num_configs_or_indicators=100,  # 100 configs
        max_lag=10,
        path_type='classical'
    )
    assert isinstance(duration, timedelta)
    assert duration.total_seconds() > 0
    
    # Test edge cases
    duration = utils.estimate_duration(0, 0, 'tweak')
    assert isinstance(duration, timedelta)
    assert duration.total_seconds() > 0  # Should still have base time

def test_run_interim_reports(temp_dir: Path, monkeypatch) -> None:
    """Test interim report generation."""
    # Patch output directories to temp_dir
    monkeypatch.setattr(app_config, "REPORTS_DIR", temp_dir)
    monkeypatch.setattr(app_config, "HEATMAPS_DIR", temp_dir)
    monkeypatch.setattr(app_config, "LINE_CHARTS_DIR", temp_dir)
    monkeypatch.setattr(app_config, "COMBINED_CHARTS_DIR", temp_dir)

    # Create test database and data
    db_path = temp_dir / "test.db"
    symbol_id = 1
    timeframe_id = 1
    max_lag = 5
    
    # Create sample configs
    configs = [
        {'config_id': 1, 'indicator_name': 'RSI', 'params': {'period': 14}},
        {'config_id': 2, 'indicator_name': 'MACD', 'params': {'fast': 12, 'slow': 26}}
    ]
    
    # Create sample correlation data with Optional[float]
    correlation_data: Dict[int, List[Optional[float]]] = {
        1: [0.5, 0.4, 0.3, 0.2, 0.1],
        2: [0.6, 0.5, 0.4, 0.3, 0.2]
    }
    
    # Test report generation with provided data
    utils.run_interim_reports(
        db_path=db_path,
        symbol_id=symbol_id,
        timeframe_id=timeframe_id,
        configs_for_report=configs,
        max_lag=max_lag,
        file_prefix="test",
        stage_name="Test",
        correlation_data=correlation_data
    )
    
    # Verify report files were created
    report_files = list(temp_dir.glob("test_TEST_*.txt"))
    assert len(report_files) > 0
    
    # Test with empty correlation data
    utils.run_interim_reports(
        db_path=db_path,
        symbol_id=symbol_id,
        timeframe_id=timeframe_id,
        configs_for_report=[],
        max_lag=max_lag,
        file_prefix="test_empty",
        stage_name="Empty",
        correlation_data={}
    )
    
    # Verify no report files were created for empty data
    empty_report_files = list(temp_dir.glob("test_empty_EMPTY_*.txt"))
    assert len(empty_report_files) == 0

def test_safe_divide():
    assert utils.safe_divide(10, 2) == 5
    assert utils.safe_divide(10, 0) == 0
    assert utils.safe_divide(0, 10) == 0
    assert utils.safe_divide(-10, 2) == -5
    assert utils.safe_divide(10, -2) == -5

def test_rolling_apply():
    arr = np.arange(10)
    result = utils.rolling_apply(arr, 3, np.mean)
    assert np.allclose(result[2:], [1, 2, 3, 4, 5, 6, 7, 8])
    assert np.isnan(result[0]) and np.isnan(result[1])
    # Edge case: window larger than array
    arr = np.arange(2)
    result = utils.rolling_apply(arr, 3, np.mean)
    assert np.all(np.isnan(result))

def test_flatten_dict():
    d = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
    flat = utils.flatten_dict(d)
    assert flat["a.b"] == 1
    assert flat["a.c.d"] == 2
    assert flat["e"] == 3

def test_dict_hash():
    d1 = {"a": 1, "b": 2}
    d2 = {"b": 2, "a": 1}
    assert utils.dict_hash(d1) == utils.dict_hash(d2)
    d3 = {"a": 1, "b": 3}
    assert utils.dict_hash(d1) != utils.dict_hash(d3)

def test_chunks():
    l = list(range(10))
    c = list(utils.chunks(l, 3))
    assert c[0] == [0, 1, 2]
    assert c[-1] == [9]
    # Edge case: chunk size larger than list
    c = list(utils.chunks(l, 20))
    assert c[0] == l

def test_is_number():
    assert utils.is_number(1)
    assert utils.is_number(1.5)
    assert utils.is_number("1")
    assert not utils.is_number("a")
    assert not utils.is_number(None)

def test_parse_timeframe():
    assert utils.parse_timeframe("1h") == 3600
    assert utils.parse_timeframe("1d") == 86400
    assert utils.parse_timeframe("15m") == 900
    with pytest.raises(ValueError):
        utils.parse_timeframe("bad")

def test_format_timedelta():
    assert utils.format_timedelta(3661) == "1h 1m 1s"
    assert utils.format_timedelta(59) == "59s"
    assert utils.format_timedelta(3600) == "1h"

def test_human_readable_size():
    assert utils.human_readable_size(1023) == "1023.0 B"
    assert utils.human_readable_size(1024) == "1.0 KB"
    assert utils.human_readable_size(1024**2) == "1.0 MB"
    assert utils.human_readable_size(1024**3) == "1.0 GB"

def test_ensure_dir(tmp_path):
    d = tmp_path / "subdir"
    utils.ensure_dir(d)
    assert d.exists() and d.is_dir()
    # Should not raise if already exists
    utils.ensure_dir(d)

def test_retry():
    calls = {"count": 0}
    
    @utils.retry(tries=3, delay=0.01)
    def flaky():
        calls["count"] += 1
        if calls["count"] < 2:
            raise ValueError("fail")
        return 42
    
    assert flaky() == 42
    
    # Should raise after max tries
    calls["count"] = 0
    
    @utils.retry(tries=2, delay=0.01)
    def always_fail():
        raise ValueError("fail")
    
    with pytest.raises(ValueError):
        always_fail()

def test_timer():
    t = utils.Timer()
    t.start()
    time.sleep(0.01)
    elapsed = t.stop()
    assert elapsed > 0
    # Timer context manager
    with utils.Timer() as timer:
        time.sleep(0.01)
    assert timer.elapsed > 0 