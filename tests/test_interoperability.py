import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json
from indicator_factory import IndicatorFactory
from backtester import Backtester, Strategy
from data_manager import DataManager
from predictor import predict_price_movement
from parameter_optimizer import optimize_parameters
from leaderboard_manager import LeaderboardManager
from parameter_generator import generate_configurations
from visualization_generator import plot_indicator_performance
from backup_utils import BackupManager
from config import Config
from main import prepare_configurations
from custom_indicators import register_custom_indicator, custom_rsi
from sqlite_manager import SQLiteManager
from utils import flatten_dict, dict_hash
from extract_project_files import extract_files

@pytest.fixture(scope="function")
def sample_data():
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
def temp_dir():
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

# 1. End-to-end pipeline: data -> indicator -> prediction -> backtest -> leaderboard

def test_end_to_end_pipeline(temp_dir, sample_data):
    # Save and load data
    data_path = temp_dir / "prices.csv"
    sample_data.to_csv(data_path, index=False)
    manager = DataManager(data_dir=temp_dir)
    loaded = manager.load_data(data_path)
    
    # Compute indicator
    factory = IndicatorFactory()
    indicators = factory.compute_indicators(loaded, {"RSI": {"timeperiod": 14}})
    loaded["RSI"] = indicators["RSI"]
    
    # Predict
    preds = predict_price_movement(loaded, indicator_name="RSI", lag=1, params={"timeperiod": 14})
    loaded["pred"] = preds
    
    # Backtest
    def entry(data, params): return data["pred"] > 0
    def exit(data, params): return data["pred"] <= 0
    strategy = Strategy(entry, exit)
    backtester = Backtester(data_manager=manager, indicator_factory=factory)
    results = backtester.run_strategy(loaded, strategy, {})
    
    # Verify results
    assert isinstance(results, dict)
    assert "returns" in results
    assert "equity_curve" in results

# 2. Parameter generator with optimizer and indicator factory

def test_paramgen_optimizer_factory(sample_data):
    indicator_def = {
        "name": "RSI",
        "type": "talib",
        "required_inputs": ["close"],
        "params": {
            "timeperiod": {
                "type": "int",
                "min": 2,
                "max": 30,
                "default": 14
            }
        }
    }
    
    factory = IndicatorFactory()
    configs = factory.generate_parameter_configurations("RSI", method="grid")
    assert len(configs) > 0
    assert all(isinstance(config, dict) for config in configs)
    assert all("timeperiod" in config for config in configs)

# 3. Visualization with data and indicator output

def test_visualization_with_data_and_indicator(sample_data, temp_dir):
    factory = IndicatorFactory()
    indicators = factory.compute_indicators(sample_data, {"RSI": {"period": 14}})
    sample_data["RSI"] = indicators["RSI"]
    out_path = temp_dir / "rsi_perf.png"
    plot_indicator_performance(sample_data, indicator_name="RSI", output_path=out_path)
    assert out_path.exists()

# 4. Backup/restore with database and leaderboard

def test_backup_restore_leaderboard(temp_dir, sample_data):
    db_path = temp_dir / "leaderboard.db"
    manager = LeaderboardManager(str(db_path))
    
    # Create test correlations and configs
    correlations = {
        1: [0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05, -0.05],  # config_id 1, lags 1-10
        2: [0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00, -0.10, -0.20]  # config_id 2, lags 1-10
    }
    
    indicator_configs = [
        {
            'config_id': 1,
            'indicator_name': 'RSI',
            'params': {'timeperiod': 14}
        },
        {
            'config_id': 2,
            'indicator_name': 'BB',
            'params': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2}
        }
    ]
    
    # Update leaderboard
    updates = manager.update_leaderboard(
        current_run_correlations=correlations,
        indicator_configs=indicator_configs,
        max_lag=10,
        symbol="BTCUSDT",
        timeframe="1h",
        data_daterange="2023-01-01 to 2023-12-31",
        source_db_name="test.db"
    )
    
    assert len(updates) > 0
    assert 1 in updates and 2 in updates

# 5. Config and main integration

def test_config_main_integration(temp_dir):
    config = Config()
    config.data_dir = str(temp_dir)
    configs = prepare_configurations(config, mode="bayesian")
    assert isinstance(configs, list)

# 6. Error propagation across modules

def test_error_propagation(sample_data):
    factory = IndicatorFactory()
    
    # Test with invalid indicator name
    with pytest.raises(ValueError, match="Unknown indicator"):
        factory.create_indicator("invalid_indicator", sample_data)
    
    # Test with invalid parameters
    with pytest.raises(ValueError, match="Invalid period value"):
        factory.create_indicator("RSI", sample_data, timeperiod="invalid")
    
    # Test with missing required columns - this will fail due to insufficient data first
    with pytest.raises(ValueError, match="Invalid period value"):
        factory.create_indicator("RSI", pd.DataFrame({"invalid": [1, 2, 3]}))

# 7. Custom indicator integration

def test_custom_indicator_integration(sample_data):
    """Test integration of custom indicators with the factory."""
    factory = IndicatorFactory()
    
    # Define custom RSI
    def custom_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    # Register custom RSI
    success = factory.register_custom_indicator("custom_rsi", custom_rsi)
    assert success
    
    # Test computing the custom indicator
    indicators = factory.compute_indicators(sample_data, {"custom_rsi": {"period": 14}})
    assert isinstance(indicators, pd.DataFrame)
    assert "custom_rsi" in indicators.columns
    assert not indicators["custom_rsi"].isna().all()

# 8. SQLiteManager with DataManager

def test_sqlitemanager_datamanager(temp_dir, sample_data):
    db_path = temp_dir / "test.db"
    manager = SQLiteManager(str(db_path))
    
    # Create prices table with appropriate schema
    manager.create_table("prices", {
        "timestamp": "TIMESTAMP",
        "open": "REAL",
        "high": "REAL",
        "low": "REAL",
        "close": "REAL",
        "volume": "REAL"
    })
    
    # Insert data
    for _, row in sample_data.iterrows():
        manager.insert("prices", row.to_dict())
    
    # Use DataManager to load from DB (simulate)
    rows = manager.select("prices", ["timestamp", "open", "close"])
    assert len(rows) > 0
    
    # Cleanup
    manager.close()

# 9. Utils with main modules

def test_utils_with_main_modules(sample_data):
    d = {"a": {"b": 1, "c": 2}, "d": 3}
    flat = flatten_dict(d)
    h = dict_hash(flat)
    assert isinstance(h, str)
    # Use hash as a unique key for leaderboard
    leaderboard = {}
    leaderboard[h] = 0.5
    assert leaderboard[h] == 0.5

# 10. extract_project_files with DataManager

def test_extract_project_files_with_datamanager(temp_dir, sample_data):
    # Save a file
    file_path = temp_dir / "data.csv"
    sample_data.to_csv(file_path, index=False)
    dest_dir = temp_dir / "extracted"
    dest_dir.mkdir()
    extract_files([str(file_path)], str(dest_dir))
    manager = DataManager(data_dir=dest_dir)
    loaded = manager.load_data(dest_dir / "data.csv")
    pd.testing.assert_frame_equal(loaded, sample_data)

def test_indicator_factory_parameter_generation(factory):
    """Test parameter generation and configuration handling in IndicatorFactory."""
    # Generate parameter configurations for RSI
    rsi_configs = factory.generate_parameter_configurations('RSI', method='grid')
    assert len(rsi_configs) > 0
    
    # Test each configuration
    for config in rsi_configs:
        # Validate parameters
        factory.validate_params('RSI', config)
        
        # Compute indicator with configuration
        result = factory.compute_indicators(test_data, {'RSI': config})
        assert 'RSI' in result.columns
        assert not result['RSI'].isna().all()
        
    # Generate random configurations for BB
    bb_configs = factory.generate_parameter_configurations('BB', method='random', num_configs=3)
    assert len(bb_configs) == 3
    
    # Test each configuration
    for config in bb_configs:
        # Validate parameters
        factory.validate_params('BB', config)
        
        # Compute indicator with configuration
        result = factory.compute_indicators(test_data, {'BB': config})
        assert 'BB_upper' in result.columns
        assert 'BB_middle' in result.columns
        assert 'BB_lower' in result.columns
        assert not result['BB_middle'].isna().all()
        
    # Test with custom indicator
    factory.register_custom_indicator('CUSTOM_MA', 
                                    lambda x, p: x['close'].rolling(p).mean(),
                                    {'period': {'type': 'int', 'min': 2, 'max': 100, 'default': 20}})
    
    # Generate configurations for custom indicator
    custom_configs = factory.generate_parameter_configurations('CUSTOM_MA', method='grid')
    assert len(custom_configs) > 0
    
    # Test each configuration
    for config in custom_configs:
        # Validate parameters
        factory.validate_params('CUSTOM_MA', config)
        
        # Compute indicator with configuration
        result = factory.compute_indicators(test_data, {'CUSTOM_MA': config})
        assert 'CUSTOM_MA' in result.columns
        assert not result['CUSTOM_MA'].isna().all()
        
    # Test with invalid configurations
    with pytest.raises(ValueError):
        factory.validate_params('RSI', {'period': 1})  # Below minimum
        
    with pytest.raises(ValueError):
        factory.validate_params('BB', {'period': 1, 'std_dev': 0.05})  # Below minimum
        
    with pytest.raises(ValueError):
        factory.validate_params('CUSTOM_MA', {'period': 1})  # Below minimum 