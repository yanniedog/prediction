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
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
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
    indicators = factory.compute_indicators(loaded, {"RSI": {"period": 14}})
    loaded["RSI"] = indicators["RSI"]
    # Predict
    preds = predict_price_movement(loaded, indicator_name="RSI", lag=1, params={"period": 14})
    loaded["pred"] = preds
    # Backtest
    def entry(data, params): return data["pred"] > 0
    def exit(data, params): return data["pred"] <= 0
    strategy = Strategy("PredStrategy", entry, exit, {})
    backtester = Backtester(loaded, strategy)
    results = backtester.run()
    # Leaderboard
    leaderboard_path = temp_dir / "leaderboard.db"
    manager_lb = LeaderboardManager(str(leaderboard_path))
    manager_lb.update_leaderboard_bulk([
        {"indicator": "RSI", "params": {"period": 14}, "correlation": results["returns"].mean()}
    ])
    best = manager_lb.find_best_predictor(lag=1)
    assert best is not None

# 2. Parameter generator with optimizer and indicator factory

def test_paramgen_optimizer_factory(sample_data):
    indicator_def = {
        "type": "momentum",
        "inputs": ["close"],
        "parameters": {"period": {"type": "int", "min": 2, "max": 30, "default": 14}}
    }
    configs = generate_configurations("RSI", indicator_def, method="bayesian")
    assert isinstance(configs, list) and configs
    best_params, best_score = optimize_parameters(
        data=sample_data,
        indicator_name="RSI",
        indicator_definition=indicator_def,
        method="bayesian"
    )
    factory = IndicatorFactory()
    indicators = factory.compute_indicators(sample_data, {"RSI": best_params})
    assert "RSI" in indicators.columns

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
    manager.update_leaderboard_bulk([
        {"indicator": "RSI", "params": {"period": 14}, "correlation": 0.5}
    ])
    backup_dir = temp_dir / "backups"
    backup_manager = BackupManager(backup_dir)
    backup_path = backup_manager.create_backup(db_path)
    restored_path = temp_dir / "restored_leaderboard.db"
    backup_manager.restore_backup(backup_path, restored_path)
    restored_manager = LeaderboardManager(str(restored_path))
    best = restored_manager.find_best_predictor(lag=1)
    assert best is not None

# 5. Config and main integration

def test_config_main_integration(temp_dir):
    config = Config()
    config.data_dir = str(temp_dir)
    configs = prepare_configurations(config, mode="bayesian")
    assert isinstance(configs, list)

# 6. Error propagation across modules

def test_error_propagation(sample_data):
    factory = IndicatorFactory()
    # Invalid indicator params
    with pytest.raises(Exception):
        factory.compute_indicators(sample_data, {"RSI": {"period": -1}})
    # Invalid prediction
    with pytest.raises(Exception):
        predict_price_movement(sample_data, indicator_name="NON_EXISTENT", lag=1, params={})

# 7. Custom indicator integration

def test_custom_indicator_integration(sample_data):
    register_custom_indicator("CUSTOM_RSI", custom_rsi)
    factory = IndicatorFactory()
    indicators = factory.compute_indicators(sample_data, {"CUSTOM_RSI": {"period": 14}})
    assert "CUSTOM_RSI" in indicators.columns

# 8. SQLiteManager with DataManager

def test_sqlitemanager_datamanager(temp_dir, sample_data):
    db_path = temp_dir / "test.db"
    manager = SQLiteManager(str(db_path))
    # Insert data
    for _, row in sample_data.iterrows():
        manager.insert("prices", row.to_dict())
    # Use DataManager to load from DB (simulate)
    # (Assume DataManager can load from DB if implemented, or just check DB content)
    rows = manager.select("prices", ["timestamp", "open", "close"])
    assert len(rows) > 0

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