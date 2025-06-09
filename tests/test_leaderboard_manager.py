import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from leaderboard_manager import LeaderboardManager
import json
from unittest.mock import patch

@pytest.fixture(scope="function")
def temp_db_dir():
    """Create a temporary directory for test database."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def leaderboard_manager(temp_db_dir):
    """Create a LeaderboardManager instance with temporary database."""
    with patch('leaderboard_manager.LEADERBOARD_DB_PATH', temp_db_dir / "leaderboard.db"):
        manager = LeaderboardManager()
        yield manager

@pytest.fixture(scope="function")
def test_data():
    """Provide test data for leaderboard updates."""
    return {
        'symbol': 'BTCUSDT',
        'timeframe': '1h',
        'data_daterange': '2023-01-01 to 2023-12-31',
        'source_db_name': 'test.db'
    }

def test_leaderboard_initialization(leaderboard_manager):
    """Test leaderboard database initialization."""
    # Verify database was created
    assert leaderboard_manager.create_connection() is not None
    
    # Verify tables were created
    conn = leaderboard_manager.create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert 'leaderboard' in tables
    conn.close()

def test_update_single_lag(leaderboard_manager, test_data):
    """Test updating a single lag in the leaderboard."""
    # Test updating with valid data
    success = leaderboard_manager.check_and_update_single_lag(
        lag=5,
        correlation_value=0.85,
        indicator_name='RSI',
        params={'timeperiod': 14},
        config_id=1,
        **test_data
    )
    assert success
    
    # Verify the update
    leaderboard = leaderboard_manager.load_leaderboard()
    key = (1, 'positive')  # (config_id, correlation_type)
    assert key in leaderboard
    assert leaderboard[key]['correlation_value'] == 0.85
    assert leaderboard[key]['indicator_name'] == 'RSI'
    
    # Test updating with lower correlation (should not update)
    success = leaderboard_manager.check_and_update_single_lag(
        lag=5,
        correlation_value=0.75,  # Lower than 0.85
        indicator_name='RSI',
        params={'timeperiod': 14},
        config_id=1,
        **test_data
    )
    assert not success  # Should not update
    
    # Verify the value didn't change
    leaderboard = leaderboard_manager.load_leaderboard()
    assert leaderboard[key]['correlation_value'] == 0.85

def test_update_leaderboard_bulk(leaderboard_manager, test_data):
    """Test bulk leaderboard updates."""
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
    updates = leaderboard_manager.update_leaderboard(
        current_run_correlations=correlations,
        indicator_configs=indicator_configs,
        max_lag=10,
        **test_data
    )
    
    # Verify updates
    assert len(updates) > 0
    assert 1 in updates and 2 in updates
    
    # Check that each config has updates
    for config_id in [1, 2]:
        config_updates = updates[config_id]
        assert 'updates' in config_updates
        assert 'best_correlation' in config_updates
        assert len(config_updates['updates']) > 0
        assert config_updates['best_correlation'] > 0
    
    # Verify database entries
    leaderboard = leaderboard_manager.load_leaderboard()
    assert len(leaderboard) > 0
    
    # Check specific entries
    for config in indicator_configs:
        config_id = config['config_id']
        # Should have both positive and negative correlations
        has_pos = any((lag, 'positive') in leaderboard for lag in range(1, 11))
        has_neg = any((lag, 'negative') in leaderboard for lag in range(1, 11))
        assert has_pos or has_neg, f"Config {config_id} should have at least one correlation entry"

def test_find_best_predictor(leaderboard_manager, test_data):
    """Test finding best predictor for a lag."""
    # First add some test data
    test_configs = [
        {
            'config_id': 1,
            'indicator_name': 'RSI',
            'params': {'timeperiod': 14}
        },
        {
            'config_id': 2,
            'indicator_name': 'BB',
            'params': {'timeperiod': 20}
        }
    ]
    
    # Add correlations
    for config in test_configs:
        leaderboard_manager.check_and_update_single_lag(
            lag=5,
            correlation_value=0.85 if config['config_id'] == 1 else 0.75,
            **config,
            **test_data
        )
    
    # Test finding best predictor
    predictor = leaderboard_manager.find_best_predictor_for_lag(5)
    assert predictor is not None
    assert predictor['indicator_name'] == 'RSI'  # Should be RSI with 0.85 correlation
    assert predictor['correlation_value'] == 0.85
    
    # Test with non-existent lag
    predictor = leaderboard_manager.find_best_predictor_for_lag(999)
    assert predictor is None

def test_export_leaderboard(leaderboard_manager, test_data, temp_db_dir):
    """Test exporting leaderboard to text file."""
    # First add some test data
    leaderboard_manager.check_and_update_single_lag(
        lag=5,
        correlation_value=0.85,
        indicator_name='RSI',
        params={'timeperiod': 14},
        config_id=1,
        **test_data
    )
    
    # Export leaderboard
    success = leaderboard_manager.export_leaderboard_to_text()
    assert success
    
    # Verify export file exists
    export_path = Path('reports/leaderboard.txt')
    assert export_path.exists()
    assert export_path.stat().st_size > 0
    
    # Clean up
    export_path.unlink()

def test_generate_reports(leaderboard_manager, test_data, temp_db_dir):
    """Test generating various reports."""
    # First add some test data
    test_configs = [
        {
            'config_id': 1,
            'indicator_name': 'RSI',
            'params': {'timeperiod': 14}
        },
        {
            'config_id': 2,
            'indicator_name': 'BB',
            'params': {'timeperiod': 20}
        }
    ]
    
    # Add correlations for both configs
    for config in test_configs:
        leaderboard_manager.check_and_update_single_lag(
            lag=5,
            correlation_value=0.85 if config['config_id'] == 1 else 0.75,
            **config,
            **test_data
        )
    
    # Test leading indicator report
    success = leaderboard_manager.generate_leading_indicator_report()
    assert success
    
    # Test consistency report
    correlations_by_config = {
        1: [0.85, 0.80, 0.75],  # RSI correlations
        2: [0.75, 0.70, 0.65]   # BB correlations
    }
    
    success = leaderboard_manager.generate_consistency_report(
        correlations_by_config_id=correlations_by_config,
        indicator_configs_processed=test_configs,
        max_lag=5,
        output_dir=temp_db_dir,
        file_prefix='test_consistency',
        abs_corr_threshold=0.15
    )
    assert success
    
    # Verify report files exist
    assert (temp_db_dir / 'test_consistency_report.txt').exists()
    assert (temp_db_dir / 'test_consistency_heatmap.png').exists()

def test_error_handling(leaderboard_manager, test_data):
    """Test error handling in various scenarios."""
    # Test with invalid database connection
    with patch('leaderboard_manager._create_leaderboard_connection', return_value=None):
        with pytest.raises(Exception):
            leaderboard_manager.load_leaderboard()
    
    # Test with invalid correlation value
    with pytest.raises(ValueError):
        leaderboard_manager.check_and_update_single_lag(
            lag=5,
            correlation_value=1.5,  # Invalid correlation > 1
            indicator_name='RSI',
            params={'timeperiod': 14},
            config_id=1,
            **test_data
        )
    
    # Test with missing required data
    with pytest.raises(KeyError):
        leaderboard_manager.check_and_update_single_lag(
            lag=5,
            correlation_value=0.85,
            indicator_name='RSI',
            params={'timeperiod': 14},
            config_id=1,
            symbol='BTCUSDT'  # Missing other required fields
        ) 