import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from leaderboard_manager import LeaderboardManager
import json

@pytest.fixture
def leaderboard_manager():
    """Provide a LeaderboardManager instance for testing."""
    return LeaderboardManager()

@pytest.fixture
def test_data():
    """Provide test data for leaderboard operations."""
    return {
        'lag': 5,
        'correlation_value': 0.85,
        'indicator_name': 'Test Indicator',
        'params': {'window': 20, 'threshold': 0.5},
        'config_id': 1,
        'symbol': 'BTCUSDT',
        'timeframe': '1h',
        'data_daterange': '2023-01-01 to 2023-12-31',
        'source_db_name': 'test.db'
    }

def test_leaderboard_manager_initialization(leaderboard_manager):
    """Test LeaderboardManager initialization."""
    assert leaderboard_manager is not None
    assert hasattr(leaderboard_manager, 'create_connection')
    assert hasattr(leaderboard_manager, 'load_leaderboard')

def test_check_and_update_single_lag(leaderboard_manager, test_data):
    """Test checking and updating a single lag."""
    # Test positive correlation update
    result = leaderboard_manager.check_and_update_single_lag(
        lag=test_data['lag'],
        correlation_value=test_data['correlation_value'],
        indicator_name=test_data['indicator_name'],
        params=test_data['params'],
        config_id=test_data['config_id'],
        symbol=test_data['symbol'],
        timeframe=test_data['timeframe'],
        data_daterange=test_data['data_daterange'],
        source_db_name=test_data['source_db_name']
    )
    assert isinstance(result, bool)
    
    # Test negative correlation update
    result = leaderboard_manager.check_and_update_single_lag(
        lag=test_data['lag'],
        correlation_value=-test_data['correlation_value'],
        indicator_name=test_data['indicator_name'],
        params=test_data['params'],
        config_id=test_data['config_id'],
        symbol=test_data['symbol'],
        timeframe=test_data['timeframe'],
        data_daterange=test_data['data_daterange'],
        source_db_name=test_data['source_db_name']
    )
    assert isinstance(result, bool)

def test_update_leaderboard(leaderboard_manager, test_data):
    """Test updating the leaderboard with multiple correlations."""
    # Create test correlations
    correlations = {
        5: [0.85, 0.75, 0.65],  # lag 5
        10: [0.70, 0.60, 0.50]  # lag 10
    }
    
    indicator_configs = [
        {
            'config_id': 1,
            'indicator_name': 'Test Indicator 1',
            'params': {'window': 20}
        },
        {
            'config_id': 2,
            'indicator_name': 'Test Indicator 2',
            'params': {'window': 30}
        }
    ]
    
    leaderboard_manager.update_leaderboard(
        current_run_correlations=correlations,
        indicator_configs=indicator_configs,
        max_lag=10,
        symbol=test_data['symbol'],
        timeframe=test_data['timeframe'],
        data_daterange=test_data['data_daterange'],
        source_db_name=test_data['source_db_name']
    )
    
    # Verify leaderboard data
    leaderboard_data = leaderboard_manager.load_leaderboard()
    assert isinstance(leaderboard_data, dict)
    assert len(leaderboard_data) > 0

def test_export_leaderboard_to_text(leaderboard_manager, tmp_path):
    """Test exporting leaderboard to text file."""
    # First add some data
    test_data = {
        'lag': 5,
        'correlation_value': 0.85,
        'indicator_name': 'Test Indicator',
        'params': {'window': 20},
        'config_id': 1,
        'symbol': 'BTCUSDT',
        'timeframe': '1h',
        'data_daterange': '2023-01-01 to 2023-12-31',
        'source_db_name': 'test.db'
    }
    
    leaderboard_manager.check_and_update_single_lag(**test_data)
    
    # Export to text
    result = leaderboard_manager.export_leaderboard_to_text()
    assert result is True

def test_find_best_predictor_for_lag(leaderboard_manager, test_data):
    """Test finding the best predictor for a lag."""
    # First add some data
    leaderboard_manager.check_and_update_single_lag(**test_data)
    
    # Find best predictor
    best_predictor = leaderboard_manager.find_best_predictor_for_lag(test_data['lag'])
    assert best_predictor is None or isinstance(best_predictor, dict)

def test_generate_leading_indicator_report(leaderboard_manager, test_data):
    """Test generating leading indicator report."""
    # First add some data
    leaderboard_manager.check_and_update_single_lag(**test_data)
    
    # Generate report
    result = leaderboard_manager.generate_leading_indicator_report()
    assert isinstance(result, bool)

def test_generate_consistency_report(leaderboard_manager, test_data, tmp_path):
    """Test generating consistency report."""
    # Create test data
    correlations = {
        1: [0.85, 0.75, 0.65],
        2: [0.70, 0.60, 0.50]
    }
    
    indicator_configs = [
        {
            'config_id': 1,
            'indicator_name': 'Test Indicator 1',
            'params': {'window': 20}
        },
        {
            'config_id': 2,
            'indicator_name': 'Test Indicator 2',
            'params': {'window': 30}
        }
    ]
    
    # Generate report
    result = leaderboard_manager.generate_consistency_report(
        correlations_by_config_id=correlations,
        indicator_configs_processed=indicator_configs,
        max_lag=2,
        output_dir=tmp_path,
        file_prefix='test_consistency',
        abs_corr_threshold=0.15
    )
    assert isinstance(result, bool) 