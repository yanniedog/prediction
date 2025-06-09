import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Generator, cast
from unittest.mock import patch, MagicMock, call
import main
import utils
import config
import data_manager
import sqlite_manager
import parameter_generator
import parameter_optimizer
import indicator_factory
import visualization_generator
import predictor
import backtester
import itertools
import gc
import sqlite3

def _initialize_database(db_path: Path, symbol: str, timeframe: str) -> bool:
    """Helper function to initialize database with proper schema."""
    return sqlite_manager.initialize_database(db_path, symbol, timeframe)

@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass

@pytest.fixture(scope="function")
def sample_data() -> pd.DataFrame:
    """Create sample data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    np.random.seed(42)
    data = pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(200, 300, 100),
        'low': np.random.uniform(50, 100, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    })
    # Ensure price relationships are valid
    data['high'] = data[['open', 'close']].max(axis=1) + np.random.uniform(0, 50, 100)
    data['low'] = data[['open', 'close']].min(axis=1) - np.random.uniform(0, 50, 100)
    return data

@pytest.fixture(scope="function")
def sample_indicator_definitions() -> Dict[str, Dict[str, Any]]:
    """Create sample indicator definitions for testing."""
    return {
        'BB': {
            'conditions': [],
            'name': 'BBANDS',
            'params': {
                'nbdevdn': {
                    'default': 2.0,
                    'description': 'Lower band deviation',
                    'max': 5.0,
                    'min': 0.1
                },
                'nbdevup': {
                    'default': 2.0,
                    'description': 'Upper band deviation',
                    'max': 5.0,
                    'min': 0.1
                },
                'timeperiod': {
                    'default': 20,
                    'description': 'Time period',
                    'max': 100,
                    'min': 2
                }
            },
            'required_inputs': ['close'],
            'type': 'talib'
        },
        'RSI': {
            'conditions': [],
            'name': 'RSI',
            'params': {
                'timeperiod': {
                    'default': 14,
                    'description': 'RSI period',
                    'max': 100,
                    'min': 2
                }
            },
            'required_inputs': ['close'],
            'type': 'talib'
        }
    }

@pytest.fixture(scope="function")
def mock_db_connection() -> MagicMock:
    """Create a mock database connection."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn

def test_setup_and_select_mode(temp_dir: Path) -> None:
    """Test setup and mode selection."""
    # Test with valid input
    with patch('builtins.input', return_value='a'):
        result = main._setup_and_select_mode("20240101_120000")
        assert result == 'a'
    
    # Test with invalid input followed by valid input
    with patch('builtins.input', side_effect=['invalid', 'c']):
        result = main._setup_and_select_mode("20240101_120000")
        assert result == 'c'
    
    # Test with quit input
    with patch('builtins.input', return_value='q'):
        with pytest.raises(SystemExit):
            main._setup_and_select_mode("20240101_120000")

def test_select_data_source_and_lag(temp_dir: Path, sample_data: pd.DataFrame) -> None:
    """Test data source and lag selection."""
    # Create test database
    db_path = temp_dir / "BTCUSD_1h.db"
    _initialize_database(db_path, "BTCUSD", "1h")
    
    # Insert test data
    conn = sqlite_manager.create_connection(str(db_path))
    try:
        cursor = conn.cursor()
        # Get symbol and timeframe IDs
        cursor.execute("SELECT id FROM symbols WHERE symbol = ?", ("BTCUSD",))
        symbol_id = cursor.fetchone()[0]
        cursor.execute("SELECT id FROM timeframes WHERE timeframe = ?", ("1h",))
        timeframe_id = cursor.fetchone()[0]
        
        # Insert test data
        for _, row in sample_data.iterrows():
            cursor.execute("""
                INSERT INTO historical_data 
                (symbol_id, timeframe_id, open_time, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol_id, timeframe_id,
                int(row['date'].timestamp() * 1000),
                row['open'], row['high'], row['low'], row['close'], row['volume']
            ))
        conn.commit()
    finally:
        conn.close()
    
    # Test with valid database
    with patch('main.data_manager.manage_data_source', return_value=(db_path, "BTCUSD", "1h")):
        with patch('builtins.input', return_value='5'):  # max_lag=5
            result = main._select_data_source_and_lag(max_lag=5)
            assert len(result) == 8  # Should return 8 values
            assert result[0] == db_path
            assert result[1] == "BTCUSD"
            assert result[2] == "1h"
            assert result[4] == 5  # max_lag

def test_prepare_configurations(
    temp_dir: Path,
    sample_data: pd.DataFrame,
    sample_indicator_definitions: Dict[str, Dict[str, Any]]
) -> None:
    """Test configuration preparation."""
    # Create test database
    db_path = temp_dir / "BTCUSD_1h.db"
    _initialize_database(db_path, "BTCUSD", "1h")
    
    # Mock indicator factory
    with patch('main.indicator_factory.IndicatorFactory') as mock_factory:
        mock_instance = MagicMock()
        mock_instance.indicator_params = sample_indicator_definitions
        mock_instance.compute_indicators.return_value = pd.DataFrame({
            'RSI': [0.5, 0.6, 0.7] * 33 + [0.5],  # 100 values
            'BB_upper': [1.0] * 100,
            'BB_middle': [0.5] * 100,
            'BB_lower': [0.0] * 100
        })
        mock_factory.return_value = mock_instance
        
        # Mock database operations
        with patch('main.sqlite_manager.create_connection') as mock_conn:
            mock_conn.return_value = MagicMock()
            
            # Mock cursor operations
            def mock_fetchone():
                # Get the current query parameters
                import inspect
                frame = inspect.currentframe()
                while frame:
                    if 'self' in frame.f_locals and hasattr(frame.f_locals['self'], 'last_query'):
                        query = frame.f_locals['self'].last_query
                        if 'symbols' in query:
                            return (1,)  # symbol_id
                        elif 'timeframes' in query:
                            return (1,)  # timeframe_id
                        elif 'indicator_configs' in query:
                            return (1,)  # config_id
                    frame = frame.f_back
                return (1,)  # default
            
            mock_cursor = MagicMock()
            mock_cursor.fetchone.side_effect = mock_fetchone
            mock_cursor.fetchall.return_value = [(1,)]  # config_id
            mock_conn.return_value.cursor.return_value = mock_cursor
            
            # Test configuration preparation
            with patch('builtins.input', side_effect=['y', '5']):  # confirm analysis, max_lag=5
                # Mock the display progress function
                mock_display = MagicMock()
                
                configs, is_bayesian, total_steps = main._prepare_configurations(
                    mock_display,  # _display_progress_func
                    1,  # current_step
                    db_path,  # db_path
                    "BTCUSD",  # symbol
                    "1h",  # timeframe
                    5,  # max_lag
                    sample_data,  # data
                    sample_indicator_definitions,  # indicator_definitions
                    1,  # symbol_id
                    1,  # timeframe_id
                    "20240101-20240131",  # data_daterange_str
                    "20240101_120000"  # timestamp_str
                )
                assert isinstance(configs, list)
                assert isinstance(is_bayesian, bool)
                assert isinstance(total_steps, int)

def test_calculate_indicators_and_correlations(
    temp_dir: Path,
    sample_data: pd.DataFrame,
    sample_indicator_definitions: Dict[str, Dict[str, Any]]
) -> None:
    """Test indicator calculation and correlation computation."""
    # Create test database
    db_path = temp_dir / "BTCUSD_1h.db"
    _initialize_database(db_path, "BTCUSD", "1h")
    
    # Mock indicator factory
    with patch('main.indicator_factory.IndicatorFactory') as mock_factory:
        mock_instance = MagicMock()
        mock_instance.indicator_params = sample_indicator_definitions
        mock_instance.compute_indicators.return_value = pd.DataFrame({
            'RSI': [0.5, 0.6, 0.7] * 33 + [0.5],  # 100 values
            'BB_upper': [1.0] * 100,
            'BB_middle': [0.5] * 100,
            'BB_lower': [0.0] * 100
        })
        mock_factory.return_value = mock_instance
        
        # Mock database operations
        with patch('main.sqlite_manager.create_connection') as mock_conn:
            mock_conn.return_value = MagicMock()
            
            # Mock cursor operations
            def mock_fetchone():
                return (1,)  # config_id
            
            mock_cursor = MagicMock()
            mock_cursor.fetchone.side_effect = mock_fetchone
            mock_cursor.fetchall.return_value = [(1,)]  # config_id
            mock_conn.return_value.cursor.return_value = mock_cursor
            
            # Test calculation
            configs = [
                {
                    'indicator_name': 'RSI',
                    'params': {'timeperiod': 14},
                    'config_id': 1
                }
            ]
            
            # Mock the display progress function
            mock_display = MagicMock()
            
            final_configs, correlations, max_lag, total_steps = main._calculate_indicators_and_correlations(
                mock_display,  # _display_progress_func
                1,  # current_step
                db_path,  # db_path
                1,  # symbol_id
                1,  # timeframe_id
                5,  # max_lag
                sample_data,  # data
                configs,  # indicator_configs_to_process
                time.time(),  # analysis_start_time_global
                10  # total_analysis_steps_global
            )
            assert isinstance(final_configs, list)
            assert isinstance(correlations, dict)
            assert isinstance(max_lag, int)
            assert isinstance(total_steps, int)

def test_generate_final_reports_and_predict(
    temp_dir: Path,
    sample_data: pd.DataFrame,
    sample_indicator_definitions: Dict[str, Dict[str, Any]]
) -> None:
    """Test final report generation and prediction."""
    # Create test database
    db_path = temp_dir / "BTCUSD_1h.db"
    _initialize_database(db_path, "BTCUSD", "1h")
    
    # Create sample configurations and correlations
    configs = [
        {
            'indicator_name': 'RSI',
            'params': {'period': 14},
            'config_id': 1
        }
    ]
    correlations: Dict[int, List[Optional[float]]] = {
        1: [0.5, 0.4, 0.3, 0.2, 0.1]
    }
    
    # Mock display progress function
    mock_display = MagicMock()
    
    # Mock visualization functions
    with patch('main.visualization_generator.plot_correlation_lines') as mock_plot_lines:
        with patch('main.visualization_generator.generate_combined_correlation_chart') as mock_combined:
            with patch('main.visualization_generator.generate_enhanced_heatmap') as mock_heatmap:
                with patch('main.visualization_generator.generate_correlation_envelope_chart') as mock_envelope:
                    with patch('main.predictor.predict_price') as mock_predict:
                        # Patch input to avoid OSError from stdin
                        with patch('builtins.input', return_value='n'):
                            # Test successful report generation
                            main._generate_final_reports_and_predict(
                                mock_display,
                                current_step=1,
                                db_path=db_path,
                                symbol="BTCUSD",
                                timeframe="1h",
                                max_lag=5,
                                symbol_id=1,
                                timeframe_id=1,
                                data_daterange_str="20240101-20240131",
                                timestamp_str="20240101_120000",
                                is_bayesian_path=False,
                                final_configs_for_corr=configs,
                                correlations_by_config_id=correlations,
                                analysis_start_time_global=time.time(),
                                total_analysis_steps_global=10
                            )
                        mock_plot_lines.assert_called_once()
                        mock_combined.assert_called_once()
                        # Heatmap is called twice: once in run_interim_reports and once in main visualization
                        assert mock_heatmap.call_count == 2
                        mock_envelope.assert_called_once()
                        mock_predict.assert_called_once()

def test_run_analysis(temp_dir: Path, sample_data: pd.DataFrame) -> None:
    """Test main analysis orchestration."""
    # Create test database
    db_path = temp_dir / "BTCUSD_1h.db"
    _initialize_database(db_path, "BTCUSD", "1h")
    
    # Mock all necessary functions
    with patch('main._setup_and_select_mode', return_value='a'):
        with patch('main._select_data_source_and_lag', return_value=(
            db_path, "BTCUSD", "1h", sample_data, 10, 1, 1, "20240101-20240131"
        )):
            with patch('main.indicator_factory.IndicatorFactory') as mock_factory:
                # Create a mock instance
                mock_instance = MagicMock()
                # Set up the indicator_params
                mock_instance.indicator_params = {
                    "RSI": {
                        "name": "RSI",
                        "type": "talib",
                        "params": {
                            "timeperiod": "close"
                        }
                    }
                }
                # Set up _compute_single_indicator to return a test DataFrame
                mock_instance._compute_single_indicator.return_value = pd.DataFrame({
                    'RSI': [0.5, 0.6, 0.7]
                })
                # Make the factory return our mock instance
                mock_factory.return_value = mock_instance
                # Patch _prepare_configurations to set _confirmed_analysis_start_time
                def patched_prepare_configurations(*args, **kwargs):
                    import main
                    import time
                    main._confirmed_analysis_start_time = time.time()
                    return ([{'indicator_name': 'RSI', 'params': {'period': 14}, 'config_id': 1}], False, 2)
                with patch('main._prepare_configurations', side_effect=patched_prepare_configurations):
                    with patch('main._calculate_indicators_and_correlations', return_value=(
                        [{'indicator_name': 'RSI', 'params': {'period': 14}, 'config_id': 1}],
                        {1: [0.5, 0.4, 0.3, 0.2, 0.1]},
                        5,
                        3
                    )):
                        with patch('main._generate_final_reports_and_predict'):
                            # Test successful analysis run
                            main.run_analysis()
                            # Verify all functions were called
                            mock_factory.assert_called_once()

@pytest.mark.no_stdin
@pytest.mark.timeout(10)
def test_error_handling(temp_dir: Path) -> None:
    """Test error handling in main functions."""
    # Test user quit during data source selection
    with patch('main.data_manager.manage_data_source', return_value=None):  # Simulate user quit
        with pytest.raises(SystemExit):
            main._select_data_source_and_lag()

    # Test invalid data source (None returned from manage_data_source)
    with patch('main.data_manager.manage_data_source', return_value=None):
        with pytest.raises(SystemExit):
            main._select_data_source_and_lag()

    # Test data validation failure
    with patch('main.data_manager.manage_data_source', return_value=(Path("test.db"), "BTCUSD", "1h")):
        with patch('main.data_manager.load_data', return_value=None):  # Simulate data load failure
            with pytest.raises(ValueError):
                main._select_data_source_and_lag()

    # Test database connection failure
    with patch('main.data_manager.manage_data_source', return_value=(Path("test.db"), "BTCUSD", "1h")):
        with patch('main.sqlite_manager.create_connection', return_value=None):  # Simulate connection failure
            with pytest.raises(ValueError):
                main._select_data_source_and_lag()

    # Test database with missing tables
    db_path = temp_dir / "test.db"
    # Create a database file but don't initialize it (no tables)
    db_path.touch()
    
    with patch('main.data_manager.manage_data_source', return_value=(db_path, "BTCUSD", "1h")):
        with pytest.raises(ValueError):
            main._select_data_source_and_lag()

def test_progress_display(temp_dir: Path) -> None:
    """Test progress display functionality."""
    # Create a mock display function
    mock_display = MagicMock()
    
    # Test progress updates
    mock_display("Test Stage", 1, 10)
    mock_display.assert_called_once_with("Test Stage", 1, 10)
    
    # Test ETA updates
    with patch('time.time', side_effect=[0, 1, 2]):
        mock_display("Test Stage", 2, 10)
        assert mock_display.call_count == 2
    
    # Test completion display
    mock_display("Complete", 10, 10)
    assert mock_display.call_count == 3

def test_mode_specific_functions(temp_dir: Path) -> None:
    """Test mode-specific functions (custom and backtest)."""
    # Test custom mode
    with patch('main.data_manager.select_existing_database', return_value=Path("test.db")):
        with patch('main.sqlite_manager.create_connection') as mock_conn:
            mock_conn.return_value = MagicMock()
            with patch('builtins.input', side_effect=['q']):
                main._run_custom_mode("20240101_120000")
    
    # Test backtest mode
    with patch('main.data_manager.select_existing_database', return_value=Path("BTCUSDT_1h.db")):
        with patch('main.data_manager.validate_data', return_value=True):
            with patch('main.backtester.run_backtest') as mock_backtest:
                with patch('builtins.input', side_effect=['7', '50']):  # max_lag=7, points=50
                    main._run_backtest_check_mode("20240101_120000")
                    mock_backtest.assert_called_once() 