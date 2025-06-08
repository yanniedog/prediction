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

@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(scope="function")
def sample_data() -> pd.DataFrame:
    """Create sample price data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D', tz='UTC')
    data = pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(100, 200, len(dates)),
        'high': np.random.uniform(200, 300, len(dates)),
        'low': np.random.uniform(50, 100, len(dates)),
        'close': np.random.uniform(100, 200, len(dates)),
        'volume': np.random.uniform(1000, 5000, len(dates))
    })
    return data

@pytest.fixture(scope="function")
def sample_indicator_definitions() -> Dict[str, Dict[str, Any]]:
    """Create sample indicator definitions for testing."""
    return {
        "RSI": {
            "name": "RSI",
            "params": {
                "period": {
                    "default": 14,
                    "min": 2,
                    "max": 100
                }
            },
            "parameters": {
                "period": {
                    "default": 14,
                    "min": 2,
                    "max": 100
                }
            }
        },
        "MACD": {
            "name": "MACD",
            "params": {
                "fast": {
                    "default": 12.0,
                    "min": 1.0,
                    "max": 50.0
                },
                "slow": {
                    "default": 26.0,
                    "min": 5.0,
                    "max": 100.0
                }
            },
            "parameters": {
                "fast": {
                    "default": 12.0,
                    "min": 1.0,
                    "max": 50.0
                },
                "slow": {
                    "default": 26.0,
                    "min": 5.0,
                    "max": 100.0
                }
            }
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
    # Mock leaderboard initialization
    with patch('main.leaderboard_manager.initialize_leaderboard_db', return_value=True):
        # Mock cleanup functions
        with patch('main.utils.cleanup_previous_content') as mock_cleanup:
            # Test mode selection
            with patch('builtins.input', side_effect=['a']):
                mode = main._setup_and_select_mode("20240101_120000")
                assert mode == 'a'
                mock_cleanup.assert_called_once()
            
            # Test quit option
            with patch('builtins.input', side_effect=['q']):
                mode = main._setup_and_select_mode("20240101_120000")
                assert mode is None
            
            # Test invalid option
            with patch('builtins.input', side_effect=['invalid', 'a']):
                mode = main._setup_and_select_mode("20240101_120000")
                assert mode == 'a'

def test_select_data_source_and_lag(temp_dir: Path, sample_data: pd.DataFrame) -> None:
    """Test data source selection and lag input."""
    # Create a test database file
    db_path = temp_dir / "BTCUSD_1h.db"
    sample_data.to_sql('prices', sqlite_manager.create_connection(str(db_path)), if_exists='replace', index=False)
    
    # Mock data manager functions
    with patch('main.data_manager.manage_data_source', return_value=(db_path, "BTCUSD", "1h")):
        with patch('main.data_manager.load_data', return_value=sample_data):
            # Mock database operations
            with patch('main.sqlite_manager.create_connection') as mock_conn:
                mock_conn_instance = MagicMock()
                mock_cursor = MagicMock()
                # fetchone() returns (1,) for symbol, timeframe, then (1, '{}') for config id
                fetchone_side_effects = [(1,), (1,)] + [(1, '{}')] * 20
                mock_cursor.fetchone.side_effect = fetchone_side_effects
                mock_conn_instance.cursor.return_value = mock_cursor
                mock_conn.return_value = mock_conn_instance
                # Patch input to avoid stdin error
                with patch('builtins.input', return_value=''):
                    # Test successful selection
                    result = main._select_data_source_and_lag()
                    assert len(result) == 8
                    db_path, symbol, timeframe, data, max_lag, symbol_id, timeframe_id, data_daterange = result
                    assert symbol == "BTCUSD"
                    assert timeframe == "1h"
                    assert isinstance(data, pd.DataFrame)
                    assert max_lag > 0
                    assert symbol_id > 0
                    assert timeframe_id > 0
                    assert isinstance(data_daterange, str)

def test_prepare_configurations(
    temp_dir: Path,
    sample_data: pd.DataFrame,
    sample_indicator_definitions: Dict[str, Dict[str, Any]]
) -> None:
    """Test configuration preparation."""
    # Create test database
    db_path = temp_dir / "BTCUSD_1h.db"
    sample_data.to_sql('prices', sqlite_manager.create_connection(str(db_path)), if_exists='replace', index=False)
    
    # Mock display progress function
    mock_display = MagicMock()
    
    # Test classical path
    with patch('builtins.input', side_effect=['c', 'y']):
        with patch('main.sqlite_manager.create_connection') as mock_conn:
            mock_conn_instance = MagicMock()
            mock_cursor = MagicMock()
            
            # Create a sequence of mock responses that matches the actual database structure
            # First, mock the indicator lookups
            indicator_lookups = {
                'RSI': 1,
                'MACD': 2
            }
            
            # Then mock the config lookups with proper JSON strings
            config_lookups = {
                # RSI configs
                (1, utils.get_config_hash({'period': 14})): (1, json.dumps({'period': 14}, sort_keys=True)),
                (1, utils.get_config_hash({'period': 12})): (2, json.dumps({'period': 12}, sort_keys=True)),
                (1, utils.get_config_hash({'period': 16})): (3, json.dumps({'period': 16}, sort_keys=True)),
                # MACD configs
                (2, utils.get_config_hash({'fast': 12, 'slow': 26})): (4, json.dumps({'fast': 12, 'slow': 26}, sort_keys=True)),
                (2, utils.get_config_hash({'fast': 10, 'slow': 24})): (5, json.dumps({'fast': 10, 'slow': 24}, sort_keys=True)),
                (2, utils.get_config_hash({'fast': 14, 'slow': 28})): (6, json.dumps({'fast': 14, 'slow': 28}, sort_keys=True))
            }
            
            def mock_fetchone():
                # Get the current query parameters
                query = mock_cursor.execute.call_args[0][0]
                params = mock_cursor.execute.call_args[0][1]
                
                if "SELECT id FROM indicators" in query:
                    # Indicator lookup
                    return (indicator_lookups.get(params[0]),)
                elif "SELECT config_id, config_json FROM indicator_configs" in query:
                    # Config lookup
                    return config_lookups.get((params[0], params[1]))
                return None
            
            mock_cursor.fetchone.side_effect = mock_fetchone
            mock_conn_instance.cursor.return_value = mock_cursor
            mock_conn.return_value = mock_conn_instance
            
            configs, is_bayesian, step = main._prepare_configurations(
                mock_display,
                current_step=1,
                db_path=db_path,
                symbol="BTCUSD",
                timeframe="1h",
                max_lag=10,
                data=sample_data,
                indicator_definitions=sample_indicator_definitions,
                symbol_id=1,
                timeframe_id=1,
                data_daterange_str="20240101-20240131",
                timestamp_str="20240101_120000",
                analysis_start_time_global=None,
                total_analysis_steps_global=10
            )
            assert isinstance(configs, list)
            assert not is_bayesian
            assert step > 1
    
    # Test Bayesian path (if skopt available)
    if parameter_optimizer.SKOPT_AVAILABLE:
        # Mock all necessary inputs for Bayesian path including scope selection and confirmation
        with patch('builtins.input', side_effect=['b', 'a', 'y', 'y']):  # Added extra 'y' for any additional confirmations
            with patch('main.parameter_optimizer.optimize_parameters_bayesian_per_lag') as mock_optimize:
                # Mock the optimization to return some test results
                mock_optimize.return_value = (
                    {'best_params': {'period': 14}, 'best_score': 0.8},  # best_res_log
                    [{'indicator_name': 'RSI', 'params': {'period': 14}, 'config_id': 1}]  # eval_configs_ind
                )
                with patch('main.sqlite_manager.create_connection') as mock_conn:
                    mock_conn_instance = MagicMock()
                    mock_cursor = MagicMock()
                    
                    # Reuse the same mock responses for Bayesian path
                    mock_cursor.fetchone.side_effect = mock_fetchone
                    mock_conn_instance.cursor.return_value = mock_cursor
                    mock_conn.return_value = mock_conn_instance
                    
                    configs, is_bayesian, step = main._prepare_configurations(
                        mock_display,
                        current_step=1,
                        db_path=db_path,
                        symbol="BTCUSD",
                        timeframe="1h",
                        max_lag=10,
                        data=sample_data,
                        indicator_definitions=sample_indicator_definitions,
                        symbol_id=1,
                        timeframe_id=1,
                        data_daterange_str="20240101-20240131",
                        timestamp_str="20240101_120000",
                        analysis_start_time_global=None,
                        total_analysis_steps_global=10
                    )
                    assert isinstance(configs, list)
                    assert is_bayesian
                    assert step > 1

def test_calculate_indicators_and_correlations(
    temp_dir: Path,
    sample_data: pd.DataFrame,
    sample_indicator_definitions: Dict[str, Dict[str, Any]]
) -> None:
    """Test indicator calculation and correlation computation."""
    # Create test database
    db_path = temp_dir / "BTCUSD_1h.db"
    sample_data.to_sql('prices', sqlite_manager.create_connection(str(db_path)), if_exists='replace', index=False)
                   
    # Create sample configurations
    configs = [
        {
            'indicator_name': 'RSI',
            'params': {'period': 14},
            'config_id': 1
        },
        {
            'indicator_name': 'MACD',
            'params': {'fast': 12, 'slow': 26},
            'config_id': 2
        }
    ]

    # Mock display progress function
    mock_display = MagicMock()

    # Create mock data with sufficient rows for correlation
    mock_data = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=250, freq='H'),
        'close': np.random.randn(250).cumsum() + 100,  # Random walk
        'RSI_1': np.random.uniform(0, 100, 250),  # Random RSI values, column name matches config_id
        'MACD_2': np.random.randn(250)  # Random MACD values, column name matches config_id
    })

    # Mock database operations and indicator factory
    with patch('main.sqlite_manager.create_connection') as mock_conn, \
         patch('main.indicator_factory.compute_configured_indicators') as mock_compute, \
         patch('main.correlation_calculator.process_correlations') as mock_corr:
        
        # Set up mocks
        mock_compute.return_value = (mock_data, set())  # No failed configs
        mock_conn.return_value = MagicMock()
        mock_corr.return_value = True  # Simulate successful correlation processing

        # Mock correlation data
        mock_correlations = {
            1: [0.5] * 10,  # RSI correlations for lag 0-9
            2: [0.3] * 10   # MACD correlations for lag 0-9
        }

        # Mock sqlite_manager.fetch_correlations
        with patch('main.sqlite_manager.fetch_correlations') as mock_fetch:
            mock_fetch.return_value = mock_correlations

            # Test successful calculation
            final_configs, correlations, max_lag, step = main._calculate_indicators_and_correlations(
                mock_display,
                current_step=1,
                db_path=db_path,
                symbol_id=1,
                timeframe_id=1,
                max_lag=10,
                data=sample_data,
                indicator_configs_to_process=configs,
                analysis_start_time_global=time.time(),
                total_analysis_steps_global=10
            )

            # Verify results
            assert len(final_configs) == 2  # Both configs should be processed
            assert correlations == mock_correlations  # Should match mock correlations
            assert max_lag == 10  # Should maintain original max_lag
            assert step > 1  # Step should be updated

            # Verify mock calls
            mock_compute.assert_called_once()
            mock_corr.assert_called_once()
            mock_fetch.assert_called_once()

def test_generate_final_reports_and_predict(
    temp_dir: Path,
    sample_data: pd.DataFrame,
    sample_indicator_definitions: Dict[str, Dict[str, Any]]
) -> None:
    """Test final report generation and prediction."""
    # Create test database
    db_path = temp_dir / "BTCUSD_1h.db"
    sample_data.to_sql('prices', sqlite_manager.create_connection(str(db_path)), if_exists='replace', index=False)
    
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
                        mock_heatmap.assert_called_once()
                        mock_envelope.assert_called_once()
                        mock_predict.assert_called_once()

def test_run_analysis(temp_dir: Path, sample_data: pd.DataFrame) -> None:
    """Test main analysis orchestration."""
    # Create test database
    db_path = temp_dir / "BTCUSD_1h.db"
    sample_data.to_sql('prices', sqlite_manager.create_connection(str(db_path)), if_exists='replace', index=False)
    
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
    
    # Test invalid indicator definitions
    with patch('main._setup_and_select_mode', return_value='a'):
        with patch('main._select_data_source_and_lag', return_value=(
            Path("test.db"), "BTCUSD", "1h", pd.DataFrame(), 10, 1, 1, "20240101-20240131"
        )):
            with patch('main.indicator_factory.IndicatorFactory') as mock_factory:
                mock_factory.return_value.indicator_params = {}
                with pytest.raises(SystemExit):
                    main.run_analysis()
    
    # Test database connection failure during data source selection
    with patch('main.data_manager.manage_data_source', return_value=(Path("test.db"), "BTCUSD", "1h")):
        with patch('main.sqlite_manager.create_connection', return_value=None):
            with pytest.raises(ValueError):
                main._select_data_source_and_lag()
    
    # Force cleanup after each test case
    gc.collect()
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except (ImportError, NameError):
        pass

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