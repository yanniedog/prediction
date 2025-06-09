import pytest
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from parameter_optimizer import (
    _get_config_hash,
    _objective_function,
    optimize_parameters_bayesian_per_lag,
    _define_search_space,
    _process_default_config_fast_eval,
    _format_final_evaluated_configs,
    _log_optimization_summary,
    SKOPT_AVAILABLE,
    optimize_parameters,
    optimize_parameters_bayesian,
    optimize_parameters_classical,
    objective_function,
    _prepare_optimization_data,
    _evaluate_configuration
)
from pathlib import Path
import tempfile
import shutil
import json

# Test fixtures
@pytest.fixture
def sample_indicator_definition() -> Dict[str, Any]:
    return {
        'name': 'RSI',
        'parameters': {
            'period': {'default': 14, 'min': 2, 'max': 50},
            'fast': {'default': 12, 'min': 2, 'max': 30},
            'slow': {'default': 26, 'min': 5, 'max': 50},
            'factor': {'default': 0.5, 'min': 0.1, 'max': 1.0},
            'scalar': {'default': 2.0, 'min': 0.1, 'max': 5.0}
        },
        'conditions': [
            {
                'fast': {'lt': 'slow'},
                'period': {'gte': 2}
            }
        ]
    }

@pytest.fixture
def sample_base_data() -> pd.DataFrame:
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(200, 300, 100),
        'low': np.random.uniform(50, 100, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    })
    return data

@pytest.fixture
def sample_progress_info() -> Dict[str, Any]:
    return {
        'start_time': time.time(),
        'current_step': 1,
        'total_steps': 10,
        'eta': None,
        'last_update': time.time()
    }

@pytest.fixture
def sample_shifted_closes() -> Dict[int, pd.Series]:
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    base_series = pd.Series(np.random.uniform(100, 200, 100), index=dates)
    return {
        1: base_series.shift(-1),
        2: base_series.shift(-2),
        3: base_series.shift(-3)
    }

@pytest.fixture(scope="function")
def sample_data() -> pd.DataFrame:
    """Create sample price data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
    np.random.seed(42)
    data = pd.DataFrame({
        "open": np.random.normal(100, 1, 100),
        "high": np.random.normal(101, 1, 100),
        "low": np.random.normal(99, 1, 100),
        "close": np.random.normal(100, 1, 100),
        "volume": np.random.normal(1000, 100, 100)
    }, index=dates)
    data["high"] = data[["open", "close"]].max(axis=1) + abs(np.random.normal(0, 0.1, 100))
    data["low"] = data[["open", "close"]].min(axis=1) - abs(np.random.normal(0, 0.1, 100))
    return data

@pytest.fixture(scope="function")
def sample_indicator_definition() -> Dict[str, Any]:
    """Create sample indicator definition for testing."""
    return {
        "RSI": {
            "type": "talib",
            "required_inputs": ["close"],
            "params": {
                "timeperiod": {
                    "default": 14,
                    "min": 2,
                    "max": 100
                }
            }
        }
    }

@pytest.fixture(scope="function")
def optimization_data(sample_data: pd.DataFrame, sample_indicator_definition: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Prepare optimization data for testing."""
    return _prepare_optimization_data(sample_data, sample_indicator_definition)

# Test _get_config_hash
def test_get_config_hash_consistent():
    """Test that config hash is consistent for same parameters."""
    params1 = {'a': 1, 'b': 2, 'c': 3.0}
    params2 = {'c': 3.0, 'b': 2, 'a': 1}  # Different order
    assert _get_config_hash(params1) == _get_config_hash(params2)

def test_get_config_hash_different():
    """Test that config hash is different for different parameters."""
    params1 = {'a': 1, 'b': 2}
    params2 = {'a': 1, 'b': 3}
    assert _get_config_hash(params1) != _get_config_hash(params2)

def test_get_config_hash_types():
    """Test config hash with different parameter types."""
    params = {
        'int': 1,
        'float': 1.0,
        'str': 'test',
        'bool': True,
        'none': None,
        'list': [1, 2, 3],
        'dict': {'a': 1}
    }
    hash_value = _get_config_hash(params)
    assert isinstance(hash_value, str)
    assert len(hash_value) == 64  # SHA-256 hash length

# Test _objective_function
def test_objective_function_valid_params(
    sample_indicator_definition,
    sample_base_data,
    sample_progress_info,
    sample_shifted_closes,
    tmp_path
):
    """Test objective function with valid parameters."""
    # Create test database
    db_path = str(tmp_path / "test.db")
    
    # Setup test parameters
    param_names = ['period', 'fast', 'slow']
    params_list = [14, 12, 26]  # Valid parameters
    
    # Mock master_evaluated_configs
    master_evaluated_configs = {}
    
    # Mock failure tracker
    failure_tracker = defaultdict(int)
    
    # Mock interim correlations accumulator
    interim_correlations_accumulator = defaultdict(list)
    
    result = _objective_function(
        params_list=params_list,
        param_names=param_names,
        target_lag=1,
        indicator_name='RSI',
        indicator_def=sample_indicator_definition,
        base_data_with_required=sample_base_data,
        db_path=db_path,
        symbol_id=1,
        timeframe_id=1,
        master_evaluated_configs=master_evaluated_configs,
        progress_info=sample_progress_info,
        shifted_closes_global_cache=sample_shifted_closes,
        failure_tracker=failure_tracker,
        interim_correlations_accumulator=interim_correlations_accumulator,
        max_lag_for_accumulator=3,
        symbol='BTCUSD',
        timeframe='1h',
        data_daterange='20240101-20240131',
        source_db_name='test.db'
    )
    
    assert isinstance(result, float)
    assert result <= 0  # Negative correlation (we want to maximize absolute correlation)

def test_objective_function_invalid_params(
    sample_indicator_definition,
    sample_base_data,
    sample_progress_info,
    sample_shifted_closes,
    tmp_path
):
    """Test objective function with invalid parameters."""
    # Create test database
    db_path = str(tmp_path / "test.db")
    
    # Setup test parameters that violate conditions
    param_names = ['period', 'fast', 'slow']
    params_list = [14, 30, 20]  # Invalid: fast > slow
    
    # Mock master_evaluated_configs
    master_evaluated_configs = {}
    
    # Mock failure tracker
    failure_tracker = defaultdict(int)
    
    # Mock interim correlations accumulator
    interim_correlations_accumulator = defaultdict(list)
    
    result = _objective_function(
        params_list=params_list,
        param_names=param_names,
        target_lag=1,
        indicator_name='RSI',
        indicator_def=sample_indicator_definition,
        base_data_with_required=sample_base_data,
        db_path=db_path,
        symbol_id=1,
        timeframe_id=1,
        master_evaluated_configs=master_evaluated_configs,
        progress_info=sample_progress_info,
        shifted_closes_global_cache=sample_shifted_closes,
        failure_tracker=failure_tracker,
        interim_correlations_accumulator=interim_correlations_accumulator,
        max_lag_for_accumulator=3,
        symbol='BTCUSD',
        timeframe='1h',
        data_daterange='20240101-20240131',
        source_db_name='test.db'
    )
    
    assert result == 1e6  # High cost for invalid parameters

# Test _define_search_space
def test_define_search_space(sample_indicator_definition):
    """Test search space definition."""
    param_defs = sample_indicator_definition['RSI']['params']
    space, param_names, param_bounds, has_categorical = _define_search_space(param_defs)
    
    assert isinstance(space, list)
    assert isinstance(param_names, list)
    assert isinstance(param_bounds, dict)
    assert isinstance(has_categorical, bool)
    
    # Verify space dimensions match parameter definitions
    assert len(space) == len(param_defs)
    assert len(param_names) == len(param_defs)
    assert len(param_bounds) == len(param_defs)
    
    # Verify parameter names
    for name in param_names:
        assert name in param_defs
    
    # Verify bounds
    for name, bounds in param_bounds.items():
        assert name in param_defs
        assert len(bounds) == 2
        assert bounds[0] <= bounds[1]

# Test _process_default_config_fast_eval
def test_process_default_config_fast_eval(
    sample_indicator_definition,
    sample_base_data,
    sample_shifted_closes,
    tmp_path
):
    """Test processing default configuration."""
    # Create test database
    db_path = str(tmp_path / "test.db")
    
    # Setup test parameters
    param_names = ['period', 'fast', 'slow']
    fixed_params = {'period': 14, 'fast': 12, 'slow': 26}
    
    # Mock master_evaluated_configs
    master_evaluated_configs = {}
    
    # Mock interim correlations accumulator
    interim_correlations_accumulator = defaultdict(list)
    
    config_id = _process_default_config_fast_eval(
        indicator_name='RSI',
        indicator_def=sample_indicator_definition,
        fixed_params=fixed_params,
        param_names=param_names,
        db_path=db_path,
        master_evaluated_configs=master_evaluated_configs,
        base_data=sample_base_data,
        max_lag=3,
        shifted_closes_cache=sample_shifted_closes,
        interim_correlations_accumulator=interim_correlations_accumulator,
        max_lag_for_accumulator=3,
        symbol='BTCUSD',
        timeframe='1h',
        data_daterange='20240101-20240131',
        source_db_name='test.db'
    )
    
    assert isinstance(config_id, (int, type(None)))

# Test _format_final_evaluated_configs
def test_format_final_evaluated_configs():
    """Test formatting final evaluated configurations."""
    indicator_name = 'RSI'
    master_evaluated_configs = {
        'hash1': {
            'params': {'period': 14, 'fast': 12, 'slow': 26},
            'config_id': 1,
            'correlations': {1: 0.5, 2: 0.3}
        },
        'hash2': {
            'params': {'period': 20, 'fast': 10, 'slow': 30},
            'config_id': 2,
            'correlations': {1: 0.4, 2: 0.2}
        }
    }
    
    formatted_configs = _format_final_evaluated_configs(indicator_name, master_evaluated_configs)
    
    assert isinstance(formatted_configs, list)
    assert len(formatted_configs) == len(master_evaluated_configs)
    
    for config in formatted_configs:
        assert isinstance(config, dict)
        assert 'indicator_name' in config
        assert 'params' in config
        assert 'config_id' in config
        assert 'correlations' in config
        assert config['indicator_name'] == indicator_name

# Test _log_optimization_summary
def test_log_optimization_summary(caplog):
    """Test logging optimization summary."""
    indicator_name = 'RSI'
    max_lag = 3
    best_config_per_lag = {
        1: {'config_id': 1, 'correlation': 0.5},
        2: {'config_id': 2, 'correlation': 0.3},
        3: {'config_id': 3, 'correlation': 0.2}
    }
    final_all_evaluated_configs = [
        {'config_id': 1, 'params': {'period': 14}},
        {'config_id': 2, 'params': {'period': 20}},
        {'config_id': 3, 'params': {'period': 30}}
    ]
    
    _log_optimization_summary(
        indicator_name,
        max_lag,
        best_config_per_lag,
        final_all_evaluated_configs
    )
    
    # Verify logging
    assert any(f"Optimization summary for {indicator_name}" in record.message for record in caplog.records)
    assert any("Best configurations per lag:" in record.message for record in caplog.records)
    assert any("Total configurations evaluated:" in record.message for record in caplog.records)

# Test optimize_parameters_bayesian_per_lag
@pytest.mark.skipif(not SKOPT_AVAILABLE, reason="scikit-optimize not installed")
def test_optimize_parameters_bayesian_per_lag(
    sample_indicator_definition,
    sample_base_data,
    sample_shifted_closes,
    tmp_path
):
    """Test Bayesian optimization per lag."""
    # Create test database
    db_path = str(tmp_path / "test.db")
    
    # Mock display progress function
    mock_display = lambda *args, **kwargs: None
    
    # Mock interim correlations accumulator
    interim_correlations_accumulator = defaultdict(list)
    
    best_configs, all_configs = optimize_parameters_bayesian_per_lag(
        indicator_name='RSI',
        indicator_def=sample_indicator_definition,
        base_data_with_required=sample_base_data,
        max_lag=3,
        n_calls_per_lag=2,  # Small number for testing
        n_initial_points_per_lag=1,
        db_path=db_path,
        symbol_id=1,
        timeframe_id=1,
        symbol='BTCUSD',
        timeframe='1h',
        data_daterange='20240101-20240131',
        source_db_name='test.db',
        interim_correlations_accumulator=interim_correlations_accumulator,
        analysis_start_time_global=time.time(),
        total_analysis_steps_global=10,
        current_step_base=1,
        total_steps_in_phase=5,
        indicator_index=0,
        total_indicators_in_phase=1,
        display_progress_func=mock_display
    )
    
    assert isinstance(best_configs, dict)
    assert isinstance(all_configs, list)
    
    # Verify best configs structure
    for lag, config in best_configs.items():
        assert isinstance(lag, int)
        assert isinstance(config, dict)
        assert 'config_id' in config
        assert 'correlation' in config
    
    # Verify all configs structure
    for config in all_configs:
        assert isinstance(config, dict)
        assert 'indicator_name' in config
        assert 'params' in config
        assert 'config_id' in config
        assert 'correlations' in config 

def test_prepare_optimization_data(sample_data: pd.DataFrame, sample_indicator_definition: Dict[str, Any]):
    """Test preparation of optimization data."""
    data, indicator_def = _prepare_optimization_data(sample_data, sample_indicator_definition)
    
    # Verify data preparation
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert all(col in data.columns for col in ["open", "high", "low", "close", "volume"])
    
    # Verify indicator definition
    assert isinstance(indicator_def, dict)
    assert "RSI" in indicator_def
    assert "params" in indicator_def["RSI"]
    assert "timeperiod" in indicator_def["RSI"]["params"]

def test_objective_function(optimization_data: Tuple[pd.DataFrame, Dict[str, Any]]):
    """Test objective function evaluation."""
    data, indicator_def = optimization_data
    
    # Test with valid configuration
    config = {"timeperiod": 14}
    score = objective_function(config, data, indicator_def)
    assert isinstance(score, float)
    assert not np.isnan(score)
    assert not np.isinf(score)
    
    # Test with invalid configuration
    invalid_config = {"timeperiod": 1}  # Below min
    with pytest.raises(ValueError):
        objective_function(invalid_config, data, indicator_def)
    
    # Test with missing required parameter
    invalid_config = {}  # Missing timeperiod
    with pytest.raises(ValueError):
        objective_function(invalid_config, data, indicator_def)

def test_evaluate_configuration(optimization_data: Tuple[pd.DataFrame, Dict[str, Any]]):
    """Test configuration evaluation."""
    data, indicator_def = optimization_data
    
    # Test valid configuration
    config = {"timeperiod": 14}
    score, _ = _evaluate_configuration(config, data, indicator_def)
    assert isinstance(score, float)
    assert not np.isnan(score)
    assert not np.isinf(score)
    
    # Test invalid configuration
    invalid_config = {"timeperiod": 1}  # Below min
    with pytest.raises(ValueError):
        _evaluate_configuration(invalid_config, data, indicator_def)

def test_optimize_parameters_bayesian(optimization_data: Tuple[pd.DataFrame, Dict[str, Any]]):
    """Test Bayesian optimization."""
    data, indicator_def = optimization_data
    
    # Test optimization
    best_config, best_score = optimize_parameters_bayesian(data, indicator_def, n_trials=5)
    
    # Verify results
    assert isinstance(best_config, dict)
    assert "timeperiod" in best_config
    assert best_config["timeperiod"] >= 2
    assert best_config["timeperiod"] <= 100
    
    assert isinstance(best_score, float)
    assert not np.isnan(best_score)
    assert not np.isinf(best_score)
    
    # Test with invalid data
    with pytest.raises(ValueError):
        optimize_parameters_bayesian(pd.DataFrame(), indicator_def)
    
    # Test with invalid indicator definition
    with pytest.raises(ValueError):
        optimize_parameters_bayesian(data, {})

def test_optimize_parameters_classical(optimization_data: Tuple[pd.DataFrame, Dict[str, Any]]):
    """Test classical optimization."""
    data, indicator_def = optimization_data
    
    # Test optimization
    best_config, best_score = optimize_parameters_classical(data, indicator_def)
    
    # Verify results
    assert isinstance(best_config, dict)
    assert "timeperiod" in best_config
    assert best_config["timeperiod"] >= 2
    assert best_config["timeperiod"] <= 100
    
    assert isinstance(best_score, float)
    assert not np.isnan(best_score)
    assert not np.isinf(best_score)
    
    # Test with invalid data
    with pytest.raises(ValueError):
        optimize_parameters_classical(pd.DataFrame(), indicator_def)
    
    # Test with invalid indicator definition
    with pytest.raises(ValueError):
        optimize_parameters_classical(data, {})

def test_optimize_parameters(optimization_data: Tuple[pd.DataFrame, Dict[str, Any]]):
    """Test main optimization function."""
    data, indicator_def = optimization_data
    
    # Test Bayesian optimization
    try:
        best_config, best_score = optimize_parameters(data, indicator_def, method="bayesian", n_trials=5)
        assert isinstance(best_config, dict), f"Expected best_config to be a dictionary, got {type(best_config)}"
        assert isinstance(best_score, float), f"Expected best_score to be a float, got {type(best_score)}"
        assert best_score != float('-inf'), "Optimization failed to find any valid configuration"
    except Exception as e:
        pytest.fail(f"Bayesian optimization failed: {str(e)}")
    
    # Test classical optimization
    try:
        best_config, best_score = optimize_parameters(data, indicator_def, method="classical")
        assert isinstance(best_config, dict), f"Expected best_config to be a dictionary, got {type(best_config)}"
        assert isinstance(best_score, float), f"Expected best_score to be a float, got {type(best_score)}"
        assert best_score != float('-inf'), "Optimization failed to find any valid configuration"
    except Exception as e:
        pytest.fail(f"Classical optimization failed: {str(e)}")
    
    # Test invalid method
    with pytest.raises(ValueError, match="Invalid optimization method") as exc_info:
        optimize_parameters(data, indicator_def, method="invalid")
    assert "Invalid optimization method" in str(exc_info.value), f"Expected error about invalid method, got: {exc_info.value}"

def test_optimization_constraints(optimization_data: Tuple[pd.DataFrame, Dict[str, Any]]):
    """Test optimization constraints."""
    data, indicator_def = optimization_data
    
    # Test that optimization respects parameter bounds
    best_config, _ = optimize_parameters(data, indicator_def, method="bayesian", n_trials=5)
    assert best_config["timeperiod"] >= indicator_def["RSI"]["params"]["timeperiod"]["min"]
    assert best_config["timeperiod"] <= indicator_def["RSI"]["params"]["timeperiod"]["max"]
    
    # Test that optimization improves score
    default_config = {"timeperiod": indicator_def["RSI"]["params"]["timeperiod"]["default"]}
    default_score = objective_function(default_config, data, indicator_def)
    best_config, best_score = optimize_parameters(data, indicator_def, method="bayesian", n_trials=5)
    assert best_score >= default_score  # Optimization should not make things worse

def test_optimization_consistency(optimization_data: Tuple[pd.DataFrame, Dict[str, Any]]):
    """Test optimization consistency."""
    data, indicator_def = optimization_data
    
    # Run multiple optimizations and verify consistency
    results = []
    for _ in range(3):
        best_config, best_score = optimize_parameters(data, indicator_def, method="bayesian", n_trials=5)
        results.append((best_config, best_score))
    
    # Verify that results are reasonably consistent
    scores = [score for _, score in results]
    assert np.std(scores) < np.mean(scores)  # Standard deviation should be less than mean
    
    # Verify that configurations are valid
    for config, _ in results:
        assert config["timeperiod"] >= indicator_def["RSI"]["params"]["timeperiod"]["min"]
        assert config["timeperiod"] <= indicator_def["RSI"]["params"]["timeperiod"]["max"]

def test_error_handling(optimization_data: Tuple[pd.DataFrame, Dict[str, Any]]):
    """Test error handling in optimization."""
    data, indicator_def = optimization_data
    
    # Test with empty data
    with pytest.raises(ValueError, match="Input data is empty") as exc_info:
        optimize_parameters(pd.DataFrame(), indicator_def)
    assert "Input data is empty" in str(exc_info.value), f"Expected error about empty data, got: {exc_info.value}"
    
    # Test with invalid indicator definition
    with pytest.raises(ValueError, match="Invalid indicator definition") as exc_info:
        optimize_parameters(data, {})
    assert "Invalid indicator definition" in str(exc_info.value), f"Expected error about invalid definition, got: {exc_info.value}"
    
    # Test with invalid parameter ranges
    invalid_def = indicator_def.copy()
    invalid_def["RSI"]["params"]["timeperiod"]["min"] = 100
    invalid_def["RSI"]["params"]["timeperiod"]["max"] = 50
    with pytest.raises(ValueError) as exc_info:
        optimize_parameters(data, invalid_def)
    assert "min >= max" in str(exc_info.value), f"Expected error about invalid parameter range, got: {exc_info.value}"
    assert "timeperiod" in str(exc_info.value), "Error message should mention the problematic parameter"
    
    # Test with missing required parameter
    invalid_def = indicator_def.copy()
    del invalid_def["RSI"]["params"]["timeperiod"]
    with pytest.raises(ValueError) as exc_info:
        optimize_parameters(data, invalid_def)
    assert "timeperiod" in str(exc_info.value), "Error message should mention the missing parameter"
    assert "required" in str(exc_info.value).lower(), "Error message should indicate parameter is required" 