import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Generator, cast
from pathlib import Path
import tempfile
import shutil
import json
import parameter_generator
import utils
from parameter_generator import (
    generate_param_grid,
    _evaluate_single_condition,
    evaluate_conditions,
    generate_configurations,
    generate_classical_configurations,
    generate_bayesian_configurations,
    validate_parameter_ranges,
    _generate_range_values,
    _generate_random_valid_config
)

@pytest.fixture(scope="function")
def sample_indicator_definitions() -> Dict[str, Dict[str, Any]]:
    """Create sample indicator definitions for testing."""
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
        },
        "BB": {
            "type": "talib",
            "required_inputs": ["close"],
            "params": {
                "timeperiod": {
                    "default": 20,
                    "min": 5,
                    "max": 200
                },
                "nbdevup": {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 5.0
                },
                "nbdevdn": {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 5.0
                }
            }
        },
        "MACD": {
            "type": "talib",
            "required_inputs": ["close"],
            "params": {
                "fastperiod": {
                    "default": 12,
                    "min": 2,
                    "max": 50
                },
                "slowperiod": {
                    "default": 26,
                    "min": 5,
                    "max": 100
                },
                "signalperiod": {
                    "default": 9,
                    "min": 2,
                    "max": 50
                }
            },
            "conditions": [
                {
                    "fastperiod": {
                        "lt": "slowperiod"
                    }
                }
            ]
        }
    }

@pytest.fixture(scope="function")
def sample_indicator_definition() -> Dict[str, Any]:
    """Create a sample indicator definition for testing."""
    return {
        "name": "RSI",
        "parameters": {
            "period": {
                "default": 14,
                "min": 2,
                "max": 50
            },
            "fast": {
                "default": 12.0,
                "min": 2.0,
                "max": 30.0
            },
            "slow": {
                "default": 26.0,
                "min": 5.0,
                "max": 50.0
            },
            "factor": {
                "default": 0.5,
                "min": 0.1,
                "max": 1.0
            },
            "scalar": {
                "default": 2.0,
                "min": 0.1,
                "max": 5.0
            },
            "bool_param": {
                "default": True
            },
            "str_param": {
                "default": "value"
            },
            "none_param": {
                "default": None
            }
        },
        "conditions": [
            {
                "fast": {
                    "lt": "slow"  # fast must be less than slow
                }
            },
            {
                "period": {
                    "gt": 0  # period must be positive
                }
            },
            {
                "factor": {
                    "gt": 0.1  # factor must be greater than 0.1
                }
            },
            {
                "scalar": {
                    "gt": 0.1  # scalar must be greater than 0.1
                }
            }
        ],
        "range_steps_default": 2  # Use 2 steps around default
    }

@pytest.fixture(scope="function")
def sample_parameter_definitions() -> Dict[str, Dict[str, Any]]:
    """Create sample parameter definitions for testing."""
    return {
        "period": {
            "default": 14,
            "min": 2,
            "max": 50
        },
        "fast": {
            "default": 12.0,
            "min": 2.0,
            "max": 30.0
        },
        "slow": {
            "default": 26.0,
            "min": 5.0,
            "max": 50.0
        },
        "factor": {
            "default": 0.5,
            "min": 0.1,
            "max": 1.0
        },
        "scalar": {
            "default": 2.0,
            "min": 0.1,
            "max": 5.0
        },
        "bool_param": {
            "default": True
        },
        "str_param": {
            "default": "value"
        },
        "none_param": {
            "default": None
        }
    }

@pytest.fixture(scope="function")
def sample_conditions() -> List[Dict[str, Dict[str, Any]]]:
    """Create sample conditions for testing."""
    return [
        {
            "fast": {
                "lt": "slow"  # fast must be less than slow
            }
        },
        {
            "period": {
                "gt": 0  # period must be positive
            }
        },
        {
            "factor": {
                "gt": 0.1  # factor must be greater than 0.1
            }
        },
        {
            "scalar": {
                "gt": 0.1  # scalar must be greater than 0.1
            }
        }
    ]

def test_evaluate_single_condition() -> None:
    """Test evaluation of single conditions."""
    # Test numeric comparisons
    assert parameter_generator._evaluate_single_condition(5, 'gt', 3) is True
    assert parameter_generator._evaluate_single_condition(5, 'gte', 5) is True
    assert parameter_generator._evaluate_single_condition(3, 'lt', 5) is True
    assert parameter_generator._evaluate_single_condition(5, 'lte', 5) is True
    assert parameter_generator._evaluate_single_condition(5, 'eq', 5) is True
    assert parameter_generator._evaluate_single_condition(5, 'neq', 3) is True
    
    # Test float comparisons
    assert parameter_generator._evaluate_single_condition(5.0, 'gt', 3.0) is True
    assert parameter_generator._evaluate_single_condition(5.0, 'eq', 5.0) is True
    
    # Test None handling
    assert parameter_generator._evaluate_single_condition(None, 'eq', None) is True
    assert parameter_generator._evaluate_single_condition(None, 'neq', 5) is True
    assert parameter_generator._evaluate_single_condition(5, 'neq', None) is True
    
    # Test invalid operator
    assert parameter_generator._evaluate_single_condition(5, 'invalid', 3) is False
    
    # Test type mismatches
    assert parameter_generator._evaluate_single_condition("5", 'gt', 3) is False
    assert parameter_generator._evaluate_single_condition(5, 'gt', "3") is False
    
    # Test edge cases
    assert parameter_generator._evaluate_single_condition(float('inf'), 'gt', 1e9) is True
    assert parameter_generator._evaluate_single_condition(float('nan'), 'eq', float('nan')) is False

def test_evaluate_conditions() -> None:
    """Test evaluation of multiple conditions."""
    # Test valid conditions
    params = {
        "fast": 12.0,
        "slow": 26.0,
        "period": 14
    }
    conditions = [
        {
            "fast": {
                "lt": "slow"  # fast < slow
            }
        },
        {
            "period": {
                "gt": 0  # period > 0
            }
        }
    ]
    assert parameter_generator.evaluate_conditions(params, conditions) is True
    
    # Test invalid conditions
    invalid_params = {
        "fast": 30.0,  # fast > slow
        "slow": 26.0,
        "period": 14
    }
    assert parameter_generator.evaluate_conditions(invalid_params, conditions) is False
    
    # Test missing parameters
    missing_params = {
        "fast": 12.0,
        "period": 14
        # slow is missing
    }
    assert parameter_generator.evaluate_conditions(missing_params, conditions) is False
    
    # Test empty conditions
    assert parameter_generator.evaluate_conditions(params, []) is True
    
    # Test invalid condition format
    invalid_conditions: List[Dict[str, Dict[str, Any]]] = [
        {
            "fast": {
                "invalid": "invalid"  # Invalid operator
            }
        }
    ]
    assert parameter_generator.evaluate_conditions(params, invalid_conditions) is True  # Invalid conditions are skipped
    
    # Test None values
    none_params = {
        "fast": None,
        "slow": 26.0,
        "period": 14
    }
    assert parameter_generator.evaluate_conditions(none_params, conditions) is False

def test_generate_configurations_basic(sample_indicator_definitions):
    """Test basic configuration generation."""
    # Test with single parameter indicator
    configs = generate_configurations(sample_indicator_definitions["RSI"])
    assert isinstance(configs, list)
    assert len(configs) > 0
    for config in configs:
        assert isinstance(config, dict)
        assert "timeperiod" in config
        assert config["timeperiod"] >= 2
        assert config["timeperiod"] <= 100
    
    # Test with multiple parameter indicator
    configs = generate_configurations(sample_indicator_definitions["BB"])
    assert isinstance(configs, list)
    assert len(configs) > 0
    for config in configs:
        assert isinstance(config, dict)
        assert all(param in config for param in ["timeperiod", "nbdevup", "nbdevdn"])
        assert config["timeperiod"] >= 5 and config["timeperiod"] <= 200
        assert config["nbdevup"] >= 0.5 and config["nbdevup"] <= 5.0
        assert config["nbdevdn"] >= 0.5 and config["nbdevdn"] <= 5.0

def test_generate_classical_configurations(sample_indicator_definitions):
    """Test classical configuration generation."""
    # Test with single parameter
    configs = generate_classical_configurations(sample_indicator_definitions["RSI"])
    assert isinstance(configs, list)
    assert len(configs) > 0
    
    # Verify parameter ranges
    timeperiods = {config["timeperiod"] for config in configs}
    assert min(timeperiods) >= 2
    assert max(timeperiods) <= 100
    assert 14 in timeperiods  # Default value should be included
    
    # Test with multiple parameters
    configs = generate_classical_configurations(sample_indicator_definitions["MACD"])
    assert isinstance(configs, list)
    assert len(configs) > 0
    
    # Verify all parameters are within ranges
    for config in configs:
        assert config["fastperiod"] >= 2 and config["fastperiod"] <= 50
        assert config["slowperiod"] >= 5 and config["slowperiod"] <= 100
        assert config["signalperiod"] >= 2 and config["signalperiod"] <= 50
        assert config["fastperiod"] < config["slowperiod"]  # Logical constraint

def test_generate_bayesian_configurations(sample_indicator_definitions):
    """Test Bayesian configuration generation."""
    # Test with single parameter
    configs = generate_bayesian_configurations(sample_indicator_definitions["RSI"])
    assert isinstance(configs, list)
    assert len(configs) > 0
    
    # Verify parameter ranges
    for config in configs:
        assert config["timeperiod"] >= 2
        assert config["timeperiod"] <= 100
    
    # Test with multiple parameters
    configs = generate_bayesian_configurations(sample_indicator_definitions["BB"])
    assert isinstance(configs, list)
    assert len(configs) > 0
    
    # Verify all parameters are within ranges
    for config in configs:
        assert config["timeperiod"] >= 5 and config["timeperiod"] <= 200
        assert config["nbdevup"] >= 0.5 and config["nbdevup"] <= 5.0
        assert config["nbdevdn"] >= 0.5 and config["nbdevdn"] <= 5.0

def test_validate_parameter_ranges(sample_indicator_definitions):
    """Test parameter range validation."""
    # Test valid ranges
    assert validate_parameter_ranges(sample_indicator_definitions["RSI"]["params"])
    assert validate_parameter_ranges(sample_indicator_definitions["BB"]["params"])
    
    # Test invalid ranges
    invalid_params = {
        "timeperiod": {
            "default": 14,
            "min": 100,  # min > max
            "max": 50
        }
    }
    assert not validate_parameter_ranges(invalid_params)
    
    # Test missing required fields
    invalid_params = {
        "timeperiod": {
            "default": 14,
            "min": 2
            # missing max
        }
    }
    assert not validate_parameter_ranges(invalid_params)
    
    # Test invalid default value
    invalid_params = {
        "timeperiod": {
            "default": 150,  # default > max
            "min": 2,
            "max": 100
        }
    }
    assert not validate_parameter_ranges(invalid_params)

def test_generate_range_values():
    """Test range value generation."""
    # Test integer range
    values = _generate_range_values(2, 10, 5, is_int=True)
    assert len(values) == 5
    assert all(isinstance(v, int) for v in values)
    assert min(values) >= 2
    assert max(values) <= 10
    
    # Test float range
    values = _generate_range_values(0.5, 5.0, 5, is_int=False)
    assert len(values) == 5
    assert all(isinstance(v, float) for v in values)
    assert min(values) >= 0.5
    assert max(values) <= 5.0
    
    # Test with default value
    values = _generate_range_values(2, 10, 5, is_int=True, default=5)
    assert 5 in values  # Default value should be included
    
    # Test with small range
    values = _generate_range_values(1, 2, 5, is_int=True)
    assert len(values) == 2  # Should only generate unique values
    assert set(values) == {1, 2}

def test_error_handling(sample_indicator_definitions):
    """Test error handling in parameter generation."""
    # Test with invalid indicator definition
    with pytest.raises(ValueError):
        generate_configurations({})
    
    # Test with missing parameter definitions
    invalid_def = {
        "type": "talib",
        "required_inputs": ["close"],
        "params": {}  # Empty params
    }
    configs = generate_configurations(invalid_def)
    assert len(configs) == 1  # Should return empty config when no params
    assert configs[0] == {}  # Should be empty dict
    
    # Test with invalid parameter type
    invalid_def = {
        "type": "talib",
        "required_inputs": ["close"],
        "params": {
            "timeperiod": {
                "default": "invalid",  # Should be number
                "min": 2,
                "max": 100
            }
        }
    }
    with pytest.raises(ValueError):
        generate_configurations(invalid_def)
    
    # Test with invalid range values
    invalid_def = {
        "type": "talib",
        "required_inputs": ["close"],
        "params": {
            "timeperiod": {
                "default": 14,
                "min": "invalid",  # Should be number
                "max": 100
            }
        }
    }
    with pytest.raises(ValueError):
        generate_configurations(invalid_def)

def test_parameter_constraints(sample_indicator_definitions):
    """Test parameter constraints and relationships."""
    # Test MACD constraints (fastperiod < slowperiod)
    configs = generate_configurations(sample_indicator_definitions["MACD"])
    for config in configs:
        assert config["fastperiod"] < config["slowperiod"]
    
    # Test BB constraints (nbdevup and nbdevdn should be positive)
    configs = generate_configurations(sample_indicator_definitions["BB"])
    for config in configs:
        assert config["nbdevup"] > 0
        assert config["nbdevdn"] > 0

def test_configuration_uniqueness(sample_indicator_definitions):
    """Test that generated configurations are unique."""
    # Test RSI configurations
    configs = generate_configurations(sample_indicator_definitions["RSI"])
    param_values = [config["timeperiod"] for config in configs]
    assert len(param_values) == len(set(param_values))  # All values should be unique
    
    # Test BB configurations
    configs = generate_configurations(sample_indicator_definitions["BB"])
    param_tuples = [(config["timeperiod"], config["nbdevup"], config["nbdevdn"]) for config in configs]
    assert len(param_tuples) == len(set(param_tuples))  # All combinations should be unique

def test_generate_configurations(sample_indicator_definition: Dict[str, Any]) -> None:
    """Test generation of parameter configurations."""
    # Test basic configuration generation
    configs = parameter_generator.generate_configurations(sample_indicator_definition)
    assert len(configs) > 0
    
    # Verify all generated configs satisfy conditions
    for config in configs:
        assert parameter_generator.evaluate_conditions(config, sample_indicator_definition['conditions'])
    
    # Verify default configuration is included
    default_config = {
        param: details['default']
        for param, details in sample_indicator_definition['parameters'].items()
        if 'default' in details
    }
    assert any(utils.compare_param_dicts(config, default_config) for config in configs)
    
    # Test with invalid definition
    invalid_definition = {
        "name": "Invalid",
        "parameters": {
            "param1": {
                "default": "invalid"  # Non-numeric default
            }
        }
    }
    configs = parameter_generator.generate_configurations(invalid_definition)
    assert len(configs) == 1  # Should only include default
    
    # Test with no parameters
    empty_definition = {
        "name": "Empty",
        "parameters": {}
    }
    configs = parameter_generator.generate_configurations(empty_definition)
    assert len(configs) == 1  # Should return empty config when no params
    assert configs[0] == {}  # Should be empty dict
    
    # Test with tight bounds
    tight_definition = {
        "name": "Tight",
        "parameters": {
            "param1": {
                "default": 5,
                "min": 5,
                "max": 5
            }
        }
    }
    configs = parameter_generator.generate_configurations(tight_definition)
    assert len(configs) == 1
    assert configs[0]["param1"] == 5

def test_generate_random_valid_config(sample_parameter_definitions: Dict[str, Dict[str, Any]]) -> None:
    """Test generation of random valid configurations."""
    conditions = [
        {
            "fast": {
                "lt": "slow"  # fast < slow
            }
        }
    ]
    
    # Test successful generation
    config = parameter_generator._generate_random_valid_config(
        sample_parameter_definitions,
        conditions
    )
    assert config is not None
    assert parameter_generator.evaluate_conditions(config, conditions)
    
    # Test with impossible conditions
    impossible_conditions = [
        {
            "fast": {
                "gt": "slow"  # fast > slow
            },
            "slow": {
                "gt": "fast"  # slow > fast
            }
        }
    ]
    config = parameter_generator._generate_random_valid_config(
        sample_parameter_definitions,
        impossible_conditions
    )
    assert config is None
    
    # Test with no numeric parameters
    non_numeric_defs = {
        "param1": {
            "default": "value"
        }
    }
    config = parameter_generator._generate_random_valid_config(
        non_numeric_defs,
        []
    )
    assert config == {"param1": "value"}
    
    # Test with invalid parameter definitions
    invalid_defs = {
        "param1": {
            "min": "invalid",  # Invalid min
            "max": "invalid"   # Invalid max
        }
    }
    config = parameter_generator._generate_random_valid_config(
        invalid_defs,
        []
    )
    assert config is None
    
    # Test with max tries
    config = parameter_generator._generate_random_valid_config(
        sample_parameter_definitions,
        conditions,
        max_tries=1  # Force failure
    )
    # Should still return default if valid
    if config is not None:
        assert parameter_generator.evaluate_conditions(config, conditions)

def test_parameter_ranges_and_bounds() -> None:
    """Test parameter range generation and boundary handling."""
    definition = {
        "name": "Test",
        "parameters": {
            "int_param": {
                "default": 10,
                "min": 5,
                "max": 15
            },
            "float_param": {
                "default": 1.5,
                "min": 1.0,
                "max": 2.0
            },
            "period_param": {  # Special handling for period parameters
                "default": 14,
                "min": 1,
                "max": 50
            }
        },
        "range_steps_default": 1
    }
    
    configs = parameter_generator.generate_configurations(definition)
    assert len(configs) > 0
    
    # Verify all values are within bounds
    for config in configs:
        assert 5 <= config["int_param"] <= 15
        assert 1.0 <= config["float_param"] <= 2.0
        assert 1 <= config["period_param"] <= 50
    
    # Verify step sizes
    int_values = sorted(set(config["int_param"] for config in configs))
    float_values = sorted(set(config["float_param"] for config in configs))
    period_values = sorted(set(config["period_param"] for config in configs))
    
    # Should have values based on range_steps_default (1 step = 3 values: min, default, max)
    # But with classical_path_range_steps=3, we get more values
    assert len(int_values) >= 3  # At least 3 values
    assert len(float_values) >= 3  # At least 3 values
    assert len(period_values) >= 3  # At least 3 values
    
    # Verify default values are included
    assert 10 in int_values
    assert 1.5 in float_values
    assert 14 in period_values

def test_parameter_type_handling() -> None:
    """Test handling of different parameter types."""
    definition = {
        "name": "Test",
        "parameters": {
            "int_param": {
                "default": 10
            },
            "float_param": {
                "default": 1.5
            },
            "bool_param": {
                "default": True
            },
            "str_param": {
                "default": "value"
            },
            "none_param": {
                "default": None
            }
        }
    }
    
    configs = parameter_generator.generate_configurations(definition)
    assert len(configs) == 1  # Only default values for non-numeric
    
    config = configs[0]
    assert isinstance(config["int_param"], int)
    assert isinstance(config["float_param"], float)
    assert isinstance(config["bool_param"], bool)
    assert isinstance(config["str_param"], str)
    assert config["none_param"] is None

def test_condition_references() -> None:
    """Test handling of parameter references in conditions."""
    definition = {
        "name": "Test",
        "parameters": {
            "param1": {
                "default": 10
            },
            "param2": {
                "default": 20
            },
            "param3": {
                "default": 30
            }
        },
        "conditions": [
            {
                "param1": {
                    "lt": "param2"  # param1 < param2
                }
            },
            {
                "param2": {
                    "lt": "param3"  # param2 < param3
                }
            },
            {
                "param1": {
                    "lt": "param3"  # param1 < param3
                }
            }
        ]
    }
    
    configs = parameter_generator.generate_configurations(definition)
    assert len(configs) > 0
    
    # Verify all configs satisfy the reference conditions
    for config in configs:
        assert config["param1"] < config["param2"]
        assert config["param2"] < config["param3"]
        assert config["param1"] < config["param3"]
    
    # Test with missing referenced parameter
    invalid_definition = definition.copy()
    invalid_definition["parameters"].pop("param2")
    configs = parameter_generator.generate_configurations(invalid_definition)
    assert len(configs) == 0  # Should fail due to missing reference

# Test generate_param_grid
def test_generate_param_grid_empty():
    """Test generating parameter grid with empty input."""
    params = {}
    grid = list(generate_param_grid(params))
    assert len(grid) == 1
    assert grid[0] == {}

def test_generate_param_grid_single_param():
    """Test generating parameter grid with single parameter."""
    params = {'a': [1, 2, 3]}
    grid = list(generate_param_grid(params))
    assert len(grid) == 3
    assert grid == [{'a': 1}, {'a': 2}, {'a': 3}]

def test_generate_param_grid_multiple_params():
    """Test generating parameter grid with multiple parameters."""
    params = {'a': [1, 2], 'b': ['x', 'y']}
    grid = list(generate_param_grid(params))
    assert len(grid) == 4
    assert {'a': 1, 'b': 'x'} in grid
    assert {'a': 1, 'b': 'y'} in grid
    assert {'a': 2, 'b': 'x'} in grid
    assert {'a': 2, 'b': 'y'} in grid

# Test _evaluate_single_condition
def test_evaluate_single_condition_numeric():
    """Test evaluating single condition with numeric values."""
    assert _evaluate_single_condition(5, 'gt', 3) is True
    assert _evaluate_single_condition(5, 'gte', 5) is True
    assert _evaluate_single_condition(5, 'lt', 7) is True
    assert _evaluate_single_condition(5, 'lte', 5) is True
    assert _evaluate_single_condition(5, 'eq', 5) is True
    assert _evaluate_single_condition(5, 'neq', 3) is True

def test_evaluate_single_condition_invalid_operator():
    """Test evaluating single condition with invalid operator."""
    assert _evaluate_single_condition(5, 'invalid', 3) is False

def test_evaluate_single_condition_type_mismatch():
    """Test evaluating single condition with type mismatch."""
    assert _evaluate_single_condition('5', 'gt', 3) is False
    assert _evaluate_single_condition(5, 'gt', '3') is False

def test_evaluate_single_condition_none_values():
    """Test evaluating single condition with None values."""
    assert _evaluate_single_condition(None, 'eq', None) is True
    assert _evaluate_single_condition(None, 'neq', 3) is True
    assert _evaluate_single_condition(5, 'neq', None) is True

# Test evaluate_conditions
def test_evaluate_conditions_valid(sample_parameter_definitions, sample_conditions):
    """Test evaluating conditions with valid parameter combination."""
    params = {
        'fast': 12,
        'slow': 26,
        'period': 14,
        'factor': 0.5,
        'scalar': 2.0
    }
    assert evaluate_conditions(params, sample_conditions) is True

def test_evaluate_conditions_invalid(sample_parameter_definitions, sample_conditions):
    """Test evaluating conditions with invalid parameter combination."""
    params = {
        'fast': 30,  # Invalid: fast > slow
        'slow': 26,
        'period': 14,
        'factor': 0.5,
        'scalar': 2.0
    }
    assert evaluate_conditions(params, sample_conditions) is False

def test_evaluate_conditions_empty():
    """Test evaluating conditions with empty conditions list."""
    params = {'a': 1, 'b': 2}
    assert evaluate_conditions(params, []) is True

def test_evaluate_conditions_invalid_format():
    """Test evaluating conditions with invalid condition format."""
    params = {'a': 1, 'b': 2}
    invalid_conditions = [{'a': 'invalid'}]  # Not a dict of operations
    assert evaluate_conditions(params, invalid_conditions) is True  # Should skip invalid conditions

# Test generate_configurations
def test_generate_configurations_basic(sample_indicator_definition):
    """Test generating configurations with basic parameters."""
    configs = generate_configurations(sample_indicator_definition)
    assert isinstance(configs, list)
    assert len(configs) > 0
    
    # Verify all generated configs are valid
    for config in configs:
        assert isinstance(config, dict)
        assert 'period' in config
        assert 'fast' in config
        assert 'slow' in config
        assert 'factor' in config
        assert 'scalar' in config
        assert 'bool_param' in config
        assert 'str_param' in config
        assert 'none_param' in config

def test_generate_configurations_respects_bounds(sample_indicator_definition):
    """Test that generated configurations respect parameter bounds."""
    configs = generate_configurations(sample_indicator_definition)
    
    for config in configs:
        assert 2 <= config['period'] <= 50
        assert 2 <= config['fast'] <= 30
        assert 5 <= config['slow'] <= 50
        assert 0.1 <= config['factor'] <= 1.0
        assert 0.1 <= config['scalar'] <= 5.0
        assert config['bool_param'] is True
        assert config['str_param'] == 'value'
        assert config['none_param'] is None

def test_generate_configurations_respects_conditions(sample_indicator_definition):
    """Test that generated configurations respect conditions."""
    configs = generate_configurations(sample_indicator_definition)
    
    for config in configs:
        assert config['fast'] < config['slow']
        assert config['period'] >= 2
        assert 0.1 < config['factor'] < 1.0
        assert config['scalar'] > 0.1

def test_generate_configurations_no_parameters():
    """Test generating configurations with no parameters."""
    indicator_def = {
        'name': 'EmptyIndicator',
        'parameters': {},
        'conditions': [],
        'range_steps_default': 2
    }
    configs = generate_configurations(indicator_def)
    assert len(configs) == 1
    assert configs[0] == {}

# Test _generate_random_valid_config
def test_generate_random_valid_config(sample_parameter_definitions, sample_conditions):
    """Test generating random valid configuration."""
    config = _generate_random_valid_config(sample_parameter_definitions, sample_conditions)
    assert config is not None
    assert isinstance(config, dict)
    
    # Verify bounds
    assert 2 <= config['period'] <= 50
    assert 2 <= config['fast'] <= 30
    assert 5 <= config['slow'] <= 50
    assert 0.1 <= config['factor'] <= 1.0
    assert 0.1 <= config['scalar'] <= 5.0
    
    # Verify conditions
    assert config['fast'] < config['slow']
    assert config['period'] >= 2
    assert 0.1 < config['factor'] < 1.0
    assert config['scalar'] > 0.1

def test_generate_random_valid_config_impossible_conditions(sample_parameter_definitions):
    """Test generating random valid configuration with impossible conditions."""
    impossible_conditions = [{
        'fast': {'gt': 'slow'},  # Impossible: fast > slow
        'slow': {'lt': 'fast'}   # Impossible: slow < fast
    }]
    config = _generate_random_valid_config(sample_parameter_definitions, impossible_conditions)
    assert config is None  # Should return None for impossible conditions

def test_generate_random_valid_config_no_conditions():
    """Test generating random valid configuration with no conditions."""
    config = _generate_random_valid_config(sample_parameter_definitions, [])
    assert config is not None
    assert isinstance(config, dict)
    
    # Verify bounds only
    assert 2 <= config['period'] <= 50
    assert 2 <= config['fast'] <= 30
    assert 5 <= config['slow'] <= 50
    assert 0.1 <= config['factor'] <= 1.0
    assert 0.1 <= config['scalar'] <= 5.0 