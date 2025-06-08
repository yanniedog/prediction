import pytest
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
    _generate_random_valid_config
)

@pytest.fixture(scope="function")
def sample_indicator_definition() -> Dict[str, Any]:
    """Create a sample indicator definition for testing."""
    return {
        "name": "RSI",
        "parameters": {
            "period": {
                "default": 14,
                "min": 2,
                "max": 100
            },
            "fast": {
                "default": 12.0,
                "min": 1.0,
                "max": 50.0
            },
            "slow": {
                "default": 26.0,
                "min": 5.0,
                "max": 100.0
            },
            "signal": {
                "default": 9,
                "min": 1,
                "max": 50
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
            "max": 100
        },
        "fast": {
            "default": 12.0,
            "min": 1.0,
            "max": 50.0
        },
        "slow": {
            "default": 26.0,
            "min": 5.0,
            "max": 100.0
        },
        "signal": {
            "default": 9,
            "min": 1,
            "max": 50
        }
    }

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
    assert len(configs) == 0
    
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
    
    # Should have 3 values per parameter (default ± 1 step)
    assert len(int_values) == 3
    assert len(float_values) == 3
    assert len(period_values) == 3
    
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

def test_generate_random_valid_config_impossible_conditions():
    """Test generating random valid configuration with impossible conditions."""
    impossible_conditions = [{
        'fast': {'gt': 'slow'},  # Impossible: fast > slow
        'slow': {'lt': 'fast'}   # Impossible: slow < fast
    }]
    config = _generate_random_valid_config(sample_parameter_definitions, impossible_conditions)
    assert config is None

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