import json
from typing import Dict, Any
from indicator_params import indicator_definitions

class IndicatorFactory:
    """Class-based implementation for loading and managing indicator parameters."""
    def __init__(self):
        self.indicator_params = self.load_params()
        self.indicators = {}  # Initialize empty dict for indicators

    def load_params(self) -> Dict[str, Dict[str, Any]]:
        """Load indicator parameters from indicator_params.json or indicator_definitions."""
        try:
            with open('indicator_params.json', 'r') as f:
                params = json.load(f)
            return params
        except Exception:
            # Fallback to Python definitions if JSON fails
            return indicator_definitions

class IndicatorFactoryClass:
    def __init__(self, params_file: str):
        try:
            with open(params_file, 'r') as f:
                self.params = json.load(f)
        except FileNotFoundError:
            raise
        except json.JSONDecodeError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to load params: {e}")

    def get_indicator_params(self, name: str) -> dict:
        if name not in self.params:
            raise KeyError(name)
        return self.params[name]

    def get_all_indicators(self):
        return list(self.params.keys())

    def validate_params(self, name: str, params: dict) -> bool:
        if name not in self.params:
            raise KeyError(name)
        definition = self.params[name]
        if 'parameters' not in definition:
            raise ValueError("No parameters defined for indicator")
        for param, spec in definition['parameters'].items():
            if param not in params:
                raise ValueError(f"Missing parameter: {param}")
            value = params[param]
            if spec['type'] == 'int':
                if not isinstance(value, int):
                    raise ValueError(f"Parameter {param} must be int")
                if not (spec['min'] <= value <= spec['max']):
                    raise ValueError(f"Parameter {param} out of range")
            elif spec['type'] == 'float':
                if not isinstance(value, (float, int)):
                    raise ValueError(f"Parameter {param} must be float")
                if not (spec['min'] <= float(value) <= spec['max']):
                    raise ValueError(f"Parameter {param} out of range")
            else:
                raise ValueError(f"Unknown parameter type: {spec['type']}")
            if value is None:
                raise ValueError(f"Parameter {param} is None")
        return True 