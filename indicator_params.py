"""Module for loading and managing indicator definitions."""
import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Path to indicator parameters JSON file
INDICATOR_PARAMS_PATH = Path("indicator_params.json")
PARAMS_FILE = str(INDICATOR_PARAMS_PATH)  # Allow override in tests

def load_indicator_definitions(path: Path = INDICATOR_PARAMS_PATH) -> Dict[str, Any]:
    """Load indicator definitions from JSON file."""
    try:
        if not path.exists():
            logger.error(f"{path} not found")
            return {}
        with open(path, 'r') as f:
            definitions = json.load(f)
        if not isinstance(definitions, dict):
            logger.error("Invalid indicator definitions format")
            return {}
        return definitions
    except Exception as e:
        logger.error(f"Failed to load indicator definitions: {e}")
        return {}

# Load definitions at module level
definitions_cache = None
def _get_definitions():
    global definitions_cache
    if definitions_cache is not None:
        return definitions_cache
    try:
        with open(PARAMS_FILE, 'r') as f:
            definitions_cache = json.load(f)
    except Exception:
        definitions_cache = {}
    return definitions_cache

def get_indicator_params(name: str) -> dict:
    defs = _get_definitions()
    if name not in defs:
        raise KeyError(name)
    return defs[name]

def get_all_indicators() -> list:
    defs = _get_definitions()
    return list(defs.keys())

def validate_indicator_params(name: str, params: dict) -> bool:
    defs = _get_definitions()
    if name not in defs:
        raise KeyError(name)
    definition = defs[name]
    if 'params' not in definition:
        raise ValueError("No parameters defined for indicator")
    for param, spec in definition['params'].items():
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

# For backward compatibility
indicator_definitions = load_indicator_definitions() 