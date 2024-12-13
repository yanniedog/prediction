# indicator_config_parser.py

import json
from typing import Dict, List, Optional

def parse_indicators_json(indicator_params_path: str = 'indicator_params.json') -> Dict[str, Dict]:
    """
    Parse the indicator_params.json to extract configurable indicators and their parameters.

    Args:
        indicator_params_path (str): Path to indicator_params.json

    Returns:
        Dict[str, Dict]: A dictionary mapping indicator names to their parameter dictionaries.
    """
    with open(indicator_params_path, 'r') as f:
        indicator_params = json.load(f)
    return indicator_params

def get_configurable_indicators(indicator_params_path: str = 'indicator_params.json') -> List[str]:
    """
    Get a list of indicators that have configurable parameters.

    Args:
        indicator_params_path (str): Path to indicator_params.json

    Returns:
        List[str]: List of configurable indicator names.
    """
    indicator_params = parse_indicators_json(indicator_params_path)
    return list(indicator_params.keys())

def get_indicator_parameters(indicator_name: str, indicator_params_path: str = 'indicator_params.json') -> Optional[Dict]:
    """
    Get the parameters for a specific indicator.

    Args:
        indicator_name (str): Name of the indicator.
        indicator_params_path (str): Path to indicator_params.json

    Returns:
        Optional[Dict]: Dictionary of parameter names and their default values, or None if not found.
    """
    indicator_params = parse_indicators_json(indicator_params_path)
    return indicator_params.get(indicator_name)
