# indicator_config_parser.py
import json
from typing import Dict, List, Optional

def parse_indicators_json(indicator_params_path: str = 'indicator_params.json') -> Dict[str, Dict]:
    with open(indicator_params_path, 'r') as f:
        indicator_params = json.load(f)
    return indicator_params

def get_configurable_indicators(indicator_params_path: str = 'indicator_params.json') -> List[str]:
    indicator_params = parse_indicators_json(indicator_params_path)
    return list(indicator_params.keys())

def get_indicator_parameters(indicator_name: str, indicator_params_path: str = 'indicator_params.json') -> Optional[Dict]:
    indicator_params = parse_indicators_json(indicator_params_path)
    return indicator_params.get(indicator_name)