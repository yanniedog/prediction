# indicator_config_parser.py
import json

def parse_indicators_json(indicator_params_path='indicator_params.json'):
    with open(indicator_params_path,'r')as f:return json.load(f)

def get_configurable_indicators(indicator_params_path='indicator_params.json'):
    i=parse_indicators_json(indicator_params_path)
    return sorted(i.get("indicators",{}).keys())

def get_indicator_parameters(indicator_name,indicator_params_path='indicator_params.json'):
    i=parse_indicators_json(indicator_params_path)
    return i.get("indicators",{}).get(indicator_name)
