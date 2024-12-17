# indicators.py
import logging
import pandas as pd
import numpy as np
import talib as ta
import pandas_ta as pta
import json
from typing import List, Dict, Any
from indicator_config_parser import parse_indicators_json, get_indicator_parameters
from sqlite_data_manager import create_connection, fetch_indicator_configs
from config import DB_PATH

logger = logging.getLogger(__name__)

def evaluate_conditions(params: Dict[str, Any], conditions: List[Dict[str, Dict[str, Any]]]) -> bool:
    operator_map = {
        'gt': '>',
        'gte': '>=',
        'lt': '<',
        'lte': '<=',
        'eq': '==',
        'neq': '!='
    }
    for condition in conditions:
        for param, ops in condition.items():
            for op, value in ops.items():
                mapped_op = operator_map.get(op)
                if not mapped_op:
                    logger.error(f"Unsupported operator '{op}' in conditions.")
                    return False
                if isinstance(value, str):
                    compare_value = params.get(value)
                    if compare_value is None:
                        logger.error(f"Condition value '{value}' for parameter '{param}' not found in params.")
                        return False
                else:
                    compare_value = value
                try:
                    if mapped_op == '>':
                        if not params[param] > compare_value:
                            return False
                    elif mapped_op == '>=':
                        if not params[param] >= compare_value:
                            return False
                    elif mapped_op == '<':
                        if not params[param] < compare_value:
                            return False
                    elif mapped_op == '<=':
                        if not params[param] <= compare_value:
                            return False
                    elif mapped_op == '==':
                        if not params[param] == compare_value:
                            return False
                    elif mapped_op == '!=':
                        if not params[param] != compare_value:
                            return False
                except TypeError as te:
                    logger.error(f"Type error in condition evaluation for parameter '{param}': {te}")
                    return False
    return True

def generate_parameter_combinations(parameters: Dict[str, Any], conditions: List[Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    from itertools import product

    param_ranges = {}
    for param, details in parameters.items():
        default = details.get('default')
        if isinstance(default, int):
            if param in ['fast', 'slow', 'fastperiod', 'slowperiod', 'signalperiod', 'timeperiod', 'timeperiod1', 'timeperiod2', 'timeperiod3', 'pivot_lookback']:
                start = max(2, default - 5)
            else:
                start = max(1, default - 5)
            param_ranges[param] = [start + i for i in range(11)]
        elif isinstance(default, float):
            if param in ['fast', 'slow', 'fastperiod', 'slowperiod', 'signalperiod', 'timeperiod', 'timeperiod1', 'timeperiod2', 'timeperiod3', 'pivot_lookback']:
                start = max(2.0, default - 5.0)
                param_ranges[param] = [round(start + i, 1) for i in range(11)]
            elif 0 < default < 1:
                start = max(0.1, default * 0.9)
                param_ranges[param] = [round(start + 0.02 * i, 4) for i in range(11)]
            else:
                param_ranges[param] = [default]
        else:
            param_ranges[param] = [default]

    all_combinations = list(product(*param_ranges.values()))
    keys = list(param_ranges.keys())
    combinations = [dict(zip(keys, values)) for values in all_combinations]

    valid_combinations = []
    for combo in combinations:
        if evaluate_conditions(combo, conditions):
            valid_combinations.append(combo)
    return valid_combinations

def compute_indicator(data: pd.DataFrame, indicator_name: str, params: Dict[str, Any], input_columns: List[str], config_id: int) -> pd.DataFrame:
    try:
        indicator_type = params.get('type')
        parameters = params.get('parameters', {})
        if indicator_type == 'ta-lib':
            ta_func = getattr(ta, indicator_name.upper(), None)
            if not ta_func:
                logger.error(f"TA-Lib function for indicator '{indicator_name}' not found.")
                return data
            func_args = [data[col].values for col in input_columns]
            func_params = parameters
            result = ta_func(*func_args, **func_params)
            if isinstance(result, tuple):
                for idx, res in enumerate(result):
                    column_name = f"{indicator_name}_config_{config_id}_{idx}"
                    if res is not None:
                        data[column_name] = res
            else:
                column_name = f"{indicator_name}_config_{config_id}"
                if result is not None:
                    data[column_name] = result
        elif indicator_type == 'pandas-ta':
            pta_func = getattr(pta, indicator_name.lower(), None)
            if not pta_func:
                logger.error(f"Pandas TA function for indicator '{indicator_name}' not found.")
                return data
            func_params = {k: v for k, v in parameters.items() if k not in ['type', 'conditions']}
            input_data = {col: data[col] for col in input_columns}
            result = pta_func(**func_params, **input_data)
            if isinstance(result, pd.DataFrame):
                for col in result.columns:
                    column_name = f"{col}_config_{config_id}"
                    data[column_name] = result[col]
            else:
                column_name = f"{indicator_name}_config_{config_id}"
                data[column_name] = result
        elif indicator_type == 'custom':
            logger.error(f"Custom indicator '{indicator_name}' computation is not implemented.")
        else:
            logger.error(f"Unknown indicator type '{indicator_type}' for indicator '{indicator_name}'.")
    except Exception as e:
        logger.error(f"Error computing indicator '{indicator_name}' with parameters {parameters}: {e}")
    return data

def compute_configured_indicators(data: pd.DataFrame, indicators_list: List[str], db_path: str = DB_PATH, indicator_params_path: str = 'indicator_params.json') -> pd.DataFrame:
    try:
        indicator_params = parse_indicators_json(indicator_params_path)
    except Exception as e:
        logger.error(f"Error loading indicator parameters: {e}")
        raise e

    conn = create_connection(db_path)
    if not conn:
        logger.error("Failed to connect to the database.")
        raise ConnectionError("Failed to connect to the database.")
    cursor = conn.cursor()
    new_columns = []
    for indicator_name in indicators_list:
        try:
            cursor.execute("""
                SELECT ic.id, ic.config FROM indicator_configs ic
                JOIN indicators i ON ic.indicator_id = i.id
                WHERE i.name = ?;
            """, (indicator_name,))
            rows = cursor.fetchall()
            if not rows:
                logger.error(f"No configurations found for indicator '{indicator_name}'.")
                continue
            for config_id, config_json in rows:
                config = json.loads(config_json)
                indicator_details = get_indicator_parameters(indicator_name, indicator_params_path)
                if not indicator_details:
                    logger.error(f"Indicator '{indicator_name}' not found in parameters JSON.")
                    continue
                conditions = indicator_details.get('conditions', [])
                parameters = {'type': indicator_details.get('type'), 'parameters': config}
                try:
                    computed_data = compute_indicator(data, indicator_name, parameters, indicator_details.get('required_inputs', []), config_id)
                    cols = [col for col in computed_data.columns if col.startswith(f"{indicator_name}_config_{config_id}")]
                    if cols:
                        new_columns.append(computed_data[cols])
                except Exception as e:
                    logger.error(f"Error computing indicator '{indicator_name}' config {config_id}: {e}")
        except Exception as e:
            logger.error(f"Error processing indicator '{indicator_name}': {e}")
    conn.close()
    if new_columns:
        new_df = pd.concat(new_columns, axis=1)
        data = pd.concat([data, new_df], axis=1)
    data = data.copy()
    data.dropna(inplace=True)
    return data
