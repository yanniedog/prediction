# indicators.py
import logging
import pandas as pd
import numpy as np
import talib as ta
import pandas_ta as pta
import json
from typing import List, Dict, Any
from indicator_config_parser import parse_indicators_json
from sqlite_data_manager import create_connection, fetch_indicator_configs
from config import DB_PATH

logger = logging.getLogger(__name__)

def evaluate_conditions(params: Dict[str, Any], conditions: List[Dict[str, Dict[str, Any]]]) -> bool:
    operator_map = {
        'greater_than': '>',
        'greater_than_or_equal': '>=',
        'less_than': '<',
        'less_than_or_equal': '<=',
        'equal': '==',
        'not_equal': '!='
    }
    for condition in conditions:
        for param, ops in condition.items():
            for op, value in ops.items():
                mapped_op = operator_map.get(op)
                if not mapped_op:
                    logger.error(f"Unsupported operator '{op}' in conditions.")
                    return False
                if isinstance(value, str):
                    if value not in params:
                        logger.error(f"Condition value '{value}' for parameter '{param}' not found in params.")
                        return False
                    compare_value = params[value]
                else:
                    compare_value = value
                if mapped_op == '<':
                    if not params[param] < compare_value:
                        return False
                elif mapped_op == '<=':
                    if not params[param] <= compare_value:
                        return False
                elif mapped_op == '>':
                    if not params[param] > compare_value:
                        return False
                elif mapped_op == '>=':
                    if not params[param] >= compare_value:
                        return False
                elif mapped_op == '==':
                    if not params[param] == compare_value:
                        return False
                elif mapped_op == '!=':
                    if not params[param] != compare_value:
                        return False
    return True

def generate_parameter_combinations(parameters: Dict[str, Any], conditions: List[Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    from itertools import product

    param_ranges = {}
    for param, details in parameters.items():
        p_type = details.get('type')
        default = details.get('default')
        step = details.get('step', 1)
        if 'range' in details:
            min_val, max_val = details['range']
            if p_type == 'int':
                param_ranges[param] = list(range(max(min_val, default - 5), min(max_val, default + 5) + 1, step))
            elif p_type == 'float':
                num_steps = int(((min(max_val, default + 5) - max(min_val, default - 5)) / step)) + 1
                param_ranges[param] = [round(max(min_val, default - 5) + step * i, 4) for i in range(num_steps)]
            elif p_type == 'str':
                param_ranges[param] = details.get('options', [default])
            elif p_type == 'list':
                param_ranges[param] = details.get('options', [default])
            else:
                param_ranges[param] = [default]
        elif 'options' in details:
            param_ranges[param] = details['options']
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
            result = pta_func(**func_params, **{col: data[col] for col in input_columns})
            if isinstance(result, pd.DataFrame):
                for col in result.columns:
                    column_name = f"{col}_config_{config_id}"
                    data[column_name] = result[col]
            else:
                column_name = f"{indicator_name}_config_{config_id}"
                data[column_name] = result
        elif indicator_type == 'custom':
            logger.warning(f"Custom indicator '{indicator_name}' computation is not implemented.")
        else:
            logger.error(f"Unknown indicator type '{indicator_type}' for indicator '{indicator_name}'.")
    except Exception as e:
        logger.error(f"Error computing indicator '{indicator_name}': {e}")
    return data

def compute_configured_indicators(data: pd.DataFrame, indicators_list: List[str], db_path: str = DB_PATH, indicator_params_path: str = 'indicator_params.json') -> pd.DataFrame:
    try:
        indicator_configs = parse_indicators_json(indicator_params_path)
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
                WHERE i.name = ?
            """, (indicator_name,))
            rows = cursor.fetchall()
            if not rows:
                logger.error(f"No configurations found for indicator '{indicator_name}'.")
                continue
            for config_id, config_json in rows:
                config = json.loads(config_json)
                indicator_details = indicator_configs.get(indicator_name)
                if not indicator_details:
                    logger.error(f"Indicator '{indicator_name}' not found in parameters JSON.")
                    continue
                conditions = indicator_details.get('conditions', [])
                parameters = {'type': indicator_details.get('type'), 'parameters': config}
                try:
                    computed_data = compute_indicator(data, indicator_name, parameters, indicator_details.get('input_columns', []), config_id)
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
