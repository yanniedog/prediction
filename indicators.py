# indicators.py
import logging
import pandas as pd
import numpy as np
import talib as ta
import pandas_ta as pta
import json
from typing import List
from indicator_config_parser import parse_indicators_json
from sqlite_data_manager import create_connection, fetch_indicator_configs
from config import DB_PATH

logger = logging.getLogger(__name__)

def z_score(x: np.ndarray) -> float:
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0
    return (x[-1] - mean) / std

def compute_custom_indicator(data: pd.DataFrame, indicator_name: str, params: dict) -> pd.DataFrame:
    try:
        if hasattr(ta, indicator_name.upper()):
            ta_func = getattr(ta, indicator_name.upper())
            result = ta_func(data['close'], **params)
            if isinstance(result, tuple):
                for idx, res in enumerate(result):
                    column_name = f"{indicator_name}_{idx}"
                    data[column_name] = res
            else:
                data[indicator_name] = result
        elif indicator_name.lower() in pta.indicators():
            data[indicator_name] = pta.ta(indicator_name.lower(), close=data['close'], **params)
        else:
            logger.warning(f"Indicator '{indicator_name}' not recognized in TA-Lib or pandas_ta. Skipping.")
    except Exception as e:
        logger.error(f"Error computing indicator '{indicator_name}': {e}")
    return data

def compute_custom_indicators(data: pd.DataFrame, indicators: List[str], db_path: str = DB_PATH, indicator_params_path: str = 'indicator_params.json') -> pd.DataFrame:
    with open(indicator_params_path, 'r') as f:
        indicator_params = json.load(f)
    conn = create_connection(db_path)
    if not conn:
        logger.error("Failed to connect to the database.")
        return data
    cursor = conn.cursor()
    for indicator_name in indicators:
        try:
            cursor.execute("""
                SELECT config FROM indicator_configs 
                JOIN indicators ON indicator_configs.indicator_id = indicators.id
                WHERE indicators.name = ?
            """, (indicator_name,))
            rows = cursor.fetchall()
            configs = [json.loads(row[0]) for row in rows]
            for config in configs:
                data = compute_custom_indicator(data, indicator_name, config)
        except Exception as e:
            logger.error(f"Error processing indicator '{indicator_name}': {e}")
    conn.close()
    data.dropna(inplace=True)
    return data

def compute_all_indicators(data: pd.DataFrame, db_path: str = DB_PATH, indicator_params_path: str = 'indicator_params.json') -> pd.DataFrame:
    with open(indicator_params_path, 'r') as f:
        indicator_params = json.load(f)
    indicators_list = list(indicator_params.keys())
    data = compute_custom_indicators(data, indicators_list, db_path, indicator_params_path)
    return data
