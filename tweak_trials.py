# tweak_trials.py
import logging
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
from sqlite_data_manager import create_connection, fetch_indicator_configs, initialize_database
from indicator_config_parser import parse_indicators_json, get_indicator_parameters
from indicators import compute_indicator
from load_data import load_data
from config import DB_PATH
from logging_setup import configure_logging

logger = logging.getLogger(__name__)

def fetch_ao_configurations(conn) -> List[Dict[str, Any]]:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ic.id, ic.config FROM indicator_configs ic
        JOIN indicators i ON ic.indicator_id = i.id
        WHERE i.name = 'ao';
    """)
    rows = cursor.fetchall()
    configurations = [{'id': row[0], **json.loads(row[1])} for row in rows]
    return configurations

def attempt_compute_indicator(data: pd.DataFrame, params: Dict[str, Any], config_id: int) -> bool:
    try:
        input_columns = ['high', 'low']
        parameters = {'type': 'pandas-ta', 'parameters': params}
        compute_indicator(data, 'ao', parameters, input_columns, config_id)
        return True
    except Exception as e:
        logger.error(f"Error computing indicator 'ao' with parameters {params}: {e}")
        return False

def analyze_results(valid_params: List[Dict[str, Any]], invalid_params: List[Dict[str, Any]]) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    valid_ranges = {}
    invalid_ranges = {}
    
    if valid_params:
        for param in valid_params:
            for key, value in param.items():
                valid_ranges.setdefault(key, []).append(value)
        valid_ranges = {k: (min(v), max(v)) for k, v in valid_ranges.items()}
    else:
        valid_ranges = {k: (None, None) for k in ['fast', 'slow']}
    
    if invalid_params:
        for param in invalid_params:
            for key, value in param.items():
                invalid_ranges.setdefault(key, []).append(value)
        invalid_ranges = {k: (min(v), max(v)) for k, v in invalid_ranges.items()}
    else:
        invalid_ranges = {k: (None, None) for k in ['fast', 'slow']}
    
    return valid_ranges, invalid_ranges

def report_results(valid_ranges: Dict[str, Tuple[float, float]], invalid_ranges: Dict[str, Tuple[float, float]]):
    print("\n=== Tweak Trials Results ===")
    logger.error("\n=== Tweak Trials Results ===")
    
    print("\nValid Ranges:")
    logger.error("\nValid Ranges:")
    for param, (min_val, max_val) in valid_ranges.items():
        if min_val is not None and max_val is not None:
            print(f"  {param}: {min_val} to {max_val}")
            logger.error(f"Valid Range - {param}: {min_val} to {max_val}")
        else:
            print(f"  {param}: No valid configurations found.")
            logger.error(f"Valid Range - {param}: No valid configurations found.")
    
    print("\nInvalid Ranges:")
    logger.error("\nInvalid Ranges:")
    for param, (min_val, max_val) in invalid_ranges.items():
        if min_val is not None and max_val is not None:
            print(f"  {param}: {min_val} to {max_val}")
            logger.error(f"Invalid Range - {param}: {min_val} to {max_val}")
        else:
            print(f"  {param}: No invalid configurations found.")
            logger.error(f"Invalid Range - {param}: No invalid configurations found.")

def main():
    configure_logging(log_file_prefix="tweak_trials")
    
    initialize_database(DB_PATH)
    
    conn = create_connection(DB_PATH)
    if not conn:
        logger.error("Failed to connect to the database.")
        print("Failed to connect to the database. Check logs for details.")
        sys.exit(1)
    
    try:
        symbol = input("Enter symbol (e.g., 'BTCUSDT'): ").strip().upper()
        timeframe = input("Enter timeframe (e.g., '1d'): ").strip()
        
        data, is_rev, db_fn = load_data(symbol, timeframe)
        if data.empty:
            logger.error(f"No data found for symbol '{symbol}' and timeframe '{timeframe}'.")
            print(f"No data found for symbol '{symbol}' and timeframe '{timeframe}'. Exiting.")
            sys.exit(1)
        
        configurations = fetch_ao_configurations(conn)
        if not configurations:
            logger.error("No configurations found for indicator 'ao'. Exiting.")
            print("No configurations found for indicator 'ao'. Exiting.")
            sys.exit(1)
        
        valid_params = []
        invalid_params = []
        
        for config in configurations:
            config_id = config.pop('id')
            params = config
            logger.error(f"Attempting configuration {config_id}: {params}")
            success = attempt_compute_indicator(data, params, config_id)
            if success:
                valid_params.append(params)
            else:
                invalid_params.append(params)
        
        valid_ranges, invalid_ranges = analyze_results(valid_params, invalid_params)
        
        report_results(valid_ranges, invalid_ranges)
    
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}. Check logs for details.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
