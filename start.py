# start.py
import logging
import os
import subprocess
import sys
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from datetime import datetime
from config import DB_PATH
from indicators import compute_all_indicators
from tweak_indicator import fetch_available_indicators, insert_tweaked_configs, generate_configurations
from sqlite_data_manager import initialize_database, create_connection, create_tables
from correlation_utils import load_or_calculate_correlations

logger = logging.getLogger()

def log_and_print(message, level="info"):
    if level == "info":
        logger.info(message)
    elif level == "error":
        logger.error(message)
    print(message)

def input_with_default(prompt: str, default: str) -> str:
    val = input(prompt).strip()
    final_val = val if val else default
    logger.info(f"User Input: {prompt.strip()} -> {final_val}")
    return final_val

def input_yes_no(prompt: str, default: str = 'y') -> str:
    val = input(prompt).strip().lower()
    final_val = 'y' if val.startswith('y') else 'n' if val.startswith('n') else default
    logger.info(f"User Input: {prompt.strip()} -> {final_val}")
    return final_val

def delete_previous_output():
    target_dirs = ['csv', 'heatmaps', 'combined_charts', 'reports', 'database']
    delete_outputs = input_yes_no("Delete contents of output directories? (y/n): ")
    if delete_outputs == 'y':
        for folder in target_dirs:
            folder_path = Path(folder)
            if folder_path.exists() and folder_path.is_dir():
                try:
                    for item in folder_path.iterdir():
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                    log_and_print(f"Cleared contents of: {folder}")
                except Exception as e:
                    log_and_print(f"Error clearing contents of {folder}: {e}", "error")
        log_and_print("All selected directory contents have been cleared.")
    else:
        log_and_print("Directory contents not cleared.")

def run_tweak_indicator(symbol: str, timeframe: str):
    indicators = fetch_available_indicators()
    if not indicators:
        log_and_print("No indicators available. Check `indicators.py` or `compute_all_indicators`.", "error")
        sys.exit(1)
    log_and_print("Available indicators:")
    for idx, indicator in enumerate(indicators, 1):
        log_and_print(f"{idx}. {indicator}")
    choice = input("Select an indicator by number: ").strip()
    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(indicators):
        log_and_print("Invalid choice. Exiting.", "error")
        sys.exit(1)
    selected_indicator = indicators[int(choice) - 1]
    log_and_print(f"Selected indicator: {selected_indicator}")
    default_params = {"timeperiod": 14}
    configurations = generate_configurations(default_params.keys(), default_params)
    insert_tweaked_configs(selected_indicator, configurations)
    log_and_print(f"Configurations for {selected_indicator} added to the database.")
    return selected_indicator

def get_selected_indicator_configs(indicator_name: str):
    conn = create_connection(DB_PATH)
    if not conn:
        log_and_print("Database connection failed.", "error")
        sys.exit(1)
    create_tables(conn)  # Ensure tables are present
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM indicators WHERE name LIKE ?", (f"{indicator_name}_%",))
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]

def main():
    initialize_database(DB_PATH)  # Ensure database is initialized
    delete_previous_output()
    selected_indicator = None
    symbol = None
    timeframe = None
    if input_yes_no("Do you want to tweak indicator settings? (y/n): ") == 'y':
        symbol = input_with_default("Enter symbol (e.g., 'BTCUSDT'): ", "BTCUSDT")
        timeframe = input_with_default("Enter timeframe (e.g., '1w'): ", "1w")
        selected_indicator = run_tweak_indicator(symbol, timeframe)
    log_and_print("Proceeding with main execution.")
    # Continue with price data steps...
    # Example: Load data
    from binance_historical_data_downloader import download_binance_data
    download_binance_data(symbol, timeframe, DB_PATH)
    log_and_print("Price data steps completed.")
    if selected_indicator:
        configs = get_selected_indicator_configs(selected_indicator)
        if not configs:
            log_and_print("No configurations found for the selected indicator.", "error")
            sys.exit(1)
        # Proceed to calculate correlations only for the selected indicator's configurations
        # Assuming you have the data loaded as 'data' DataFrame
        from load_data import load_data
        data, is_rev, db_fn = load_data(symbol, timeframe)
        if data.empty:
            log_and_print("No data available for correlation computation.", "error")
            sys.exit(1)
        indicators_to_process = configs
        load_or_calculate_correlations(data, indicators_to_process, max_lag=30, reverse=False, symbol=symbol, timeframe=timeframe)
        log_and_print("Correlation computations completed for the selected indicator configurations.")
    log_and_print("Execution finished.")

if __name__ == "__main__":
    main()
