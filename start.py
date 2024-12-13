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
from sqlite_data_manager import initialize_database, create_connection, create_tables
from tweak_indicator import fetch_available_indicators, insert_indicator_configs, generate_configurations
from correlation_utils import load_or_calculate_correlations
from indicators import compute_all_indicators, compute_configured_indicators
import indicators

def configure_logging(log_file):
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d(%(funcName)s)]: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

logger = logging.getLogger()

def log_and_print(message, level="info"):
    """Log and print messages."""
    print(message)  # Print to screen
    if level == "info":
        logger.info(message)  # Log to file only
    elif level == "error":
        logger.error(message)  # Log to file only


def input_with_default(prompt: str, default: str) -> str:
    """Prompt user for input with a default value."""
    val = input(prompt).strip()
    final_val = val if val else default
    logger.info(f"User Input: {prompt.strip()} -> {final_val}")
    return final_val

def input_yes_no(prompt: str, default: str = 'y') -> str:
    """Prompt user for a yes/no input with a default value."""
    val = input(prompt).strip().lower()
    if val.startswith('y'):
        final_val = 'y'
    elif val.startswith('n'):
        final_val = 'n'
    else:
        final_val = default
    logger.info(f"User Input: {prompt.strip()} -> {final_val}")
    return final_val

def delete_previous_output():
    """Delete contents of specified output directories based on user input."""
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
    """
    Run the tweak_indicator.py script to generate configurations.
    """
    available_indicators = fetch_available_indicators(indicators)
    if not available_indicators:
        log_and_print("No indicators available. Check `indicators.py` or `compute_all_indicators`.", "error")
        sys.exit(1)
    
    log_and_print("Available indicators:")
    for idx, indicator in enumerate(available_indicators, 1):
        log_and_print(f"{idx}. {indicator}")
    
    choice = input("Select an indicator by number: ").strip()
    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(available_indicators):
        log_and_print("Invalid choice. Exiting.", "error")
        sys.exit(1)
    
    selected_indicator = available_indicators[int(choice) - 1]
    log_and_print(f"Selected indicator: {selected_indicator}")
    
    default_params = {"timeperiod": 14}
    
    configurations = generate_configurations(default_params.keys(), default_params)
    if not configurations:
        log_and_print(f"No configurations generated for '{selected_indicator}'.", "error")
        sys.exit(1)
    log_and_print(f"Generated {len(configurations)} configurations for '{selected_indicator}'.")
    
    conn = create_connection()
    if not conn:
        log_and_print("Failed to connect to the database.", "error")
        sys.exit(1)
    insert_indicator_configs(conn, selected_indicator, configurations)
    conn.close()
    log_and_print(f"Configurations for '{selected_indicator}' have been added to the database.")
    
    return selected_indicator

def get_selected_indicator_configs(indicator_name: str):
    """Retrieve all configurations for the selected indicator from the database."""
    conn = create_connection()
    if not conn:
        log_and_print("Database connection failed.", "error")
        sys.exit(1)
    create_tables(conn)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM indicators WHERE name LIKE ? ESCAPE '\\'", (f"%\\_%",))
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows if row[0].startswith(indicator_name + "_")]

def main():
    """Main execution flow."""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"execution_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    configure_logging(log_file)

    try:
        initialize_database(DB_PATH)
        log_and_print("Database initialized successfully.")

        delete_previous_output()

        selected_indicator = None
        symbol = None
        timeframe = None

        if input_yes_no("Do you want to tweak indicator settings? (y/n): ") == 'y':
            symbol = input_with_default("Enter symbol (e.g., 'BTCUSDT'): ", "BTCUSDT")
            timeframe = input_with_default("Enter timeframe (e.g., '1w'): ", "1w")
            log_and_print(f"Symbol: {symbol}, Timeframe: {timeframe}")

            selected_indicator = run_tweak_indicator(symbol, timeframe)

        else:
            log_and_print("Skipping indicator tweaking.")

        log_and_print("Proceeding with main execution.")

        from binance_historical_data_downloader import download_binance_data
        if symbol and timeframe:
            download_binance_data(symbol, timeframe, DB_PATH)
            log_and_print("Price data download and insertion completed.")
        else:
            log_and_print("Symbol and timeframe not provided. Skipping data download.", "error")
            sys.exit(1)

        if selected_indicator:
            configs = get_selected_indicator_configs(selected_indicator)
            if not configs:
                log_and_print("No configurations found for the selected indicator.", "error")
                sys.exit(1)

            from load_data import load_data as load_db_data
            data, is_rev, db_fn = load_db_data(symbol, timeframe)
            if data.empty:
                log_and_print("No data available for the selected symbol and timeframe.", "error")
                sys.exit(1)

            try:
                data = compute_all_indicators(data)
                log_and_print("Default indicators computed successfully.")
            except Exception as e:
                log_and_print(f"Error computing default indicators: {e}", "error")
                sys.exit(1)

            try:
                data = compute_configured_indicators(data, configs)
                log_and_print("Configured indicators computed successfully.")
            except Exception as e:
                log_and_print(f"Error computing configured indicators: {e}", "error")
                sys.exit(1)

            load_or_calculate_correlations(
                data=data,
                indicators=configs,
                max_lag=30,
                reverse=False,
                symbol=symbol,
                timeframe=timeframe
            )
            log_and_print("Correlation computations completed for the selected indicator configurations.")
        else:
            log_and_print("No indicators selected for correlation computations.")

        log_and_print("Execution finished successfully.")

    except Exception as e:
        logger.exception("An unexpected error occurred during execution.")
        sys.exit(1)

if __name__ == "__main__":
    main()
