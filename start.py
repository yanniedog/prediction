# start.py
import logging
import os
import subprocess
import sys
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from pathlib import Path
from datetime import datetime
from sqlite_data_manager import create_connection, initialize_database, create_tables, DB_PATH
from tweak_indicator import fetch_available_indicators, insert_tweaked_configs, generate_configurations, compute_all_indicators, compute_configured_indicators
from correlation_utils import load_or_calculate_correlations

# Configure logging
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
    if level == "info":
        logger.info(message)
    elif level == "error":
        logger.error(message)
    print(message)

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
    Run the tweak_indicator script to allow user to select and configure indicators.
    """
    from tweak_indicator import main as tweak_main
    # Simulate command-line arguments
    sys.argv = ['tweak_indicator.py', symbol, timeframe]
    tweak_main()
    # Retrieve selected configurations from the database
    conn = create_connection(DB_PATH)
    if not conn:
        log_and_print("Database connection failed after tweaking indicators.", "error")
        sys.exit(1)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM indicators WHERE name LIKE ?", ('%_%',))  # Fetch all configured indicators
    rows = cursor.fetchall()
    conn.close()
    configurations = [row[0] for row in rows]
    return configurations

def main():
    """Main execution flow."""
    # Configure logging
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"execution_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    configure_logging(log_file)

    try:
        # Initialize database
        initialize_database(DB_PATH)
        log_and_print("Database initialized successfully.")

        # Delete previous outputs if user chooses
        delete_previous_output()

        selected_indicator = None
        symbol = None
        timeframe = None
        configurations = []

        # Prompt user to tweak indicator settings
        if input_yes_no("Do you want to tweak indicator settings? (y/n): ") == 'y':
            symbol = input_with_default("Enter symbol (e.g., 'BTCUSDT'): ", "BTCUSDT")
            timeframe = input_with_default("Enter timeframe (e.g., '1w'): ", "1w")
            configurations = run_tweak_indicator(symbol, timeframe)
            if configurations:
                selected_indicator = configurations[0].split('_')[0]  # Assuming first config's base indicator
                log_and_print(f"Selected indicator: {selected_indicator}")
            else:
                log_and_print("No configurations found after tweaking indicators.", "error")
                sys.exit(1)

        log_and_print("Proceeding with main execution.")

        # Download Binance data
        from binance_historical_data_downloader import download_binance_data
        download_binance_data(symbol, timeframe, DB_PATH)
        log_and_print("Price data steps completed.")

        if selected_indicator and configurations:
            # Load data from SQLite
            from load_data import load_data
            data, is_rev, db_fn = load_data(symbol, timeframe)
            if data.empty:
                log_and_print("No data available for correlation computation.", "error")
                sys.exit(1)

            # Compute default indicators
            data = compute_all_indicators(data)
            log_and_print("Default indicators computed.")

            # Compute configured indicators
            data = compute_configured_indicators(data, configurations)
            log_and_print("Configured indicators computed.")

            # Perform correlation computations
            load_or_calculate_correlations(
                data,
                indicators=configurations,
                max_lag=30,
                reverse=False,
                symbol=symbol,
                timeframe=timeframe
            )
            log_and_print("Correlation computations completed for the selected indicator configurations.")

        log_and_print("Execution finished successfully.")

    except Exception as e:
        logger.exception("An unexpected error occurred during execution.")
        sys.exit(1)

if __name__ == "__main__":
    main()
