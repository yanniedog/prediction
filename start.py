# start.py
import logging
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime
from config import DB_PATH
from sqlite_data_manager import initialize_database, create_connection, insert_indicator_configs
from tweak_indicator import (
    fetch_available_indicators,
    generate_configurations,
    parse_indicator_parameters
)
from correlation_utils import load_or_calculate_correlations
from indicators import compute_configured_indicators
from load_data import load_data
from binance_historical_data_downloader import download_binance_data
from logging_setup import configure_logging

logger = logging.getLogger()

def input_with_default(prompt: str, default: str) -> str:
    val = input(prompt).strip()
    final_val = val if val else default
    logger.info(f"User Input: {prompt.strip()} -> {final_val}")
    return final_val

def input_yes_no(prompt: str, default: str = 'y') -> str:
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
                    logger.info(f"Cleared contents of: {folder}")
                except Exception as e:
                    logger.error(f"Error clearing contents of {folder}: {e}")
        logger.info("All selected directory contents have been cleared.")
    else:
        logger.info("Directory contents not cleared.")

def run_tweak_indicator(symbol: str, timeframe: str):
    available_indicators = fetch_available_indicators()
    if not available_indicators:
        logger.error("No indicators available. Check `indicators.py` or `compute_all_indicators`.")
        sys.exit(1)

    logger.info("Available indicators:")
    for idx, indicator in enumerate(available_indicators, 1):
        logger.info(f"{idx}. {indicator}")

    choice = input("Select an indicator by number: ").strip()
    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(available_indicators):
        logger.error("Invalid choice. Exiting.")
        sys.exit(1)

    selected_indicator = available_indicators[int(choice) - 1]
    logger.info(f"Selected indicator: {selected_indicator}")

    parameters = parse_indicator_parameters(selected_indicator)
    if not parameters:
        logger.error(f"No parameters found for '{selected_indicator}'. Using base indicator.")
    else:
        logger.info(f"Parameters for '{selected_indicator}': {parameters}")

    configurations = generate_configurations(parameters.keys(), parameters) if parameters else []
    if not configurations:
        logger.error(f"No configurations generated for '{selected_indicator}'. Using base indicator.")
    else:
        logger.info(f"Generated {len(configurations)} configurations for '{selected_indicator}'.")
        example_configs = configurations[:15]
        logger.info(f"Example configurations for '{selected_indicator}': {example_configs}")

    conn = create_connection(DB_PATH)
    if not conn:
        logger.error("Failed to connect to the database.")
        sys.exit(1)
    try:
        insert_indicator_configs(conn, selected_indicator, configurations)
    except Exception as e:
        logger.error(f"Error inserting configurations into database for '{selected_indicator}': {e}")
        sys.exit(1)
    if configurations:
        logger.info("Configurations stored in database:")
        for config in configurations[:15]:
            logger.info(config)
    conn.close()
    logger.info(f"Configurations for '{selected_indicator}' have been added to the database.")

    return selected_indicator

def main():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    configure_logging(log_file_prefix="execution")

    try:
        initialize_database(DB_PATH)
        logger.info("Database initialized successfully.")

        delete_previous_output()

        selected_indicator = None
        symbol = None
        timeframe = None

        if input_yes_no("Do you want to tweak indicator settings? (y/n): ") == 'y':
            symbol = input_with_default("Enter symbol (e.g., 'BTCUSDT'): ", "BTCUSDT")
            timeframe = input_with_default("Enter timeframe (e.g., '1w'): ", "1w")
            logger.info(f"Symbol: {symbol}, Timeframe: {timeframe}")

            selected_indicator = run_tweak_indicator(symbol, timeframe)

        else:
            logger.info("Skipping indicator tweaking.")

        logger.info("Proceeding with main execution.")

        from binance_historical_data_downloader import download_binance_data
        if symbol and timeframe:
            download_binance_data(symbol, timeframe, DB_PATH)
            logger.info("Price data download and insertion completed.")
        else:
            logger.error("Symbol and timeframe not provided. Skipping data download.")
            sys.exit(1)

        if selected_indicator:
            indicators_list = [selected_indicator]

            data, is_rev, db_fn = load_data(symbol, timeframe)
            if data.empty:
                logger.error("No data available for the selected symbol and timeframe.")
                sys.exit(1)

            try:
                data = compute_configured_indicators(data, indicators_list, db_path=DB_PATH)
                logger.info("Configured indicators computed successfully.")
                logger.info("Computing correlations.")
                load_or_calculate_correlations(
                    data=data,
                    indicators=indicators_list,
                    max_lag=30,
                    reverse=False,
                    symbol=symbol,
                    timeframe=timeframe
                )
                logger.info("Correlation computations completed for the selected indicator configurations.")
            except Exception as e:
                logger.error(f"Error computing configured indicators or correlations: {e}")
                sys.exit(1)
        else:
            logger.info("No indicators selected for correlation computations.")

        logger.info("Execution finished successfully.")

    except Exception as e:
        logger.exception("An unexpected exception occurred during execution.")
        sys.exit(1)

if __name__ == "__main__":
    main()
