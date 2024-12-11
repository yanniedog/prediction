# start.py

import os
import sys
import shutil
import re
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import time
import matplotlib
matplotlib.use('Agg')
from joblib import Parallel, delayed
from data_utils import clear_screen, prepare_data, determine_time_interval, get_original_indicators, handle_missing_indicators
from correlation_utils import load_or_calculate_correlations, calculate_correlation
from visualization_utils import generate_combined_correlation_chart, visualize_data
from generate_heatmaps import generate_heatmaps
from backup_utils import run_backup_cleanup
from table_generation import generate_best_indicator_table, generate_statistical_summary, generate_correlation_csv
from binance_historical_data_downloader import download_binance_data
from correlation_database import CorrelationDatabase
from config import DB_PATH
from load_data import load_data
from indicators import compute_all_indicators
from linear_regression import perform_linear_regression
from advanced_analysis import advanced_price_prediction
import logging

def parse_date_time_input(user_input: str, reference_datetime: datetime) -> datetime:
    """
    Parse user input for future date.

    If input is empty, return reference_datetime + 4 weeks.
    """
    user_input = user_input.strip()
    if not user_input:
        return reference_datetime + timedelta(weeks=4)
    try:
        return datetime.strptime(user_input, '%Y%m%d')
    except ValueError:
        logging.error("Invalid date format. Please use YYYYMMDD.")
        sys.exit(1)

def input_with_default(prompt: str, default: str) -> str:
    """
    Prompt the user for input with a default value.
    """
    val = input(prompt).strip()
    return val if val else default

def input_yes_no(prompt: str, default: str = 'y') -> str:
    """
    Prompt the user for a yes/no input with a default value.
    """
    val = input(prompt).strip().lower()
    if not val:
        return default
    if val in ['y', 'yes']:
        return 'y'
    elif val in ['n', 'no']:
        return 'n'
    else:
        return default

def input_yes_no_no_default(prompt: str) -> str:
    """
    Prompt the user for a yes/no input without a default value.
    """
    while True:
        val = input(prompt).strip().lower()
        if val in ['y', 'yes']:
            return 'y'
        elif val in ['n', 'no']:
            return 'n'
        else:
            print("Please enter 'y' or 'n'.")

def preview_database(db_path: str) -> bool:
    """
    Check if the database has essential tables populated.
    """
    try:
        conn = CorrelationDatabase(db_path).connection
        tables = ["symbols", "timeframes", "klines"]
        for table in tables:
            df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 5", conn)
            if df.empty:
                conn.close()
                return False
        conn.close()
        return True
    except Exception as e:
        logging.error(f"Database error: {e}")
        return False

def recreate_database(db_path: str):
    """
    Delete and recreate the database.
    """
    if os.path.exists(db_path):
        os.remove(db_path)
        logging.info(f"Deleted existing database at {db_path}.")
    from sqlite_data_manager import create_connection, create_tables
    conn = create_connection(db_path)
    if conn:
        create_tables(conn)
        conn.close()
        logging.info("Recreated the database with necessary tables.")
    else:
        logging.error("Failed to create database connection.")
        sys.exit(1)

def main() -> None:
    """
    Main function to orchestrate the data processing and analysis workflow.
    """
    clear_screen()
    run_backup_cleanup()
    new_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    reports_dir, csv_dir = 'reports', 'csv'
    predictions_dir = 'predictions'
    for d in [reports_dir, csv_dir, predictions_dir]:
        os.makedirs(d, exist_ok=True)
        logging.info(f"Ensured directory exists: {d}")
    
    # a) Delete 'csv' and 'predictions' directories if user opts
    delete_choice = input_yes_no("Do you want to delete all previously generated output? (y/n) [Default: y]: ", 'y')
    if delete_choice == 'y':
        for folder in ['csv', 'predictions']:
            folder_path = Path(folder).resolve()
            if folder_path.exists():
                shutil.rmtree(folder_path)
                logging.info(f"Deleted folder: {folder_path}")
                # Recreate the directories after deletion
                os.makedirs(folder_path, exist_ok=True)
                logging.info(f"Recreated folder: {folder_path}")
    
    # f) Set default trading symbol and interval
    symbol = input_with_default("Enter the trading symbol (e.g., 'SOLUSDT') [Default: SOLUSDT]: ", "SOLUSDT").upper()
    timeframe = input_with_default("Enter the timeframe (e.g., '1w') [Default: 1w]: ", "1w")
    
    # Prompt for future date after deleting outputs
    today = datetime.now()
    default_future_datetime = today + timedelta(weeks=4)
    default_future_date_str = default_future_datetime.strftime('%Y%m%d')
    future_date_input = input_with_default(
        f"Enter future date to project out to (YYYYMMDD) [Default: {default_future_date_str}]: ",
        default_future_date_str
    )
    future_datetime = parse_date_time_input(future_date_input, today)
    
    # Load data
    data, is_reverse_chronological, db_filename = load_data(symbol, timeframe)
    logging.info(f"Loaded data for symbol: {symbol}, timeframe: {timeframe}")
    
    # Check database integrity
    if not preview_database(DB_PATH):
        recreate_choice = input_yes_no_no_default("Database invalid. Erase and create new? (y/n): ")
        if recreate_choice == 'y':
            recreate_database(DB_PATH)
            data, is_reverse_chronological, db_filename = load_data(symbol, timeframe)
            logging.info("Reloaded data after recreating database.")
        else:
            logging.error("Database not recreated. Exiting.")
            sys.exit(1)
    
    # If no data, prompt to download
    if data.empty:
        download_choice = input_yes_no("No data found. Download now? (y/n) [Default: y]: ", 'y')
        if download_choice == 'y':
            download_binance_data(symbol, timeframe)
            data, is_reverse_chronological, db_filename = load_data(symbol, timeframe)
            logging.info("Downloaded and loaded new data.")
        if data.empty:
            logging.error("No data found after download. Exiting.")
            sys.exit(1)
    
    # Compute indicators
    data = compute_all_indicators(data)
    logging.info("Computed all indicators.")
    
    # Ensure required columns are present
    required_columns = ['Volume', 'Open', 'High', 'Low']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing required columns in data: {missing_columns}")
        sys.exit(1)
    logging.info(f"All required columns are present: {required_columns}")
    
    # Prepare data
    X_scaled, feature_names = prepare_data(data)
    logging.info("Prepared and scaled data.")
    
    # Get original indicators
    original_indicators = handle_missing_indicators(
        get_original_indicators(feature_names, data), 
        data, 
        ['FI', 'KCU_20_2.0', 'STOCHRSI_14_5_3_slowk', 'VI+_14']
    )
    if not original_indicators:
        logging.error("No valid original indicators found.")
        sys.exit(1)
    logging.info(f"Identified original indicators: {original_indicators}")
    
    # Determine max lag
    max_lag = len(data) - 51
    if max_lag < 1:
        logging.error("Not enough data to calculate correlations.")
        sys.exit(1)
    logging.info(f"Maximum lag for correlations: {max_lag}")
    
    # Calculate or load correlations and embed into the database
    load_or_calculate_correlations(
        data=data,
        original_indicators=original_indicators,
        max_lag=max_lag,
        is_reverse_chronological=is_reverse_chronological,
        symbol=symbol,
        timeframe=timeframe
    )
    logging.info("Calculated and embedded correlations into the database.")
    
    # Load correlations from the database
    correlation_db = CorrelationDatabase(DB_PATH)
    correlations = {}
    for indicator in original_indicators:
        correlations[indicator] = correlation_db.get_correlations(symbol, timeframe, indicator, max_lag)
    correlation_db.close()
    logging.info("Loaded correlations from the database.")
    
    # Generate statistical summary
    try:
        summary_df = generate_statistical_summary(correlations, max_lag)
        summary_filepath = os.path.join(csv_dir, f"{new_timestamp}_{symbol}_{timeframe}_statistical_summary.csv")
        os.makedirs(csv_dir, exist_ok=True)
        summary_df.to_csv(summary_filepath, index=True)
        logging.info(f"Generated statistical summary at {summary_filepath}.")
    except Exception as e:
        logging.error(f"Error generating statistical summary: {e}")
    
    # Generate combined correlation chart
    try:
        generate_combined_correlation_chart(
            correlations=correlations,
            max_lag=max_lag,
            time_interval=determine_time_interval(data),
            timestamp=new_timestamp,
            base_csv_filename=f"{symbol}_{timeframe}"
        )
        logging.info("Combined correlation chart generated successfully.")
    except Exception as e:
        logging.error(f"Error generating combined correlation chart: {e}")
    
    # Visualize data and generate individual indicator charts
    try:
        visualize_data(
            data=data,
            features=X_scaled,
            feature_columns=feature_names,
            timestamp=new_timestamp,
            is_reverse_chronological=is_reverse_chronological,
            time_interval=determine_time_interval(data),
            generate_charts=True,
            correlations=correlations,
            calculate_correlation_func=calculate_correlation,
            base_csv_filename=f"{symbol}_{timeframe}"
        )
        logging.info("Visualized data and generated individual indicator charts successfully.")
    except Exception as e:
        logging.error(f"Error visualizing data: {e}")
    
    # Generate heatmaps automatically
    try:
        generate_heatmaps(
            data=data,
            timestamp=new_timestamp,
            time_interval=determine_time_interval(data),
            generate_heatmaps_flag=True,
            correlations=correlations,
            calculate_correlation=calculate_correlation,
            base_csv_filename=f"{symbol}_{timeframe}"
        )
        logging.info("Generated heatmaps successfully.")
    except Exception as e:
        logging.error(f"Error generating heatmaps: {e}")
    
    # Generate best indicator table
    try:
        best_indicators_df = generate_best_indicator_table(correlations, max_lag)
        best_indicators_filepath = os.path.join(csv_dir, f"{new_timestamp}_{symbol}_{timeframe}_best_indicators.csv")
        best_indicators_df.to_csv(best_indicators_filepath, index=False)
        logging.info(f"Generated best indicator table at {best_indicators_filepath}.")
    except Exception as e:
        logging.error(f"Error generating best indicator table: {e}")
    
    # Automatically save correlation CSV
    try:
        generate_correlation_csv(correlations, max_lag, f"{symbol}_{timeframe}", csv_dir)
        logging.info(f"Saved correlation CSV at {csv_dir}.")
    except Exception as e:
        logging.error(f"Error generating correlation CSV: {e}")
    
    # Perform Linear Regression Prediction
    try:
        perform_linear_regression(
            data=data,
            correlations=correlations,
            max_lag=max_lag,
            time_interval=determine_time_interval(data),
            timestamp=new_timestamp,
            base_csv_filename=f"{symbol}_{timeframe}",
            future_datetime=future_datetime,
            lag_periods=4  # Assuming 4 weeks ahead
        )
        logging.info("Performed linear regression predictions successfully.")
    except Exception as e:
        logging.error(f"Error performing linear regression: {e}")
    
    # Perform Advanced Price Prediction
    try:
        advanced_price_prediction(
            data=data,
            correlations=correlations,
            max_lag=max_lag,
            time_interval=determine_time_interval(data),
            timestamp=new_timestamp,
            base_csv_filename=f"{symbol}_{timeframe}",
            future_datetime=future_datetime,
            lag_periods=4  # Assuming 4 weeks ahead
        )
        logging.info("Performed advanced price prediction successfully.")
    except Exception as e:
        logging.error(f"Error performing advanced price prediction: {e}")

if __name__ == "__main__":
    main()
