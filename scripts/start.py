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
from visualization_utils import generate_combined_correlation_chart, visualize_data, generate_individual_indicator_chart
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
    user_input = user_input.strip()
    if not user_input:
        return reference_datetime + timedelta(weeks=4)  # Default to 4 weeks ahead
    try:
        return datetime.strptime(user_input, '%Y%m%d')
    except ValueError:
        logging.error("Invalid date format. Please use YYYYMMDD.")
        sys.exit(1)

def input_with_default(prompt: str, default: str) -> str:
    val = input(prompt).strip()
    return val if val else default

def get_future_datetime():
    while True:
        user_input = input("Enter future date to project out to (YYYYMMDD) [Default: 4 weeks ahead]: ").strip()
        if not user_input:
            return datetime.now() + timedelta(weeks=4)
        try:
            return datetime.strptime(user_input, '%Y%m%d')
        except ValueError:
            print("Invalid format. Please use YYYYMMDD.")

def preview_database(db_path: str):
    try:
        conn = CorrelationDatabase(db_path).connection
        tables = ["symbols", "timeframes", "klines"]
        for table in tables:
            df = pd.read_sql_query(f"SELECT * FROM {table} {'LIMIT 5' if table == 'klines' else ''}", conn)
            if df.empty:
                return False
        conn.close()
        return True
    except Exception as e:
        logging.error(f"Database error: {e}")
        return False

def recreate_database(db_path: str):
    if os.path.exists(db_path):
        os.remove(db_path)
    from sqlite_data_manager import create_connection, create_tables
    conn = create_connection(db_path)
    if conn:
        create_tables(conn)
        conn.close()
    else:
        logging.error("Failed to create database connection.")
        sys.exit(1)

def main() -> None:
    clear_screen()
    run_backup_cleanup()
    new_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    reports_dir, csv_dir = 'reports', 'csv'
    for d in [reports_dir, csv_dir]:
        os.makedirs(d, exist_ok=True)

    # a) Delete 'csv' and 'predictions' directories if user opts
    if input("Do you want to delete all previously generated output? (y/n) [Default: y]: ").strip().lower() in ['y', 'yes', '']:
        for folder in ['csv', 'predictions']:
            folder_path = Path(folder).resolve()
            if folder_path.exists():
                shutil.rmtree(folder_path)
                logging.info(f"Deleted folder: {folder_path}")

    # b) Database is already stored in its own subfolder via config.py

    # c) Always generate individual indicator charts
    generate_charts = True

    # d) Always generate heatmaps
    generate_heatmaps_flag = True

    # e) Automatically save correlation CSV
    save_correlation_csv = True

    # f) Set default trading symbol and interval
    symbol = input_with_default("Enter the trading symbol (e.g., 'SOLUSDT') [Default: SOLUSDT]: ", "SOLUSDT").upper()
    timeframe = input_with_default("Enter the timeframe (e.g., '1w') [Default: 1w]: ", "1w")

    data, is_reverse_chronological, db_filename = load_data(symbol, timeframe)

    if not preview_database(DB_PATH):
        if input("Database invalid. Erase and create new? (y/n) [Default: y]: ").strip().lower() in ['y', 'yes', '']:
            recreate_database(DB_PATH)
            data, is_reverse_chronological, db_filename = load_data(symbol, timeframe)
        else:
            logging.error("Database not recreated. Exiting.")
            sys.exit(1)

    if data.empty:
        if input("No data found. Download now? (y/n) [Default: y]: ").strip().lower() in ['y', 'yes', '']:
            download_binance_data(symbol, timeframe)
            data, is_reverse_chronological, db_filename = load_data(symbol, timeframe)
        if data.empty:
            logging.error("No data found.")
            sys.exit(1)

    data = compute_all_indicators(data)

    # Rename columns if necessary
    for col_map in [('Date', 'open_time'), ('Close', 'close')]:
        if col_map[0] not in data.columns and col_map[1] in data.columns:
            data.rename(columns={col_map[1]: col_map[0]}, inplace=True)

    if 'Date' not in data.columns or 'Close' not in data.columns:
        logging.error("Required columns missing.")
        sys.exit(1)

    time_interval = determine_time_interval(data)
    X_scaled, feature_names = prepare_data(data)
    original_indicators = handle_missing_indicators(get_original_indicators(feature_names, data), data, ['FI', 'KCU_20_2.0', 'STOCHRSI_14_5_3_slowk', 'VI+_14'])

    if not original_indicators:
        logging.error("No valid original indicators found.")
        sys.exit(1)

    max_lag = len(data) - 51
    if max_lag < 1:
        logging.error("Not enough data to calculate correlations.")
        sys.exit(1)

    # Calculate or load correlations and embed into the database
    load_or_calculate_correlations(data, original_indicators, max_lag, is_reverse_chronological, symbol, timeframe)

    # Load correlations from the database
    correlation_db = CorrelationDatabase(DB_PATH)
    correlations = {indicator: [correlation_db.get_correlation(symbol, timeframe, indicator, lag) for lag in range(1, max_lag + 1)] for indicator in original_indicators}
    correlation_db.close()

    # Save statistical summary
    try:
        summary_df = generate_statistical_summary(correlations, max_lag)
        summary_df.to_csv(os.path.join(csv_dir, f"{new_timestamp}_{symbol}_{timeframe}_statistical_summary.csv"), index=True)
    except Exception as e:
        logging.error(f"Error generating statistical summary: {e}")

    # Generate combined correlation chart
    try:
        generate_combined_correlation_chart(correlations, max_lag, time_interval, new_timestamp, f"{symbol}_{timeframe}")
    except Exception as e:
        logging.error(f"Error generating combined correlation chart: {e}")

    # Always visualize data and generate individual charts
    if generate_charts:
        try:
            visualize_data(data, X_scaled, feature_names, new_timestamp, is_reverse_chronological, time_interval, generate_charts, correlations, calculate_correlation, f"{symbol}_{timeframe}")
        except Exception as e:
            logging.error(f"Error visualizing data: {e}")

    # Always generate heatmaps
    try:
        generate_heatmaps(data, new_timestamp, time_interval, generate_heatmaps_flag, correlations, calculate_correlation, f"{symbol}_{timeframe}")
    except Exception as e:
        logging.error(f"Error generating heatmaps: {e}")

    # Generate best indicator table
    try:
        best_indicators_df = generate_best_indicator_table(correlations, max_lag)
        best_indicators_df.to_csv(os.path.join(csv_dir, f"{new_timestamp}_{symbol}_{timeframe}_best_indicators.csv"), index=False)
    except Exception as e:
        logging.error(f"Error generating best indicator table: {e}")

    # Automatically save correlation CSV
    if save_correlation_csv:
        try:
            generate_correlation_csv(correlations, max_lag, f"{symbol}_{timeframe}", csv_dir)
        except Exception as e:
            logging.error(f"Error generating correlation CSV: {e}")

    # Future date input with syntax guide and default to 4 weeks ahead
    future_datetime = get_future_datetime()

    # Perform Linear Regression Prediction
    try:
        perform_linear_regression(data, correlations, max_lag, time_interval, new_timestamp, f"{symbol}_{timeframe}", future_datetime, lag_periods=4)  # Assuming lag_periods=4 for 4 weeks
    except Exception as e:
        logging.error(f"Error performing linear regression: {e}")

    # Perform Advanced Price Prediction
    try:
        advanced_price_prediction(data, correlations, max_lag, time_interval, new_timestamp, f"{symbol}_{timeframe}", future_datetime, lag_periods=4)  # Assuming lag_periods=4 for 4 weeks
    except Exception as e:
        logging.error(f"Error performing advanced price prediction: {e}")

if __name__ == "__main__":
    main()
