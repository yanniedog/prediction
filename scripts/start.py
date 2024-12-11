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
    user_input = user_input.strip()
    if not user_input:
        return reference_datetime
    m = re.match(r'^([+-]\d+)([smhdw])$', user_input)
    if m:
        amount = int(m.group(1))
        unit_map = {'s': 'seconds', 'm': 'minutes', 'h': 'hours', 'd': 'days', 'w': 'weeks'}
        return reference_datetime + timedelta(**{unit_map[m.group(2)]: amount})
    for fmt in ['%Y%m%d-%H%M', '%Y%m%d']:
        try:
            return datetime.strptime(user_input, fmt)
        except:
            pass
    from dateutil import parser
    return parser.parse(user_input, fuzzy=True)

def input_with_default(prompt: str, default: str) -> str:
    val = input(prompt).strip()
    return val if val else default

def input_yes_no(prompt: str, default: str = 'y') -> str:
    val = input(prompt).strip().lower()
    if val not in ['y', 'n', 'yes', 'no', '']:
        return default
    return 'y' if val.startswith('y') or val == '' and default == 'y' else 'n'

def input_yes_no_no_default(prompt: str) -> str:
    while True:
        val = input(prompt).strip().lower()
        if val in ['y', 'yes']:
            return 'y'
        elif val in ['n', 'no']:
            return 'n'

def preview_database(db_path: str):
    try:
        conn = CorrelationDatabase(db_path).connection
        symbols_df = pd.read_sql_query("SELECT * FROM symbols", conn)
        timeframes_df = pd.read_sql_query("SELECT * FROM timeframes", conn)
        klines_df = pd.read_sql_query("SELECT * FROM klines LIMIT 5", conn)
        conn.close()
        if symbols_df.empty or timeframes_df.empty or klines_df.empty:
            return False
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
    reports_dir = 'reports'
    csv_dir = 'csv'
    for d in [reports_dir, csv_dir]:
        os.makedirs(d, exist_ok=True)
    delete_output = input_yes_no("Do you want to delete all previously generated output? (y/n): ", 'y')
    if delete_output == 'y':
        for folder in ['indicator_charts', 'heatmaps', 'combined_charts', reports_dir]:
            folder_path = Path(folder).resolve()
            if folder_path.exists():
                for filename in os.listdir(folder_path):
                    file_path = folder_path / filename
                    try:
                        if file_path.is_file() or file_path.is_symlink():
                            file_path.unlink()
                        elif file_path.is_dir():
                            shutil.rmtree(file_path)
                    except Exception as e:
                        logging.error(f"Error deleting file/directory {file_path}: {e}")
    generate_charts = (input_yes_no("Do you want to generate individual indicator charts? (y/n): ", 'y') == 'y')
    generate_heatmaps_flag = (input_yes_no("Do you want to generate heatmaps? (y/n): ", 'y') == 'y')
    save_correlation_csv = (input_yes_no("Do you want to save correlation CSV? (y/n): ", 'y') == 'y')
    symbol = input_with_default("Enter the trading symbol (e.g., 'BTCUSDT'): ", "BTCUSDT").upper()
    timeframe = input_with_default("Enter the timeframe (e.g., '1d'): ", "1d")
    data, is_reverse_chronological, db_filename = load_data(symbol, timeframe)
    if not preview_database(DB_PATH):
        choice = input_yes_no_no_default("Database invalid. Erase and create new? (y/n): ")
        if choice == 'y':
            recreate_database(DB_PATH)
            data, is_reverse_chronological, db_filename = load_data(symbol, timeframe)
            if data.empty:
                if input_yes_no("Download data now? (y/n): ", 'y') == 'y':
                    download_binance_data(symbol, timeframe)
                    data, is_reverse_chronological, db_filename = load_data(symbol, timeframe)
                    if data.empty:
                        logging.error("No data found after download.")
                        sys.exit(1)
                else:
                    logging.error("No data found.")
                    sys.exit(1)
        else:
            logging.error("Database not recreated. Exiting.")
            sys.exit(1)
    if data.empty:
        if input_yes_no("No data found. Download now? (y/n): ", 'y') == 'y':
            download_binance_data(symbol, timeframe)
            data, is_reverse_chronological, db_filename = load_data(symbol, timeframe)
            if data.empty:
                logging.error("No data found after download.")
                sys.exit(1)
        else:
            logging.info("No data found. Exiting.")
            sys.exit(0)
    data = compute_all_indicators(data)
    if 'Date' not in data.columns:
        if 'open_time' in data.columns:
            data['Date'] = pd.to_datetime(data['open_time'])
        else:
            logging.error("No 'Date' or 'open_time' column found.")
            sys.exit(1)
    if 'Close' not in data.columns:
        if 'close' in data.columns:
            data.rename(columns={'close': 'Close'}, inplace=True)
        else:
            logging.error("No 'Close' column found.")
            sys.exit(1)
    time_interval = determine_time_interval(data)
    X_scaled, feature_names = prepare_data(data)
    original_indicators = get_original_indicators(feature_names, data)
    expected_indicators = ['FI', 'KCU_20_2.0', 'STOCHRSI_14_5_3_slowk', 'VI+_14']
    original_indicators = handle_missing_indicators(original_indicators, data, expected_indicators)
    if not original_indicators:
        logging.error("No valid original indicators found.")
        sys.exit(1)
    max_lag = len(data) - 51
    if max_lag < 1:
        logging.error("Not enough data to calculate correlations.")
        sys.exit(1)
    load_or_calculate_correlations(data, original_indicators, max_lag, is_reverse_chronological, symbol, timeframe)
    correlation_db = CorrelationDatabase(DB_PATH)
    correlations = {}
    for indicator in original_indicators:
        vals = []
        for lag in range(1, max_lag + 1):
            v = correlation_db.get_correlation(symbol, timeframe, indicator, lag)
            vals.append(v)
        correlations[indicator] = vals
    correlation_db.close()
    try:
        summary_df = generate_statistical_summary(correlations, max_lag)
        summary_csv = os.path.join(csv_dir, f"{new_timestamp}_{symbol}_{timeframe}_statistical_summary.csv")
        summary_df.to_csv(summary_csv, index=True)
    except Exception as e:
        logging.error(f"Error generating statistical summary: {e}")
    try:
        generate_combined_correlation_chart(correlations, max_lag, time_interval, new_timestamp, f"{symbol}_{timeframe}")
    except Exception as e:
        logging.error(f"Error generating combined correlation chart: {e}")
    if generate_charts:
        try:
            visualize_data(data, X_scaled, feature_names, new_timestamp, is_reverse_chronological, time_interval, generate_charts, correlations, calculate_correlation, f"{symbol}_{timeframe}")
        except Exception as e:
            logging.error(f"Error visualizing data: {e}")
    if generate_heatmaps_flag:
        try:
            generate_heatmaps(data, new_timestamp, time_interval, generate_heatmaps_flag, correlations, calculate_correlation, f"{symbol}_{timeframe}")
        except Exception as e:
            logging.error(f"Error generating heatmaps: {e}")
    try:
        best_indicators_df = generate_best_indicator_table(correlations, max_lag)
        best_indicators_csv = os.path.join(csv_dir, f"{new_timestamp}_{symbol}_{timeframe}_best_indicators.csv")
        best_indicators_df.to_csv(best_indicators_csv, index=False)
    except Exception as e:
        logging.error(f"Error generating best indicator table: {e}")
    if save_correlation_csv:
        try:
            generate_correlation_csv(correlations, max_lag, f"{symbol}_{timeframe}", csv_dir)
        except Exception as e:
            logging.error(f"Error generating correlation CSV: {e}")
    data['Date'] = pd.to_datetime(data['Date'])
    latest_date_in_data = data['Date'].max()
    current_datetime = datetime.now()
    time_interval_seconds_map = {'second': 1, 'minute': 60, 'hour': 3600, 'day': 86400, 'week': 604800}
    if time_interval not in time_interval_seconds_map:
        logging.error("Invalid time interval.")
        sys.exit(1)
    time_diff_seconds = (current_datetime - latest_date_in_data).total_seconds()
    lag_periods_behind_current = int(time_diff_seconds / time_interval_seconds_map[time_interval])
    future_input = input_with_default("Future date/time: ", "")
    future_datetime = parse_date_time_input(future_input, current_datetime) if future_input else current_datetime
    lag_seconds = (future_datetime - latest_date_in_data).total_seconds()
    if lag_seconds <= 0:
        logging.error("Future date/time must be after the latest date in data.")
        sys.exit(1)
    lag_periods = int(lag_seconds / time_interval_seconds_map[time_interval])
    if lag_periods < 1 or lag_periods > max_lag:
        logging.error("Invalid lag periods.")
        sys.exit(1)
    try:
        perform_linear_regression(data, correlations, max_lag, time_interval, new_timestamp, f"{symbol}_{timeframe}", future_datetime, lag_periods)
    except Exception as e:
        logging.error(f"Error performing linear regression: {e}")
    try:
        advanced_price_prediction(data, correlations, max_lag, time_interval, new_timestamp, f"{symbol}_{timeframe}", future_datetime, lag_periods)
    except Exception as e:
        logging.error(f"Error performing advanced price prediction: {e}")

if __name__ == "__main__":
    main()