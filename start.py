# start.py
import os
import sys
import shutil
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import List
import numpy as np
import pandas as pd
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import dateutil.parser
from joblib import Parallel, delayed
from scipy.stats import t
from sklearn.preprocessing import StandardScaler

from linear_regression import perform_linear_regression
from advanced_analysis import advanced_price_prediction
from load_data import load_data
from indicators import compute_all_indicators
from data_utils import clear_screen, prepare_data, determine_time_interval, get_original_indicators, handle_missing_indicators
from correlation_utils import load_or_calculate_correlations, calculate_correlation
from visualization_utils import generate_combined_correlation_chart, visualize_data
from generate_heatmaps import generate_heatmaps
from backup_utils import run_backup_cleanup
from table_generation import generate_best_indicator_table, generate_statistical_summary, generate_correlation_csv
from binance_historical_data_downloader import download_binance_data
from correlation_database import CorrelationDatabase
from config import DB_PATH
import sqlite3

def parse_date_time_input(user_input: str, reference_datetime: datetime) -> datetime:
    print("[DEBUG] parse_date_time_input: start")
    user_input = user_input.strip()
    if not user_input:
        print("[DEBUG] parse_date_time_input: user_input blank, returning reference_datetime")
        return reference_datetime
    relative_time_pattern = r'^([+-]\d+)([smhdw])$'
    match = re.match(relative_time_pattern, user_input)
    if match:
        print("[DEBUG] parse_date_time_input: relative time pattern matched")
        amount = int(match.group(1))
        unit = match.group(2)
        delta_kwargs = {'s': 'seconds', 'm': 'minutes', 'h': 'hours', 'd': 'days', 'w': 'weeks'}
        result = reference_datetime + timedelta(**{delta_kwargs[unit]: amount})
        print(f"[DEBUG] parse_date_time_input: returning {result}")
        return result
    for fmt in ['%Y%m%d-%H%M', '%Y%m%d']:
        try:
            result = datetime.strptime(user_input, fmt)
            print(f"[DEBUG] parse_date_time_input: parsed with {fmt}, returning {result}")
            return result
        except ValueError:
            print(f"[DEBUG] parse_date_time_input: failed {fmt}")
            continue
    print("[DEBUG] parse_date_time_input: using dateutil.parser")
    result = dateutil.parser.parse(user_input, fuzzy=True)
    print(f"[DEBUG] parse_date_time_input: returning {result}")
    return result

def input_with_default(prompt: str, default: str) -> str:
    print(f"[DEBUG] input_with_default: prompt='{prompt}', default='{default}'")
    val = input(prompt).strip()
    if val == "":
        print(f"No input given. Defaulting to '{default}'")
        return default
    print(f"User input: '{val}'")
    return val

def input_yes_no(prompt: str, default: str = 'y') -> str:
    print(f"[DEBUG] input_yes_no: prompt='{prompt}', default='{default}'")
    val = input(prompt).strip().lower()
    if val not in ['y','n','yes','no','']:
        print(f"Invalid input '{val}', defaulting to '{default}'")
        return default
    if val == '':
        print(f"No input given, defaulting to '{default}'")
        return default
    choice = 'y' if val.startswith('y') else 'n'
    print(f"User input: '{choice}'")
    return choice

def input_yes_no_no_default(prompt: str) -> str:
    """
    Prompt the user with a yes/no question.
    There is NO default response.
    If blank or invalid, keep asking until 'y' or 'n' is provided.
    """
    while True:
        val = input(prompt).strip().lower()
        if val in ['y','yes']:
            print("User input: 'y'")
            return 'y'
        elif val in ['n','no']:
            print("User input: 'n'")
            return 'n'
        else:
            print("Invalid input, please enter 'y' or 'n'.")

def preview_database(db_path: str):
    print("[DEBUG] preview_database: start")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        print("[DEBUG] preview_database: previewing symbols table")
        symbols_df = pd.read_sql_query("SELECT * FROM symbols", conn)
        print("Symbols table preview:")
        print(symbols_df.head())

        print("[DEBUG] preview_database: previewing timeframes table")
        timeframes_df = pd.read_sql_query("SELECT * FROM timeframes", conn)
        print("Timeframes table preview:")
        print(timeframes_df.head())

        print("[DEBUG] preview_database: previewing klines table")
        klines_df = pd.read_sql_query("SELECT * FROM klines LIMIT 5", conn)
        print("Klines table sample:")
        print(klines_df.head())

        conn.close()

        if symbols_df.empty or timeframes_df.empty or klines_df.empty:
            print("[DEBUG] preview_database: database seems empty or incomplete")
            return False
        print("[DEBUG] preview_database: database content looks valid")
        return True
    except Exception as e:
        print(f"Error previewing database: {e}")
        return False

def recreate_database(db_path: str):
    print("[DEBUG] recreate_database: deleting old DB and creating new empty DB")
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"[DEBUG] recreate_database: deleted existing database at {db_path}")
    from sqlite_data_manager import create_connection, create_tables
    conn = create_connection(db_path)
    if conn:
        create_tables(conn)
        conn.close()
        print("[DEBUG] recreate_database: new DB created with tables")
    else:
        print("[DEBUG] recreate_database: failed to create new database")
        sys.exit(1)

def main() -> None:
    print("Script start: Clearing screen...")
    clear_screen()

    print("Running backup cleanup...")
    run_backup_cleanup()
    print("Backup cleanup done.")

    new_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    reports_dir = 'reports'
    csv_dir = 'csv'
    print(f"Ensuring output directories exist: {reports_dir}, {csv_dir}...")
    for directory in [reports_dir, csv_dir]:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Directory ensured: {directory}")
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")
            sys.exit(1)
    print("Output directories are ready.")

    print("Do you want to delete all previously generated output? (y/n): ")
    delete_output = input_yes_no("Do you want to delete all previously generated output? (y/n): ", default='y')
    if delete_output == 'y':
        print("User chose to delete previously generated output. Cleaning...")
        for folder in ['indicator_charts', 'heatmaps', 'combined_charts', reports_dir]:
            folder_path = Path(folder).resolve()
            if folder_path.exists():
                print(f"Clearing folder: {folder_path}")
                for filename in os.listdir(folder_path):
                    file_path = folder_path / filename
                    try:
                        if file_path.is_file() or file_path.is_symlink():
                            file_path.unlink()
                            print(f"Deleted file: {file_path}")
                        elif file_path.is_dir():
                            shutil.rmtree(file_path)
                            print(f"Deleted directory: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
        print("Output folders cleaned.")
    else:
        print("User chose not to delete previously generated output.")

    generate_charts = (input_yes_no("Do you want to generate individual indicator charts? (y/n): ", default='y') == 'y')
    print(f"Generate individual indicator charts: {'Yes' if generate_charts else 'No'}")

    generate_heatmaps_flag = (input_yes_no("Do you want to generate heatmaps? (y/n): ", default='y') == 'y')
    print(f"Generate heatmaps: {'Yes' if generate_heatmaps_flag else 'No'}")

    save_correlation_csv = (input_yes_no("Do you want to save a single CSV containing each indicator's correlation values for each lag point? (y/n): ", default='y') == 'y')
    print(f"Save correlation CSV: {'Yes' if save_correlation_csv else 'No'}")

    symbol = input_with_default("Enter the trading symbol (e.g., 'BTCUSDT'): ", "BTCUSDT").upper()
    timeframe = input_with_default("Enter the timeframe (e.g., '1d'): ", "1d")
    print(f"Using symbol={symbol}, timeframe={timeframe}")

    print("Loading data...")
    start_time = time.time()
    data, is_reverse_chronological, db_filename = load_data(symbol, timeframe)
    load_duration = time.time() - start_time
    print(f"Data load completed in {load_duration:.2f} seconds. Data empty: {data.empty}")

    print("Previewing database contents...")
    db_ok = preview_database(DB_PATH)
    if not db_ok:
        print("Database content seems invalid or incomplete.")
        while True:
            choice = input_yes_no_no_default("Database content seems invalid. Erase and create a new one? (y/n): ")
            if choice == 'y':
                recreate_database(DB_PATH)
                print("New database created. Attempting to load data again...")
                data, is_reverse_chronological, db_filename = load_data(symbol, timeframe)
                if data.empty:
                    print("Data still not found after recreating the database. Attempting to download data...")
                    download_choice = input_yes_no("Do you want to download Binance historical data now? (y/n): ", default='y')
                    if download_choice == 'y':
                        try:
                            download_binance_data(symbol, timeframe)
                            data, is_reverse_chronological, db_filename = load_data(symbol, timeframe)
                            if data.empty:
                                print("Data still not found after download. Exiting.")
                                sys.exit(1)
                            else:
                                print("Data successfully loaded from database after download.")
                                break
                        except Exception as e:
                            print(f"Error occurred during data download: {e}")
                            sys.exit(1)
                    else:
                        print("Exiting since no data and no download requested.")
                        sys.exit(1)
                else:
                    print("Data successfully loaded from the new database.")
                    break
            elif choice == 'n':
                print("User chose not to recreate the database. Exiting.")
                sys.exit(1)

    if data.empty:
        print("No data found in the database.")
        download_choice = input_yes_no("Do you want to download Binance historical data now? (y/n): ", default='y')
        if download_choice == 'y':
            print("Downloading data from Binance...")
            try:
                start_time = time.time()
                download_binance_data(symbol, timeframe)
                download_duration = time.time() - start_time
                print(f"Data download completed in {download_duration:.2f} seconds. Reloading data...")
                data, is_reverse_chronological, db_filename = load_data(symbol, timeframe)
                if data.empty:
                    print("Data still not found after download. Exiting.")
                    sys.exit(1)
                else:
                    print("Data successfully loaded from database after download.")
            except Exception as e:
                print(f"Error occurred during data download: {e}")
                sys.exit(1)
        else:
            print("Exiting since no data and no download requested.")
            sys.exit(0)
    else:
        print("Data found in database. Proceeding with analysis...")

    print("Computing indicators...")
    start_time = time.time()
    data = compute_all_indicators(data)
    indicator_duration = time.time() - start_time
    print(f"Indicators computed in {indicator_duration:.2f} seconds.")

    if 'Date' not in data.columns:
        if 'open_time' in data.columns:
            data['Date'] = pd.to_datetime(data['open_time'], errors='coerce').dt.tz_localize(None)
            print("Created 'Date' column from 'open_time'.")
        else:
            print("No 'Date' or 'open_time' column found. Exiting.")
            sys.exit(1)

    if 'Close' not in data.columns:
        if 'close' in data.columns:
            data.rename(columns={'close': 'Close'}, inplace=True)
            print("Renamed 'close' to 'Close'.")
        else:
            print("No 'Close' or 'close' column found. Exiting.")
            sys.exit(1)

    print("Determining time interval between data points...")
    try:
        time_interval = determine_time_interval(data)
        print(f"Time interval determined: {time_interval}")
    except Exception as e:
        print(f"Failed to determine time interval: {e}")
        sys.exit(1)

    print("Preparing data (scaling features, etc.)...")
    start_time = time.time()
    try:
        X_scaled, feature_names = prepare_data(data)
        prep_duration = time.time() - start_time
        print(f"Data preparation done in {prep_duration:.2f} seconds. Features count: {len(feature_names)}")
    except Exception as e:
        print(f"Data preparation failed: {e}")
        sys.exit(1)

    print("Identifying original indicators...")
    original_indicators = get_original_indicators(feature_names, data)
    expected_indicators = ['FI', 'ichimoku', 'KCU_20_2.0', 'STOCHRSI_14_5_3_slowk', 'VI+_14']
    original_indicators = handle_missing_indicators(original_indicators, data, expected_indicators)
    print(f"Original indicators count: {len(original_indicators)}")

    if not original_indicators:
        print("No valid indicators found. Exiting.")
        sys.exit(1)

    max_lag = len(data) - 51
    print(f"Max lag calculated as {max_lag}")
    if max_lag < 1:
        print("Not enough data to proceed (max_lag < 1). Exiting.")
        sys.exit(1)

    print("Loading or calculating correlations. This may take a while...")
    start_time = time.time()
    load_or_calculate_correlations(
        data=data,
        original_indicators=original_indicators,
        max_lag=max_lag,
        is_reverse_chronological=is_reverse_chronological,
        symbol=symbol,
        timeframe=timeframe
    )
    corr_duration = time.time() - start_time
    print(f"Correlations loaded/calculated in {corr_duration:.2f} seconds.")

    print("Retrieving correlations from database...")
    correlation_db = CorrelationDatabase(DB_PATH)
    correlations = {}
    for i, indicator in enumerate(original_indicators, start=1):
        print(f"Processing indicator {i}/{len(original_indicators)}: {indicator}")
        correlations[indicator] = []
        for lag in range(1, max_lag + 1):
            val = correlation_db.get_correlation(symbol, timeframe, indicator, lag)
            correlations[indicator].append(val)
        print(f"Completed retrieving correlations for indicator: {indicator}")
    correlation_db.close()
    print("Finished retrieving all correlations.")

    print("Generating statistical summary of correlations...")
    try:
        summary_df = generate_statistical_summary(correlations, max_lag)
        summary_csv = os.path.join(csv_dir, f"{new_timestamp}_{symbol}_{timeframe}_statistical_summary.csv")
        summary_df.to_csv(summary_csv, index=True)
        print(f"Statistical summary saved to: {summary_csv}")
    except Exception as e:
        print(f"Error generating statistical summary: {e}")

    print("Generating combined correlation chart...")
    try:
        generate_combined_correlation_chart(correlations, max_lag, time_interval, new_timestamp, f"{symbol}_{timeframe}")
        print("Combined correlation chart generated.")
    except Exception as e:
        print(f"Error generating combined correlation chart: {e}")

    if generate_charts:
        print("Generating individual indicator charts...")
        try:
            visualize_data(
                data=data,
                features=X_scaled,
                feature_columns=feature_names,
                timestamp=new_timestamp,
                is_reverse_chronological=is_reverse_chronological,
                time_interval=time_interval,
                generate_charts=generate_charts,
                cache=correlations,
                calculate_correlation_func=calculate_correlation,
                base_csv_filename=f"{symbol}_{timeframe}"
            )
            print("Individual indicator charts generated.")
        except Exception as e:
            print(f"Error generating individual charts: {e}")
    else:
        print("Skipping individual indicator charts as per user choice.")

    if generate_heatmaps_flag:
        print("Generating heatmaps...")
        try:
            generate_heatmaps(
                data=data,
                timestamp=new_timestamp,
                time_interval=time_interval,
                generate_heatmaps_flag=generate_heatmaps_flag,
                cache=correlations,
                calculate_correlation=calculate_correlation,
                base_csv_filename=f"{symbol}_{timeframe}"
            )
            print("Heatmaps generated.")
        except Exception as e:
            print(f"Error generating heatmaps: {e}")
    else:
        print("Skipping heatmaps as per user choice.")

    print("Generating best indicators table...")
    try:
        best_indicators_df = generate_best_indicator_table(correlations, max_lag)
        best_indicators_csv = os.path.join(csv_dir, f"{new_timestamp}_{symbol}_{timeframe}_best_indicators.csv")
        best_indicators_df.to_csv(best_indicators_csv, index=False)
        print(f"Best indicators table saved to: {best_indicators_csv}")
    except Exception as e:
        print(f"Error generating best indicators table: {e}")

    if save_correlation_csv:
        print("Generating correlation CSV for all indicators and lags...")
        try:
            generate_correlation_csv(correlations, max_lag, f"{symbol}_{timeframe}", csv_dir)
            print("Correlation CSV generated.")
        except Exception as e:
            print(f"Error generating correlation CSV: {e}")
    else:
        print("Skipping correlation CSV generation as per user choice.")

    print("Preparing future prediction steps...")
    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
    latest_date_in_data = data['Date'].max()
    current_datetime = datetime.now()
    time_interval_seconds_map = {'second': 1, 'minute': 60, 'hour': 3600, 'day': 86400, 'week': 604800}
    if time_interval not in time_interval_seconds_map:
        print("Unsupported time interval. Exiting.")
        sys.exit(1)

    time_diff_seconds = (current_datetime - latest_date_in_data).total_seconds()
    lag_periods_behind_current = int(time_diff_seconds / time_interval_seconds_map[time_interval])
    print(f"Latest data point is {lag_periods_behind_current} {time_interval}(s) behind current time.")

    future_input = input_with_default("Future date/time: ", "")
    print(f"User input for future date/time: '{future_input}'")
    try:
        future_datetime = parse_date_time_input(future_input, current_datetime) if future_input else current_datetime
        print(f"Using future datetime: {future_datetime}")
    except Exception as e:
        print(f"Error parsing future datetime: {e}")
        sys.exit(1)

    lag_seconds = (future_datetime - latest_date_in_data).total_seconds()
    print(f"Lag seconds: {lag_seconds}")
    if lag_seconds <= 0:
        print("Future date/time is not after the latest data point. Exiting.")
        sys.exit(1)
    lag_periods = int(lag_seconds / time_interval_seconds_map[time_interval])
    print(f"Calculated lag_periods for prediction: {lag_periods}")
    if lag_periods < 1:
        print("lag_periods < 1, cannot predict. Exiting.")
        sys.exit(1)
    if lag_periods > max_lag:
        print(f"Requested future lag ({lag_periods}) exceeds max_lag ({max_lag}). Exiting.")
        sys.exit(1)

    print("Performing linear regression prediction...")
    try:
        perform_linear_regression(data, correlations, max_lag, time_interval, new_timestamp, f"{symbol}_{timeframe}", future_datetime, lag_periods)
        print("Linear regression prediction done.")
    except Exception as e:
        print(f"Error performing linear regression prediction: {e}")

    print("Performing advanced analysis prediction...")
    try:
        advanced_price_prediction(data, correlations, max_lag, time_interval, new_timestamp, f"{symbol}_{timeframe}", future_datetime, lag_periods)
        print("Advanced analysis prediction done.")
    except Exception as e:
        print(f"Error performing advanced price prediction: {e}")

    print("Script completed successfully.")

if __name__ == "__main__":
    main()