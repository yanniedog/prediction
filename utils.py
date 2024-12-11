# Filename: utils.py

import os
import sys
import json
import logging
import shutil
import subprocess
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter issues
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Parallel, delayed
from scipy.stats import t
from sklearn.preprocessing import StandardScaler

# Ensure dateutil is installed; if not, provide an informative error
try:
    import dateutil.parser
except ImportError:
    print("Required module 'python-dateutil' not found. Please install it using 'pip install python-dateutil'.")
    sys.exit(1)

# Import custom modules
from linear_regression import perform_linear_regression
from advanced_analysis import advanced_price_prediction
from load_data import load_data
from indicators import compute_all_indicators

# Import other utilities
from logging_setup import configure_logging
from data_utils import (
    clear_screen,
    prepare_data,
    determine_time_interval,
    get_original_indicators,
    handle_missing_indicators
)
from correlation_utils import (
    load_or_calculate_correlations, 
    calculate_correlation, 
    calculate_and_save_indicator_correlations
)
from visualization_utils import generate_combined_correlation_chart, visualize_data, generate_heatmaps
from backup_utils import run_backup_cleanup
from table_generation import (
    generate_best_indicator_table,
    generate_statistical_summary,
    generate_correlation_csv
)
from binance_historical_data_downloader import download_binance_data, fetch_klines, process_klines, save_to_sqlite

# Suppress warnings for clean output
import warnings
warnings.filterwarnings('ignore')


def configure_logging():
    """Configures the logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("application.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def run_backup_cleanup():
    """Runs backup and cleanup operations."""
    logging.info("Running backup and cleanup operations.")
    # Implement backup and cleanup logic here
    # For example, deleting old log files, compressing backups, etc.
    pass


def list_database_files(database_dir: str) -> List[str]:
    """Lists all SQLite database files in the specified directory."""
    return [f for f in os.listdir(database_dir) if f.endswith('.db')]


def select_existing_database(database_dir: str) -> Optional[str]:
    """Prompts the user to select an existing database for updating or previewing."""
    db_files = list_database_files(database_dir)
    if not db_files:
        print("No existing databases found.")
        logging.info("No existing databases found.")
        return None

    print("\nExisting Databases:")
    for idx, db in enumerate(db_files, 1):
        print(f"{idx}. {db}")

    while True:
        selected = input(f"Enter the number of the database to select (1-{len(db_files)}) or type 'x' to go back: ").strip()
        if selected.lower() == 'x':
            return None  # Return to main menu
        if selected.isdigit() and 1 <= int(selected) <= len(db_files):
            selected_db = db_files[int(selected) - 1]
            print(f"Selected Database: {selected_db}")
            logging.info(f"Selected existing database: {selected_db}")
            return os.path.join(database_dir, selected_db)
        else:
            print("Invalid selection. Please try again.")


def preview_database(db_path: str) -> None:
    """Displays a preview of the latest entries in the specified database."""
    try:
        data, is_reverse_chronological, _ = load_data(db_path)
        if data.empty:
            print("The selected database is empty.")
            logging.info(f"Preview: Database '{db_path}' is empty.")
        else:
            print(f"\nPreview of the latest data in '{os.path.basename(db_path)}':")
            print(data.tail())
            logging.info(f"Previewed data from '{db_path}'.")
    except Exception as e:
        logging.error(f"Failed to preview database '{db_path}': {e}")
        print(f"Failed to preview database '{db_path}': {e}")


def update_database(db_path: str) -> None:
    """Updates the specified database by fetching the latest data from Binance."""
    try:
        base_filename = os.path.basename(db_path)
        symbol, interval = os.path.splitext(base_filename)[0].split('_')
    except ValueError:
        logging.error(f"Database filename '{db_path}' does not follow the 'symbol_interval.db' format.")
        print(f"Database filename '{db_path}' does not follow the 'symbol_interval.db' format.")
        return

    print(f"Updating database for {symbol} with interval {interval}...")
    logging.info(f"Updating database '{db_path}' for symbol '{symbol}' and interval '{interval}'.")

    # Prompt user for start and end dates
    print("Please enter the date range for the additional data.")
    start_date_input = input("Enter the start date (YYYY-MM-DD) or press Enter to use the latest date in the database: ").strip()
    end_date_input = input("Enter the end date (YYYY-MM-DD) or press Enter to use today's date: ").strip()

    # Load existing data to determine the latest date if needed
    try:
        data, is_reverse_chronological, _ = load_data(db_path)
        if data.empty:
            print("The selected database is empty. Downloading full dataset.")
            logging.warning(f"Database '{db_path}' is empty. Initiating full download.")
            download_binance_data(symbol, interval, db_path)
            return
    except Exception as e:
        logging.error(f"Failed to load data from '{db_path}': {e}")
        print(f"Failed to load data from '{db_path}': {e}")
        return

    # Determine start_time
    if start_date_input:
        try:
            start_datetime = datetime.strptime(start_date_input, '%Y-%m-%d')
        except ValueError:
            print("Invalid start date format. Please use YYYY-MM-DD.")
            logging.error("Invalid start date format entered.")
            return
    else:
        latest_timestamp = data['Date'].max()
        start_datetime = latest_timestamp + timedelta(seconds=1)  # Start from the next second
        print(f"No start date entered. Using the next timestamp after the latest data point: {start_datetime}")

    # Determine end_time
    if end_date_input:
        try:
            end_datetime = datetime.strptime(end_date_input, '%Y-%m-%d')
        except ValueError:
            print("Invalid end date format. Please use YYYY-MM-DD.")
            logging.error("Invalid end date format entered.")
            return
    else:
        end_datetime = datetime.now()
        print(f"No end date entered. Using current date and time: {end_datetime}")

    # Convert to milliseconds for Binance API
    start_time = int(start_datetime.timestamp() * 1000)
    end_time = int(end_datetime.timestamp() * 1000)

    if start_time >= end_time:
        print("Start date must be before end date.")
        logging.error("Start date is not before end date.")
        return

    try:
        # Fetch new klines data
        klines = fetch_klines(symbol, interval, start_time, end_time)
        if not klines:
            print("No new data available to update for the specified date range.")
            logging.info(f"No new data fetched for '{db_path}' within the specified date range.")
            return

        # Process klines data
        df = process_klines(klines)

        # Save to SQLite
        save_to_sqlite(df, db_path)

        print(f"Successfully updated the database '{os.path.basename(db_path)}' with {len(df)} new records.")
        logging.info(f"Updated database '{db_path}' with {len(df)} new records.")

    except Exception as e:
        logging.error(f"Failed to update database '{db_path}': {e}")
        print(f"Failed to update database '{db_path}': {e}")


def download_new_dataset(database_dir: str, default_interval: str = '1d') -> Optional[str]:
    """Facilitates downloading a new dataset from Binance with a default interval."""
    return create_new_database(database_dir, default_interval=default_interval)


def create_new_database(database_dir: str, default_interval: str = '1d') -> Optional[str]:
    """Creates a new database by downloading data from Binance with a default interval."""
    symbol = input("Enter the trading symbol (e.g., BTCUSDT): ").strip().upper()
    if not symbol:
        print("Symbol cannot be empty.")
        logging.error("Symbol input was empty.")
        return None  # Return to main menu
    interval = default_interval  # Use default interval
    db_filename = f"{symbol}_{interval}.db"
    db_path = os.path.join(database_dir, db_filename)
    if os.path.exists(db_path):
        print(f"Database '{db_filename}' already exists.")
        logging.warning(f"Attempted to create a database that already exists: {db_filename}")
        return db_path
    else:
        # Create the new database by downloading data
        download_binance_data(symbol, interval, db_path)
        return db_path


def perform_analysis(db_path: str, reports_dir: str, cache_dir: str, timestamp: str) -> None:
    """Performs data analysis and prediction on the selected database."""
    try:
        data, is_reverse_chronological, db_filename = load_data(db_path)
        logging.info("Data loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        print(f"Failed to load data: {e}")
        return

    # Check if data is empty
    if data.empty:
        logging.warning("Database is empty. Prompting user to download Binance data.")
        print("No data found in the database.")

        download_choice = input("Do you want to download Binance historical data now? (y/n): ").strip().lower()
        if download_choice == 'y':
            print("Starting Binance data download...")
            logging.info("User opted to download Binance data.")
            # Assuming symbol and interval are part of db_filename
            try:
                symbol, interval = os.path.splitext(os.path.basename(db_path))[0].split('_')
            except ValueError:
                logging.error(f"Database filename '{db_path}' does not follow the 'symbol_interval.db' format.")
                print(f"Database filename '{db_path}' does not follow the 'symbol_interval.db' format.")
                return
            download_binance_data(symbol, interval, db_path)

            # Reload data after download
            try:
                data, is_reverse_chronological, db_filename = load_data(db_path)
                if data.empty:
                    logging.error("Database is still empty after downloading data. Exiting.")
                    print("Database is still empty after downloading data. Please check the downloader script.")
                    return
                else:
                    logging.info("Data loaded successfully after downloading.")
                    print("Data loaded successfully after downloading.")
            except Exception as e:
                logging.error(f"Failed to load data after downloading: {e}")
                print(f"Failed to load data after downloading: {e}")
                return
        else:
            logging.info("User declined to download Binance data. Exiting analysis.")
            print("No data available to proceed with analysis.")
            return

    # Proceed with the rest of the script
    try:
        # Compute Indicators
        data = compute_all_indicators(data)
        logging.info("Indicators computed successfully.")
    except Exception as e:
        logging.error(f"Failed to compute indicators: {e}")
        print(f"Failed to compute indicators: {e}")
        return

    print("Data loaded and indicators computed.")

    # Determine Time Interval
    try:
        time_interval = determine_time_interval(data)
        logging.info(f"Determined time interval: {time_interval}")
    except Exception as e:
        logging.error(f"Failed to determine time interval: {e}")
        print(f"Failed to determine time interval: {e}")
        return

    print(f"Determined time interval: {time_interval}")

    # Prepare Data
    print("Preparing data for analysis...")
    try:
        X_scaled, feature_names = prepare_data(data)
        print("Data prepared.")
        logging.info(f"Feature columns: {feature_names}")
    except Exception as e:
        logging.error(f"Failed to prepare data: {e}")
        print(f"Failed to prepare data: {e}")
        return

    # Identify Original Indicators
    original_indicators = get_original_indicators(feature_names, data)

    # Handle Missing Indicators
    expected_indicators = ['FI', 'ichimoku', 'KCU_20_2.0', 'STOCHRSI_14_5_3_slowk', 'VI+_14']
    original_indicators = handle_missing_indicators(original_indicators, data, expected_indicators)

    if not original_indicators:
        logging.error("No valid indicators found for correlation calculation after excluding missing indicators.")
        print("No valid indicators found for correlation calculation after excluding missing indicators.")
        return

    # Determine Base Filenames
    base_csv_filename = os.path.splitext(os.path.basename(db_filename))[0]
    cache_filename = os.path.join(cache_dir, f"{base_csv_filename}.json")

    # Define max_lag
    max_lag = len(data) - 51  # Adjust as needed
    if max_lag < 1:
        logging.error("Insufficient data length to compute correlations with the specified max_lag.")
        print("Insufficient data length to compute correlations with the specified max_lag.")
        return

    # Load or Calculate Correlations
    try:
        correlations = load_or_calculate_correlations(
            data=data,
            original_indicators=original_indicators,
            max_lag=max_lag,
            is_reverse_chronological=is_reverse_chronological,
            cache_filename=cache_filename,
            db_path=db_path  # Pass the database path for storing correlations
        )
    except ValueError as ve:
        logging.error(str(ve))
        print(str(ve))
        return
    except Exception as e:
        logging.error(f"Failed to load or calculate correlations: {e}")
        print(f"Failed to load or calculate correlations: {e}")
        return

    # Generate Summary Table of Correlations
    try:
        summary_df = generate_statistical_summary(correlations, max_lag)
        summary_csv = os.path.join(reports_dir, f"{timestamp}_{base_csv_filename}_statistical_summary.csv")
        summary_df.to_csv(summary_csv, index=True)
        print(f"Generated statistical summary: {summary_csv}")
        logging.info(f"Statistical summary saved as '{summary_csv}'.")
    except Exception as e:
        logging.error(f"Failed to generate statistical summary: {e}")
        print(f"Failed to generate statistical summary: {e}")

    # Generate Combined Correlation Chart
    try:
        generate_combined_correlation_chart(
            correlations=correlations,
            max_lag=max_lag,
            time_interval=time_interval,
            timestamp=timestamp,
            base_csv_filename=base_csv_filename
        )
    except Exception as e:
        logging.error(f"Failed to generate combined correlation chart: {e}")
        print(f"Failed to generate combined correlation chart: {e}")

    # Visualize Data (Generate Individual Indicator Charts)
    generate_charts = input("Do you want to generate individual indicator charts? (y/n): ").strip().lower() == 'y'
    generate_heatmaps_flag = input("Do you want to generate heatmaps? (y/n): ").strip().lower() == 'y'
    save_correlation_csv = input("Do you want to save a single CSV containing each indicator's correlation values for each lag point? (y/n): ").strip().lower() == 'y'

    if generate_charts:
        try:
            visualize_data(
                data=data,
                features=X_scaled,
                feature_columns=feature_names,
                timestamp=timestamp,
                is_reverse_chronological=is_reverse_chronological,
                time_interval=time_interval,
                generate_charts=generate_charts,
                cache=correlations,
                calculate_correlation=calculate_correlation,
                base_csv_filename=base_csv_filename
                # db_path removed
            )
            logging.info("Visualization completed.")
        except Exception as e:
            logging.error(f"Failed to visualize data: {e}")
            print(f"Failed to visualize data: {e}")

    # Generate Heatmaps
    if generate_heatmaps_flag:
        try:
            generate_heatmaps(
                data=data,
                timestamp=timestamp,
                time_interval=time_interval,
                generate_heatmaps=generate_heatmaps_flag,
                cache=correlations,
                calculate_correlation=calculate_correlation,
                base_csv_filename=base_csv_filename
                # db_path removed
            )
            logging.info("Heatmaps generated.")
        except Exception as e:
            logging.error(f"Failed to generate heatmaps: {e}")
            print(f"Failed to generate heatmaps: {e}")

    # Generate Best Indicator Table
    try:
        best_indicators_df = generate_best_indicator_table(correlations, max_lag)
        best_indicators_csv = os.path.join(reports_dir, f"{timestamp}_{base_csv_filename}_best_indicators.csv")
        best_indicators_df.to_csv(best_indicators_csv, index=False)
        print(f"Generated best indicator table: {best_indicators_csv}")
        logging.info(f"Best indicator table saved as '{best_indicators_csv}'.")
    except Exception as e:
        logging.error(f"Failed to generate best indicator table: {e}")
        print(f"Failed to generate best indicator table: {e}")

    # Generate Correlation CSV Table
    if save_correlation_csv:
        try:
            generate_correlation_csv(correlations, max_lag, base_csv_filename, reports_dir)
        except TypeError as te:
            logging.error(f"TypeError during correlation CSV generation: {te}")
            print(f"Error during correlation CSV generation: {te}")
        except Exception as e:
            logging.error(f"Failed to generate correlation table: {e}")
            print(f"Failed to generate correlation table: {e}")

    # Get the latest date/time in the data
    try:
        data['Date'] = pd.to_datetime(data['Date'])
        latest_date_in_data = data['Date'].max()
        print(f"\nLatest date/time in the CSV data is: {latest_date_in_data}")
        logging.info(f"Latest date/time in the CSV data: {latest_date_in_data}")
    except Exception as e:
        logging.error(f"Failed to determine the latest date/time in the data: {e}")
        print(f"Failed to determine the latest date/time in the data: {e}")
        return

    # Calculate how many lag periods behind the current date/time it is
    current_datetime = datetime.now()
    time_interval_seconds = {
        '1s': 1,
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        '4h': 14400,
        '1d': 86400,
        '1w': 604800
    }

    if time_interval not in time_interval_seconds:
        logging.error(f"Unsupported time interval '{time_interval}'.")
        print(f"Unsupported time interval '{time_interval}'.")
        return

    time_diff_seconds = (current_datetime - latest_date_in_data).total_seconds()
    lag_periods_behind_current = int(time_diff_seconds / time_interval_seconds[time_interval])

    print(f"The latest date/time in the data is {lag_periods_behind_current} {time_interval}(s) behind the current date/time.")
    logging.info(f"Latest data is {lag_periods_behind_current} {time_interval}(s) behind current time.")

    # Explain to the user that they can enter relative times like +1h, +4d, -9m
    print("\nYou can enter the future date/time you wish to predict the price for.")
    print("You can enter an absolute date/time in the format YYYYMMDD-HHMM (e.g., 20231123-1530),")
    print("or you can enter a relative time like '+1h' for 1 hour into the future, '+4d' for 4 days into the future,")
    print("'-9m' for 9 minutes in the past (relative to current time), or leave it blank to use the current date/time.")

    # Get the user's date/time input
    user_input = input("Enter the future date/time you wish to predict the price for: ").strip()

    # Parse the user's input
    try:
        future_datetime = parse_date_time_input(user_input, current_datetime) if user_input else current_datetime
        if user_input:
            print(f"Using future date/time: {future_datetime}")
        else:
            print(f"No input provided. Using current date/time: {future_datetime}")
    except ValueError as e:
        print(f"Error parsing date/time input: {e}")
        logging.error(f"Error parsing date/time input: {e}")
        return

    # Calculate the lag period between latest_date_in_data and future_datetime
    lag_seconds = (future_datetime - latest_date_in_data).total_seconds()
    if lag_seconds <= 0:
        print("The future date/time must be after the latest date/time in the dataset.")
        logging.error("Future date/time is before the latest date/time in the dataset.")
        return

    lag_periods = int(lag_seconds / time_interval_seconds[time_interval])

    if lag_periods < 1:
        print("The future date/time must be at least one lag period ahead of the latest data point.")
        logging.error("Calculated lag period is less than 1.")
        return

    if lag_periods > max_lag:
        print(f"The lag period must be between 1 and {max_lag}.")
        logging.error(f"Lag period {lag_periods} exceeds max_lag {max_lag}.")
        return

    print(f"Calculated lag period: {lag_periods} {time_interval}(s)")
    logging.info(f"Calculated lag period: {lag_periods} {time_interval}(s)")

    # Perform linear regression for price prediction
    try:
        perform_linear_regression(
            data=data,
            correlations=correlations,
            max_lag=max_lag,
            time_interval=time_interval,
            timestamp=timestamp,
            base_csv_filename=base_csv_filename,
            future_datetime=future_datetime,
            lag_periods=lag_periods
        )
        logging.info("Linear regression prediction completed.")
    except TypeError as te:
        logging.error(f"TypeError during linear regression: {te}")
        print(f"Error during linear regression: {te}")
    except Exception as e:
        logging.error(f"Failed to perform linear regression: {e}")
        print(f"Error during linear regression: {e}")

    # Perform advanced analysis for price prediction
    try:
        advanced_price_prediction(
            data=data,
            correlations=correlations,
            max_lag=max_lag,
            time_interval=time_interval,
            timestamp=timestamp,
            base_csv_filename=base_csv_filename,
            future_datetime=future_datetime,
            lag_periods=lag_periods
        )
        logging.info("Advanced analysis and prediction completed.")
    except TypeError as te:
        logging.error(f"TypeError during advanced analysis: {te}")
        print(f"Error during advanced analysis: {te}")
    except Exception as e:
        logging.error(f"Failed to perform advanced analysis: {e}")
        print(f"Error during advanced analysis: {e}")

    # Calculate and save indicator-to-indicator correlations
    try:
        calculate_and_save_indicator_correlations(
            data=data,
            indicators=original_indicators,
            max_lag=max_lag,
            is_reverse_chronological=is_reverse_chronological,
            db_path=db_path
        )
        logging.info("Indicator-to-indicator correlations calculated and saved successfully.")
    except Exception as e:
        logging.error(f"Failed to calculate and save indicator-to-indicator correlations: {e}")
        print(f"Failed to calculate and save indicator-to-indicator correlations: {e}")

    logging.info("All processes completed successfully.")
    print("All processes completed successfully.")


def parse_date_time_input(user_input: str, reference_datetime: datetime) -> datetime:
    """
    Parses user input for date/time, which can be absolute or relative.

    Args:
        user_input (str): User input string.
        reference_datetime (datetime): Reference datetime for relative calculations.

    Returns:
        datetime: Parsed datetime.
    """
    user_input = user_input.strip()
    if not user_input:
        return reference_datetime

    # Check for relative time formats like '+1h', '+4d', '-9m'
    relative_time_pattern = r'^([+-]\d+)([smhdw])$'  # s: seconds, m: minutes, h: hours, d: days, w: weeks
    match = re.match(relative_time_pattern, user_input)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        delta_kwargs = {
            's': 'seconds',
            'm': 'minutes',
            'h': 'hours',
            'd': 'days',
            'w': 'weeks'
        }
        delta = timedelta(**{delta_kwargs[unit]: amount})
        return reference_datetime + delta

    # Try to parse as absolute datetime
    for fmt in ['%Y%m%d-%H%M', '%Y%m%d']:
        try:
            return datetime.strptime(user_input, fmt)
        except ValueError:
            continue

    # Use dateutil.parser to parse other formats
    try:
        return dateutil.parser.parse(user_input, fuzzy=True)
    except ValueError:
        raise ValueError(f"Could not parse date/time input: '{user_input}'")