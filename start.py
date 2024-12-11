# filename = start.py
# version = 11.0.0.9
# details = initial github version

import os
import sys
import logging
import shutil
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

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

# Import custom modules after logging is configured
from logging_setup import configure_logging
from linear_regression import perform_linear_regression
from advanced_analysis import advanced_price_prediction
from load_data import load_data
from indicators import compute_all_indicators
from data_utils import (
    clear_screen,
    prepare_data,
    determine_time_interval,
    get_original_indicators,
    handle_missing_indicators
)
from correlation_utils import load_or_calculate_correlations, calculate_correlation
from visualization_utils import generate_combined_correlation_chart, visualize_data, generate_heatmaps
from backup_utils import run_backup_cleanup
from table_generation import (
    generate_best_indicator_table,
    generate_statistical_summary,
    generate_correlation_csv
)

# Import the downloader function
try:
    from binance_historical_data_downloader import download_binance_data
except ImportError:
    logging.error("Failed to import 'download_binance_data' from 'binance_historical_data_downloader.py'.")
    print("Failed to import 'download_binance_data'. Please ensure 'binance_historical_data_downloader.py' is in the same directory.")
    sys.exit(1)

# Import CorrelationDatabase
from correlation_database import CorrelationDatabase

# Suppress warnings for clean output
import warnings
warnings.filterwarnings('ignore')

# Import the database path from config.py
from config import DB_PATH

def delete_generated_output(folders: List[str]) -> None:
    """
    Deletes all files within the specified folders.

    Args:
        folders (List[str]): List of folder paths whose contents need to be deleted.
    """
    for folder in folders:
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
                    logging.error(f"Failed to delete '{file_path}'. Reason: {e}")
            logging.info(f"Cleared contents of '{folder_path}'.")
        else:
            logging.info(f"Folder '{folder_path}' does not exist. Skipping deletion.")

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

def main() -> None:
    """Main function orchestrating the data analysis and prediction workflow."""
    # Initial Setup
    configure_logging()  # Configure logging first

    # Test logging
    logging.info("Logging has been successfully configured.")
    print("Logging has been successfully configured.")

    # Run backup_cleanup.py at the beginning
    run_backup_cleanup()

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    # Define directories
    reports_dir = 'reports'
    csv_dir = 'csv'  # New directory for CSV files

    for directory in [reports_dir, csv_dir]:
        try:
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Ensured directory '{directory}' exists.")
        except Exception as e:
            logging.error(f"Failed to create or access '{directory}' directory: {e}")
            print(f"Failed to create or access '{directory}' directory: {e}")
            sys.exit(1)

    # User Inputs for Deletion and Generation Options
    delete_output = input("Do you want to delete all previously generated output? (y/n): ").strip().lower() == 'y'
    if delete_output:
        folders_to_delete = [
            'indicator_charts',
            'heatmaps',
            'combined_charts',
            reports_dir  # This will delete everything within 'reports'
        ]
        delete_generated_output(folders_to_delete)
    else:
        logging.info("Retaining existing generated output.")

    generate_charts = input("Do you want to generate individual indicator charts? (y/n): ").strip().lower() == 'y'
    generate_heatmaps_flag = input("Do you want to generate heatmaps? (y/n): ").strip().lower() == 'y'
    save_correlation_csv = input("Do you want to save a single CSV containing each indicator's correlation values for each lag point? (y/n): ").strip().lower() == 'y'

    logging.info(f"Launching script with timestamp {timestamp}.")

    # Prompt user for symbol and timeframe
    symbol = input("Enter the trading symbol (e.g., 'BTCUSDT'): ").strip().upper()
    timeframe = input("Enter the timeframe (e.g., '1d'): ").strip()

    # Load Data
    print("Loading data...")
    try:
        data, is_reverse_chronological, db_filename = load_data(symbol, timeframe)
        logging.info("Data loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        print(f"Failed to load data: {e}")
        sys.exit(1)

    # Check if data is empty
    if data.empty:
        logging.warning("Database is empty. Prompting user to download Binance data.")
        print("No data found in the database for the specified symbol and timeframe.")

        download_choice = input("Do you want to download Binance historical data now? (y/n): ").strip().lower()
        if download_choice == 'y':
            print("Starting Binance data download...")
            logging.info("User opted to download Binance data.")

            # Modify download_binance_data to accept symbol and timeframe
            download_binance_data(symbol, timeframe)

            # Reload data after download
            try:
                data, is_reverse_chronological, db_filename = load_data(symbol, timeframe)
                if data.empty:
                    logging.error("Database is still empty after downloading data. Exiting.")
                    print("Database is still empty after downloading data. Please check the downloader script.")
                    sys.exit(1)
                else:
                    logging.info("Data loaded successfully after downloading.")
                    print("Data loaded successfully after downloading.")
            except Exception as e:
                logging.error(f"Failed to load data after downloading: {e}")
                print(f"Failed to load data after downloading: {e}")
                sys.exit(1)
        else:
            logging.info("User declined to download Binance data. Exiting.")
            print("No data available to proceed. Exiting.")
            sys.exit(0)

    # Compute Indicators
    try:
        data = compute_all_indicators(data)
        logging.info("Indicators computed successfully.")
    except Exception as e:
        logging.error(f"Failed to compute indicators: {e}")
        print(f"Failed to compute indicators: {e}")
        sys.exit(1)

    print("Data loaded and indicators computed.")

    # Ensure 'Date' column exists
    if 'Date' not in data.columns:
        if 'open_time' in data.columns:
            data['Date'] = pd.to_datetime(data['open_time'], errors='coerce')
            data['Date'] = data['Date'].dt.tz_localize(None)
            logging.info("'Date' column created from 'open_time'.")
        else:
            logging.error("Neither 'Date' nor 'open_time' column found in data.")
            print("Error: Neither 'Date' nor 'open_time' column found in data.")
            sys.exit(1)

    # Ensure 'Close' column exists
    if 'Close' not in data.columns:
        # Attempt to rename 'close' to 'Close' if present
        if 'close' in data.columns:
            data.rename(columns={'close': 'Close'}, inplace=True)
            logging.info("'close' column renamed to 'Close'.")
        else:
            logging.error("'Close' column is missing from the data.")
            print("Error: 'Close' column is missing from the data.")
            sys.exit(1)

    # Determine Time Interval
    try:
        time_interval = determine_time_interval(data)
        logging.info(f"Determined time interval: {time_interval}")
    except Exception as e:
        logging.error(f"Failed to determine time interval: {e}")
        print(f"Failed to determine time interval: {e}")
        sys.exit(1)

    print(f"Determined time interval: {time_interval}")

    # Prepare Data
    print("Preparing data for analysis...")
    try:
        X_scaled, feature_names = prepare_data(data)
    except Exception as e:
        logging.error(f"Failed during data preparation: {e}")
        print(f"Failed during data preparation: {e}")
        sys.exit(1)
    print("Data prepared.")
    logging.info(f"Feature columns: {feature_names}")

    # Identify Original Indicators
    original_indicators = get_original_indicators(feature_names, data)

    # Handle Missing Indicators
    expected_indicators = ['FI', 'ichimoku', 'KCU_20_2.0', 'STOCHRSI_14_5_3_slowk', 'VI+_14']
    original_indicators = handle_missing_indicators(original_indicators, data, expected_indicators)

    if not original_indicators:
        logging.error("No valid indicators found for correlation calculation after excluding missing indicators.")
        print("No valid indicators found for correlation calculation after excluding missing indicators.")
        sys.exit(1)

    # Define max_lag
    max_lag = len(data) - 51  # Adjust as needed
    if max_lag < 1:
        logging.error("Insufficient data length to compute correlations with the specified max_lag.")
        print("Insufficient data length to compute correlations with the specified max_lag.")
        sys.exit(1)

    # Load or Calculate Correlations
    try:
        load_or_calculate_correlations(
            data=data,
            original_indicators=original_indicators,
            max_lag=max_lag,
            is_reverse_chronological=is_reverse_chronological,
            symbol=symbol,
            timeframe=timeframe
        )
    except ValueError as ve:
        logging.error(str(ve))
        print(str(ve))
        sys.exit(1)

    # Fetch correlations from the database
    correlation_db = CorrelationDatabase(DB_PATH)
    correlations = {}
    for indicator in original_indicators:
        correlations[indicator] = []
        for lag in range(1, max_lag + 1):
            corr_value = correlation_db.get_correlation(symbol, timeframe, indicator, lag)
            correlations[indicator].append(corr_value)
    correlation_db.close()

    # Generate Summary Table of Correlations
    try:
        summary_df = generate_statistical_summary(correlations, max_lag)
        summary_csv = os.path.join(csv_dir, f"{timestamp}_{symbol}_{timeframe}_statistical_summary.csv")  # Save to CSV directory
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
            base_filename=f"{symbol}_{timeframe}"
        )
    except Exception as e:
        logging.error(f"Failed to generate combined correlation chart: {e}")
        print(f"Failed to generate combined correlation chart: {e}")

    # Visualize Data (Generate Individual Indicator Charts)
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
                correlations=correlations,
                calculate_correlation_func=calculate_correlation,  # Pass the function directly
                base_filename=f"{symbol}_{timeframe}"
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
                generate_heatmaps_flag=generate_heatmaps_flag,
                correlations=correlations,
                calculate_correlation=calculate_correlation,  # Pass the function directly
                base_filename=f"{symbol}_{timeframe}"
            )
            logging.info("Heatmaps generated.")
        except Exception as e:
            logging.error(f"Failed to generate heatmaps: {e}")
            print(f"Failed to generate heatmaps: {e}")

    # Generate Best Indicator Table
    try:
        best_indicators_df = generate_best_indicator_table(correlations, max_lag)
        best_indicators_csv = os.path.join(csv_dir, f"{timestamp}_{symbol}_{timeframe}_best_indicators.csv")  # Save to CSV directory
        best_indicators_df.to_csv(best_indicators_csv, index=False)
        print(f"Generated best indicator table: {best_indicators_csv}")
        logging.info(f"Best indicator table saved as '{best_indicators_csv}'.")
    except Exception as e:
        logging.error(f"Failed to generate best indicator table: {e}")
        print(f"Failed to generate best indicator table: {e}")

    # Generate Correlation CSV Table
    if save_correlation_csv:
        try:
            generate_correlation_csv(correlations, max_lag, f"{symbol}_{timeframe}", csv_dir)  # Save to CSV directory
        except Exception as e:
            logging.error(f"Failed to generate correlation table: {e}")
            print(f"Failed to generate correlation table: {e}")

    # Get the latest date/time in the data
    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
    latest_date_in_data = data['Date'].max()
    print(f"\nLatest date/time in the data is: {latest_date_in_data}")
    logging.info(f"Latest date/time in the data: {latest_date_in_data}")

    # Calculate how many lag periods behind the current date/time it is
    current_datetime = datetime.now()
    time_interval_seconds = {
        'second': 1,
        'minute': 60,
        'hour': 3600,
        'day': 86400,
        'week': 604800
    }

    if time_interval not in time_interval_seconds:
        logging.error(f"Unsupported time interval '{time_interval}'.")
        print(f"Unsupported time interval '{time_interval}'.")
        sys.exit(1)

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
        sys.exit(1)

    # Calculate the lag period between latest_date_in_data and future_datetime
    lag_seconds = (future_datetime - latest_date_in_data).total_seconds()
    if lag_seconds <= 0:
        print("The future date/time must be after the latest date/time in the dataset.")
        logging.error("Future date/time is before the latest date/time in the dataset.")
        sys.exit(1)

    lag_periods = int(lag_seconds / time_interval_seconds[time_interval])

    if lag_periods < 1:
        print("The future date/time must be at least one lag period ahead of the latest data point.")
        logging.error("Calculated lag period is less than 1.")
        sys.exit(1)

    if lag_periods > max_lag:
        print(f"The lag period must be between 1 and {max_lag}.")
        logging.error(f"Lag period {lag_periods} exceeds max_lag {max_lag}.")
        sys.exit(1)

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
            base_filename=f"{symbol}_{timeframe}",
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
            base_filename=f"{symbol}_{timeframe}",
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

    logging.info("All processes completed successfully.")

if __name__ == "__main__":
    main()