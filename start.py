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
from visualization_utils import generate_combined_correlation_chart, visualize_data, generate_heatmaps
from backup_utils import run_backup_cleanup
from table_generation import generate_best_indicator_table, generate_statistical_summary, generate_correlation_csv
from binance_historical_data_downloader import download_binance_data
from correlation_database import CorrelationDatabase
from config import DB_PATH

timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
working_dir_name = Path.cwd().name
log_filename = f"{working_dir_name}_{timestamp}.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

# Remove existing handlers if any
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add file handler
file_handler = logging.FileHandler(log_filename, mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Redirect stdout and stderr to logger
class LogCapture:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.buffer = ''

    def write(self, message):
        self.buffer += message
        parts = self.buffer.split('\n')
        for part in parts[:-1]:
            self.logger.log(self.level, part)
        self.buffer = parts[-1]

    def flush(self):
        if self.buffer:
            self.logger.log(self.level, self.buffer)
        self.buffer = ''

sys.stdout = LogCapture(logger, logging.INFO)
sys.stderr = LogCapture(logger, logging.ERROR)

def parse_date_time_input(user_input: str, reference_datetime: datetime) -> datetime:
    user_input = user_input.strip()
    if not user_input:
        return reference_datetime
    relative_time_pattern = r'^([+-]\d+)([smhdw])$'
    match = re.match(relative_time_pattern, user_input)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        delta_kwargs = {'s': 'seconds', 'm': 'minutes', 'h': 'hours', 'd': 'days', 'w': 'weeks'}
        return reference_datetime + timedelta(**{delta_kwargs[unit]: amount})
    for fmt in ['%Y%m%d-%H%M', '%Y%m%d']:
        try:
            return datetime.strptime(user_input, fmt)
        except ValueError:
            continue
    return dateutil.parser.parse(user_input, fuzzy=True)

def main() -> None:
    logger.info("Logging configured. Starting script.")
    clear_screen()
    run_backup_cleanup()

    new_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    reports_dir = 'reports'
    csv_dir = 'csv'
    for directory in [reports_dir, csv_dir]:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Ensured directory '{directory}' exists.")
        except Exception as e:
            logger.error(f"Failed to create or access '{directory}' directory: {e}")
            sys.exit(1)

    delete_output = input("Do you want to delete all previously generated output? (y/n): ").strip().lower() == 'y'
    if delete_output:
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
                        logger.error(f"Failed to delete '{file_path}'. Reason: {e}")
                logger.info(f"Cleared contents of '{folder_path}'.")
            else:
                logger.info(f"Folder '{folder_path}' does not exist. Skipping deletion.")
    else:
        logger.info("Retaining existing generated output.")

    generate_charts = input("Do you want to generate individual indicator charts? (y/n): ").strip().lower() == 'y'
    generate_heatmaps_flag = input("Do you want to generate heatmaps? (y/n): ").strip().lower() == 'y'
    save_correlation_csv = input("Do you want to save a single CSV containing each indicator's correlation values for each lag point? (y/n): ").strip().lower() == 'y'

    logger.info(f"Launching script with timestamp {new_timestamp}.")

    symbol = input("Enter the trading symbol (e.g., 'BTCUSDT'): ").strip().upper()
    timeframe = input("Enter the timeframe (e.g., '1d'): ").strip()
    logger.info("Loading data...")
    try:
        data, is_reverse_chronological, db_filename = load_data(symbol, timeframe)
        logger.info("Data loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load data: {e}", exc_info=True)
        sys.exit(1)

    if data.empty:
        logger.warning("Database is empty. Prompting user to download Binance data.")
        logger.info("No data found in the database for the specified symbol and timeframe.")
        if input("Do you want to download Binance historical data now? (y/n): ").strip().lower() == 'y':
            logger.info("Starting Binance data download...")
            try:
                download_binance_data(symbol, timeframe)
                data, is_reverse_chronological, db_filename = load_data(symbol, timeframe)
                if data.empty:
                    logger.error("Database still empty after downloading data. Check the downloader.", exc_info=True)
                    sys.exit(1)
                else:
                    logger.info("Data loaded successfully after downloading.")
            except Exception as e:
                logger.error(f"Failed to download or load data: {e}", exc_info=True)
                sys.exit(1)
        else:
            logger.info("User declined download. Exiting.")
            sys.exit(0)

    try:
        data = compute_all_indicators(data)
        logger.info("Indicators computed successfully.")
    except Exception as e:
        logger.error(f"Failed to compute indicators: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Data loaded and indicators computed.")

    if 'Date' not in data.columns:
        if 'open_time' in data.columns:
            data['Date'] = pd.to_datetime(data['open_time'], errors='coerce').dt.tz_localize(None)
            logger.info("'Date' column created from 'open_time'.")
        else:
            logger.error("No 'Date' or 'open_time' column found.", exc_info=True)
            sys.exit(1)

    if 'Close' not in data.columns:
        if 'close' in data.columns:
            data.rename(columns={'close': 'Close'}, inplace=True)
            logger.info("'close' renamed to 'Close'.")
        else:
            logger.error("'Close' column missing.", exc_info=True)
            sys.exit(1)

    try:
        time_interval = determine_time_interval(data)
        logger.info(f"Determined time interval: {time_interval}")
    except Exception as e:
        logger.error(f"Failed to determine time interval: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Determined time interval: {time_interval}")
    logger.info("Preparing data for analysis...")

    try:
        X_scaled, feature_names = prepare_data(data)
    except Exception as e:
        logger.error(f"Data preparation failed: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Data prepared.")
    logger.info(f"Features: {feature_names}")
    original_indicators = get_original_indicators(feature_names, data)
    expected_indicators = ['FI', 'ichimoku', 'KCU_20_2.0', 'STOCHRSI_14_5_3_slowk', 'VI+_14']
    original_indicators = handle_missing_indicators(original_indicators, data, expected_indicators)
    if not original_indicators:
        logger.error("No valid indicators after excluding missing.", exc_info=True)
        sys.exit(1)

    max_lag = len(data) - 51
    if max_lag < 1:
        logger.error("Insufficient data for given max_lag.", exc_info=True)
        sys.exit(1)

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
        logger.error(str(ve), exc_info=True)
        sys.exit(1)

    correlation_db = CorrelationDatabase(DB_PATH)
    correlations = {}
    for indicator in original_indicators:
        correlations[indicator] = []
        for lag in range(1, max_lag + 1):
            correlations[indicator].append(correlation_db.get_correlation(symbol, timeframe, indicator, lag))
    correlation_db.close()

    try:
        summary_df = generate_statistical_summary(correlations, max_lag)
        summary_csv = os.path.join(csv_dir, f"{timestamp}_{symbol}_{timeframe}_statistical_summary.csv")
        summary_df.to_csv(summary_csv, index=True)
        logger.info(f"Generated statistical summary: {summary_csv}")
    except Exception as e:
        logger.error(f"Failed to generate summary: {e}", exc_info=True)

    try:
        generate_combined_correlation_chart(correlations, max_lag, time_interval, timestamp, f"{symbol}_{timeframe}")
    except Exception as e:
        logger.error(f"Failed to generate combined correlation chart: {e}", exc_info=True)

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
                calculate_correlation_func=calculate_correlation,
                base_csv_filename=f"{symbol}_{timeframe}"
            )
            logger.info("Visualization done.")
        except Exception as e:
            logger.error(f"Visualization error: {e}", exc_info=True)

    if generate_heatmaps_flag:
        try:
            generate_heatmaps(
                data=data,
                timestamp=timestamp,
                time_interval=time_interval,
                generate_heatmaps_flag=generate_heatmaps_flag,
                cache=correlations,
                calculate_correlation=calculate_correlation,
                base_csv_filename=f"{symbol}_{timeframe}"
            )
            logger.info("Heatmaps done.")
        except Exception as e:
            logger.error(f"Heatmaps error: {e}", exc_info=True)

    try:
        best_indicators_df = generate_best_indicator_table(correlations, max_lag)
        best_indicators_csv = os.path.join(csv_dir, f"{timestamp}_{symbol}_{timeframe}_best_indicators.csv")
        best_indicators_df.to_csv(best_indicators_csv, index=False)
        logger.info(f"Generated best indicator table: {best_indicators_csv}")
    except Exception as e:
        logger.error(f"Best table error: {e}", exc_info=True)

    if save_correlation_csv:
        try:
            generate_correlation_csv(correlations, max_lag, f"{symbol}_{timeframe}", csv_dir)
        except Exception as e:
            logger.error(f"Correlation CSV error: {e}", exc_info=True)

    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
    latest_date_in_data = data['Date'].max()
    logger.info(f"\nLatest date/time in data: {latest_date_in_data}")
    logger.info(f"Latest in data: {latest_date_in_data}")
    current_datetime = datetime.now()
    time_interval_seconds_map = {'second': 1, 'minute': 60, 'hour': 3600, 'day': 86400, 'week': 604800}
    if time_interval not in time_interval_seconds_map:
        logger.error(f"Unsupported interval {time_interval}", exc_info=True)
        sys.exit(1)

    time_diff_seconds = (current_datetime - latest_date_in_data).total_seconds()
    lag_periods_behind_current = int(time_diff_seconds / time_interval_seconds_map[time_interval])
    logger.info(f"Data is {lag_periods_behind_current} {time_interval}(s) behind current time.")

    logger.info("\nEnter future date/time for prediction. Format: YYYYMMDD-HHMM or relative like +1h, or blank for current.")
    user_input = input("Future date/time: ").strip()
    try:
        future_datetime = parse_date_time_input(user_input, current_datetime) if user_input else current_datetime
        logger.info(f"Using future date/time: {future_datetime}" if user_input else f"No input, using current: {future_datetime}")
    except ValueError as e:
        logger.error(f"Date parse error: {e}", exc_info=True)
        sys.exit(1)

    lag_seconds = (future_datetime - latest_date_in_data).total_seconds()
    if lag_seconds <= 0:
        logger.error("Future must be after latest data.", exc_info=True)
        sys.exit(1)
    lag_periods = int(lag_seconds / time_interval_seconds_map[time_interval])
    if lag_periods < 1:
        logger.error("Must be at least one lag ahead.", exc_info=True)
        sys.exit(1)
    if lag_periods > max_lag:
        logger.error(f"Lag must be ≤ {max_lag}.", exc_info=True)
        sys.exit(1)

    logger.info(f"Lag period: {lag_periods} {time_interval}(s)")

    try:
        perform_linear_regression(data, correlations, max_lag, time_interval, timestamp, f"{symbol}_{timeframe}", future_datetime, lag_periods)
        logger.info("Linear regression done.")
    except Exception as e:
        logger.error(f"Linear regression failed: {e}", exc_info=True)

    try:
        advanced_price_prediction(data, correlations, max_lag, time_interval, timestamp, f"{symbol}_{timeframe}", future_datetime, lag_periods)
        logger.info("Advanced prediction done.")
    except Exception as e:
        logger.error(f"Advanced prediction failed: {e}", exc_info=True)

    logger.info("All done.")
    logger.info("All processes completed successfully.")

if __name__ == "__main__":
    main()