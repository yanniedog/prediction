# start.py
import os
import subprocess
import sys
import shutil
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import dateutil.parser
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from scipy.stats import t
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
import sqlite3

warnings.filterwarnings('ignore')

def parse_date_time_input(user_input: str, ref_dt: datetime) -> datetime:
    user_input = user_input.strip()
    if not user_input:
        return ref_dt
    rel_pat = r'^([+-]\d+)([smhdw])$'
    match = re.match(rel_pat, user_input)
    if match:
        amt, unit = int(match.group(1)), match.group(2)
        delta = {'s': 'seconds', 'm': 'minutes', 'h': 'hours', 'd': 'days', 'w': 'weeks'}[unit]
        return ref_dt + timedelta(**{delta: amt})
    for fmt in ['%Y%m%d-%H%M', '%Y%m%d']:
        try:
            return datetime.strptime(user_input, fmt)
        except:
            continue
    return dateutil.parser.parse(user_input, fuzzy=True)

def input_with_default(prompt: str, default: str) -> str:
    val = input(prompt).strip()
    return val if val else default

def input_yes_no(prompt: str, default: str = 'y') -> str:
    val = input(prompt).strip().lower()
    return 'y' if val.startswith('y') else 'n' if val.startswith('n') else default

def input_yes_no_no_default(prompt: str) -> str:
    while True:
        val = input(prompt).strip().lower()
        if val in ['y', 'yes']:
            return 'y'
        elif val in ['n', 'no']:
            return 'n'
        print("Invalid input, enter 'y' or 'n'.")

def delete_previous_output():
    from pathlib import Path
    import os

    target_dirs = ['csv', 'heatmaps', 'combined_charts', 'reports', 'database']
    delete_outputs = input("Delete contents of output directories? (y/n): ").strip().lower()

    if delete_outputs == 'y':
        for folder in target_dirs:
            folder_path = Path(folder)
            if folder_path.exists() and folder_path.is_dir():
                try:
                    for item in folder_path.iterdir():
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            for sub_item in item.rglob('*'):
                                if sub_item.is_file():
                                    sub_item.unlink()
                                elif sub_item.is_dir():
                                    sub_item.rmdir()
                            item.rmdir()
                    print(f"Cleared contents of: {folder_path}")
                except Exception as e:
                    print(f"Error clearing contents of {folder_path}: {e}")
        print("All selected directory contents have been cleared.")
    else:
        print("Directory contents not cleared.")



def recreate_database(db_path: str):
    print("Recreating database...")
    if os.path.exists(db_path):
        os.remove(db_path)
    from sqlite_data_manager import create_connection, create_tables
    conn = create_connection(db_path)
    if conn:
        create_tables(conn)
        conn.close()
        print("New database created.")
    else:
        print("Failed to create DB.")
        sys.exit(1)

def main():
    clear_screen()
    run_backup_cleanup()
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    for d in ['reports', 'csv']:
        Path(d).mkdir(exist_ok=True)

    delete_previous_output()

    gen_charts = input_yes_no("Generate individual charts? (y/n): ") == 'y'
    gen_heatmaps = input_yes_no("Generate heatmaps? (y/n): ") == 'y'
    save_corr_csv = input_yes_no("Save correlation CSV? (y/n): ") == 'y'
    tweak = input_yes_no("Do you want to tweak indicator settings? (y/n): ") == 'y'

    symbol = input_with_default("Enter symbol (e.g., 'BTCUSDT'): ", "BTCUSDT").upper()
    timeframe = input_with_default("Enter timeframe (e.g., '1w'): ", "1w")
    print(f"Symbol: {symbol}, Timeframe: {timeframe}")

    if tweak:
        subprocess.run([sys.executable, 'tweak-indicator.py', symbol, timeframe])

    data, is_rev, db_fn = load_data(symbol, timeframe)
    if not data.empty:
        if input_yes_no("Invalid DB. Recreate? (y/n): ") == 'y':
            recreate_database(DB_PATH)
            data, is_rev, db_fn = load_data(symbol, timeframe)

    if data.empty and input_yes_no("Download data? (y/n): ") == 'y':
        download_binance_data(symbol, timeframe)
        data, is_rev, db_fn = load_data(symbol, timeframe)

    if data.empty:
        print("No data available.")
        sys.exit(0)

    data = compute_all_indicators(data)
    data['Date'] = pd.to_datetime(data.get('Date') or data.get('open_time'), errors='coerce').dt.tz_localize(None)
    if 'close' in data.columns:
        data.rename(columns={'close': 'Close'}, inplace=True)

    time_interval = determine_time_interval(data)
    X_scaled, feature_cols = prepare_data(data)
    original_inds = get_original_indicators(feature_cols, data)
    if not original_inds:
        print("No valid indicators.")
        sys.exit(1)

    max_lag = len(data) - 51
    if max_lag < 1:
        print("Insufficient data.")
        sys.exit(1)

    load_or_calculate_correlations(data, original_inds, max_lag, is_rev, symbol, timeframe)
    db = CorrelationDatabase(DB_PATH)
    correlations = {ind: [db.get_correlation(symbol, timeframe, ind, lag) for lag in range(1, max_lag + 1)] for ind in original_inds}
    db.close()

    summary_df = generate_statistical_summary(correlations, max_lag)
    summary_df.to_csv(f"csv/{timestamp}_{symbol}_{timeframe}_stat_summary.csv")

    generate_combined_correlation_chart(correlations, max_lag, time_interval, timestamp, f"{symbol}_{timeframe}")

    if gen_charts:
        visualize_data(data, X_scaled, feature_cols, timestamp, is_rev, time_interval, gen_charts, correlations, calculate_correlation, f"{symbol}_{timeframe}")

    if gen_heatmaps:
        generate_heatmaps(data, timestamp, time_interval, gen_heatmaps, correlations, calculate_correlation, f"{symbol}_{timeframe}")

    best_df = generate_best_indicator_table(correlations, max_lag)
    best_df.to_csv(f"csv/{timestamp}_{symbol}_{timeframe}_best_indicators.csv", index=False)

    if save_corr_csv:
        generate_correlation_csv(correlations, max_lag, f"{symbol}_{timeframe}", 'csv')

    print("Script completed successfully.")

if __name__ == "__main__":
    main()
