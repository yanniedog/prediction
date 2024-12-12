import os, sys, shutil, re, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns, dateutil.parser, warnings
from pathlib import Path
from datetime import datetime, timedelta
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from scipy.stats import t
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
import sqlite3

warnings.filterwarnings('ignore')

def parse_date_time_input(user_input: str, ref_dt: datetime) -> datetime:
    user_input = user_input.strip()
    if not user_input: return ref_dt
    rel_pat = r'^([+-]\d+)([smhdw])$'
    match = re.match(rel_pat, user_input)
    if match:
        amt, unit = int(match.group(1)), match.group(2)
        delta = {'s': 'seconds', 'm': 'minutes', 'h': 'hours', 'd': 'days', 'w': 'weeks'}[unit]
        return ref_dt + timedelta(**{delta: amt})
    for fmt in ['%Y%m%d-%H%M', '%Y%m%d']:
        try: return datetime.strptime(user_input, fmt)
        except: continue
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
        if val in ['y','yes']: return 'y'
        elif val in ['n','no']: return 'n'
        print("Invalid input, enter 'y' or 'n'.")

def preview_database(db_path: str):
    try:
        conn = sqlite3.connect(db_path)
        for tbl in ['symbols','timeframes','klines']:
            df = pd.read_sql_query(f"SELECT * FROM {tbl} LIMIT 5", conn)
            print(f"{tbl.capitalize()} table preview:\n{df.head()}")
        conn.close()
        return True
    except:
        print(f"Error previewing DB '{db_path}'.")
        return False

def recreate_database(db_path: str):
    print("Recreating database...")
    if os.path.exists(db_path): os.remove(db_path)
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
    for d in ['reports','csv']: os.makedirs(d, exist_ok=True)
    if input_yes_no("Delete previous output? (y/n): ") == 'y':
        for folder in ['indicator_charts','heatmaps','combined_charts','reports']:
            for f in Path(folder).glob('*'):
                f.unlink() if f.is_file() else shutil.rmtree(f)
        print("Outputs cleared.")
    gen_charts = input_yes_no("Generate individual charts? (y/n): ") == 'y'
    gen_heatmaps = input_yes_no("Generate heatmaps? (y/n): ") == 'y'
    save_corr_csv = input_yes_no("Save correlation CSV? (y/n): ") == 'y'
    symbol = input_with_default("Enter symbol (e.g., 'BTCUSDT'): ", "BTCUSDT").upper()
    timeframe = input_with_default("Enter timeframe (e.g., '1d'): ", "1d")
    print(f"Symbol: {symbol}, Timeframe: {timeframe}")
    data, is_rev, db_fn = load_data(symbol, timeframe)
    if not preview_database(DB_PATH):
        while True:
            choice = input_yes_no_no_default("Invalid DB. Recreate? (y/n): ")
            if choice == 'y':
                recreate_database(DB_PATH)
                data, is_rev, db_fn = load_data(symbol, timeframe)
                if data.empty:
                    if input_yes_no("Download data? (y/n): ") == 'y':
                        download_binance_data(symbol, timeframe)
                        data, is_rev, db_fn = load_data(symbol, timeframe)
                        if data.empty: sys.exit(1)
                        else: break
                    else: sys.exit(1)
                else: break
            elif choice == 'n':
                sys.exit(1)
    if data.empty:
        if input_yes_no("Download data? (y/n): ") == 'y':
            download_binance_data(symbol, timeframe)
            data, is_rev, db_fn = load_data(symbol, timeframe)
            if data.empty: sys.exit(1)
        else:
            sys.exit(0)
    data = compute_all_indicators(data)
    data['Date'] = pd.to_datetime(data.get('Date') or data.get('open_time'), errors='coerce').dt.tz_localize(None)
    if 'close' in data.columns: data.rename(columns={'close':'Close'}, inplace=True)
    time_interval = determine_time_interval(data)
    X_scaled, feature_cols = prepare_data(data)
    original_inds = get_original_indicators(feature_cols, data)
    expected_inds = ['FI','ichimoku','KCU_20_2.0','STOCHRSI_14_5_3_slowk','VI+_14']
    original_inds = handle_missing_indicators(original_inds, data, expected_inds)
    if not original_inds:
        print("No valid indicators.")
        sys.exit(1)
    max_lag = len(data) -51
    if max_lag <1:
        print("Insufficient data.")
        sys.exit(1)
    load_or_calculate_correlations(data, original_inds, max_lag, is_rev, symbol, timeframe)
    db = CorrelationDatabase(DB_PATH)
    correlations = {ind: [db.get_correlation(symbol, timeframe, ind, lag) for lag in range(1,max_lag+1)] for ind in original_inds}
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
    latest_date = data['Date'].max()
    current_dt = datetime.now()
    interval_sec = {'second':1, 'minute':60, 'hour':3600, 'day':86400, 'week':604800}[time_interval]
    lag_sec = (current_dt - latest_date).total_seconds()
    lag_periods = int(lag_sec / interval_sec)
    user_input = input("Future date/time: ").strip()
    future_dt = parse_date_time_input(user_input, current_dt) if user_input else current_dt
    lag_sec = (future_dt - latest_date).total_seconds()
    lag_periods = int(lag_sec / interval_sec)
    if lag_periods <=0 or lag_periods > max_lag: sys.exit(1)
    perform_linear_regression(data, correlations, max_lag, time_interval, timestamp, f"{symbol}_{timeframe}", future_dt, lag_periods)
    advanced_price_prediction(data, correlations, max_lag, time_interval, timestamp, f"{symbol}_{timeframe}", future_dt, lag_periods)
    print("Script completed successfully.")

if __name__=="__main__":
    main()
