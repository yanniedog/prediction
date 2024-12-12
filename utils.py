import os, sys, json, shutil, subprocess, re
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import t
from sklearn.preprocessing import StandardScaler
import dateutil.parser
from load_data import load_data
from indicators import compute_all_indicators
from data_utils import clear_screen, prepare_data, determine_time_interval, get_original_indicators, handle_missing_indicators
from correlation_utils import load_or_calculate_correlations, calculate_correlation, calculate_and_save_indicator_correlations
from visualization_utils import generate_combined_correlation_chart, visualize_data, generate_heatmaps
from backup_utils import run_backup_cleanup
from table_generation import generate_best_indicator_table, generate_statistical_summary, generate_correlation_csv
from binance_historical_data_downloader import download_binance_data, fetch_klines, process_klines, save_to_sqlite
import warnings
warnings.filterwarnings('ignore')

def run_backup_cleanup():
    pass

def list_database_files(database_dir: str) -> List[str]:
    return [f for f in os.listdir(database_dir) if f.endswith('.db')]

def select_existing_database(database_dir: str) -> Optional[str]:
    db_files = list_database_files(database_dir)
    if not db_files:
        print("No existing databases found.")
        return None
    print("\nExisting Databases:")
    for idx, db in enumerate(db_files,1): print(f"{idx}. {db}")
    while True:
        selected = input(f"Select DB (1-{len(db_files)}) or 'x' to exit: ").strip()
        if selected.lower() == 'x': return None
        if selected.isdigit() and 1 <= int(selected) <= len(db_files):
            selected_db = db_files[int(selected)-1]
            print(f"Selected DB: {selected_db}")
            return os.path.join(database_dir, selected_db)
        print("Invalid selection.")

def preview_database(db_path: str) -> None:
    try:
        data, is_rev, _ = load_data(db_path)
        if data.empty:
            print("Selected DB is empty.")
        else:
            print(f"\nLatest data in '{os.path.basename(db_path)}':\n{data.tail()}")
    except:
        print(f"Failed to preview DB '{db_path}'.")

def update_database(db_path: str) -> None:
    try:
        base = os.path.basename(db_path).split('.')[0]
        symbol, interval = base.split('_')
    except ValueError:
        print("DB filename must be 'symbol_interval.db'.")
        return
    print(f"Updating DB for {symbol} {interval}...")
    start = input("Start date (YYYY-MM-DD) or Enter for latest: ").strip()
    end = input("End date (YYYY-MM-DD) or Enter for today: ").strip()
    try:
        data, is_rev, _ = load_data(db_path)
        if data.empty:
            print("DB empty. Downloading full dataset.")
            download_binance_data(symbol, interval, db_path)
            return
    except:
        print(f"Failed to load data from '{db_path}'.")
        return
    start_dt = datetime.strptime(start, '%Y-%m-%d') if start else data['Date'].max() + timedelta(seconds=1)
    end_dt = datetime.strptime(end, '%Y-%m-%d') if end else datetime.now()
    start_ts, end_ts = int(start_dt.timestamp()*1000), int(end_dt.timestamp()*1000)
    if start_ts >= end_ts:
        print("Start must be before end.")
        return
    try:
        klines = fetch_klines(symbol, interval, start_ts, end_ts)
        if not klines:
            print("No new data.")
            return
        df = process_klines(klines)
        save_to_sqlite(df, db_path)
        print(f"Updated '{os.path.basename(db_path)}' with {len(df)} records.")
    except:
        print(f"Failed to update DB '{db_path}'.")
