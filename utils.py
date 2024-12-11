# filename: utils.py
import os
import sys
import json
import shutil
import subprocess
import re
from pathlib import Path
from datetime import datetime,timedelta
from typing import List,Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel,delayed
from scipy.stats import t
from sklearn.preprocessing import StandardScaler
import dateutil.parser
from linear_regression import perform_linear_regression
from advanced_analysis import advanced_price_prediction
from load_data import load_data
from indicators import compute_all_indicators
from data_utils import clear_screen,prepare_data,determine_time_interval,get_original_indicators,handle_missing_indicators
from correlation_utils import load_or_calculate_correlations,calculate_correlation,calculate_and_save_indicator_correlations
from visualization_utils import generate_combined_correlation_chart,visualize_data,generate_heatmaps
from backup_utils import run_backup_cleanup
from table_generation import generate_best_indicator_table,generate_statistical_summary,generate_correlation_csv
from binance_historical_data_downloader import download_binance_data,fetch_klines,process_klines,save_to_sqlite
import warnings
warnings.filterwarnings('ignore')

def run_backup_cleanup():
    pass

def list_database_files(database_dir:str)->List[str]:
    return[f for f in os.listdir(database_dir)if f.endswith('.db')]

def select_existing_database(database_dir:str)->Optional[str]:
    db_files=list_database_files(database_dir)
    if not db_files:
        print("No existing databases found.")
        return None
    print("\nExisting Databases:")
    for idx,db in enumerate(db_files,1):
        print(f"{idx}. {db}")
    while True:
        selected=input(f"Enter the number of the database to select (1-{len(db_files)}) or type 'x' to go back: ").strip()
        if selected.lower()=='x':
            return None
        if selected.isdigit()and 1<=int(selected)<=len(db_files):
            selected_db=db_files[int(selected)-1]
            print(f"Selected Database: {selected_db}")
            return os.path.join(database_dir,selected_db)
        else:
            print("Invalid selection. Please try again.")

def preview_database(db_path:str)->None:
    try:
        data,is_reverse_chronological,_=load_data(db_path)
        if data.empty:
            print("The selected database is empty.")
        else:
            print(f"\nPreview of the latest data in '{os.path.basename(db_path)}':")
            print(data.tail())
    except:
        print(f"Failed to preview database '{db_path}'.")

def update_database(db_path:str)->None:
    try:
        base_filename=os.path.basename(db_path)
        symbol,interval=os.path.splitext(base_filename)[0].split('_')
    except ValueError:
        print(f"Database filename '{db_path}' does not follow the 'symbol_interval.db' format.")
        return
    print(f"Updating database for {symbol} with interval {interval}...")
    print("Please enter the date range.")
    start_date_input=input("Enter the start date (YYYY-MM-DD) or press Enter to use the latest date in the database: ").strip()
    end_date_input=input("Enter the end date (YYYY-MM-DD) or press Enter to use today's date: ").strip()
    try:
        data,is_reverse_chronological,_=load_data(db_path)
        if data.empty:
            print("The selected database is empty. Downloading full dataset.")
            download_binance_data(symbol,interval,db_path)
            return
    except:
        print(f"Failed to load data from '{db_path}'.")
        return
    if start_date_input:
        try:
            start_datetime=datetime.strptime(start_date_input,'%Y-%m-%d')
        except ValueError:
            print("Invalid start date format.")
            return
    else:
        latest_timestamp=data['Date'].max()
        start_datetime=latest_timestamp+timedelta(seconds=1)
        print(f"No start date entered. Using {start_datetime}")
    if end_date_input:
        try:
            end_datetime=datetime.strptime(end_date_input,'%Y-%m-%d')
        except ValueError:
            print("Invalid end date format.")
            return
    else:
        end_datetime=datetime.now()
        print(f"No end date entered. Using {end_datetime}")
    start_time=int(start_datetime.timestamp()*1000)
    end_time=int(end_datetime.timestamp()*1000)
    if start_time>=end_time:
        print("Start date must be before end date.")
        return
    try:
        klines=fetch_klines(symbol,interval,start_time,end_time)
        if not klines:
            print("No new data available.")
            return
        df=process_klines(klines)
        save_to_sqlite(df,db_path)
        print(f"Updated '{os.path.basename(db_path)}' with {len(df)} new records.")
    except:
        print(f"Failed to update database '{db_path}'.")

def download_new_dataset(database_dir:str,default_interval:str='1d')->Optional[str]:
    return create_new_database(database_dir,default_interval=default_interval)

def create_new_database(database_dir:str,default_interval:str='1d')->Optional[str]:
    symbol=input("Enter the trading symbol (e.g., BTCUSDT): ").strip().upper()
    if not symbol:
        print("Symbol cannot be empty.")
        return None
    interval=default_interval
    db_filename=f"{symbol}_{interval}.db"
    db_path=os.path.join(database_dir,db_filename)
    if os.path.exists(db_path):
        print(f"Database '{db_filename}' already exists.")
        return db_path
    else:
        download_binance_data(symbol,interval,db_path)
        return db_path

def perform_analysis(db_path:str,reports_dir:str,cache_dir:str,timestamp:str)->None:
    try:
        data,is_reverse_chronological,db_filename=load_data(db_path)
    except:
        print(f"Failed to load data: {db_path}")
        return
    if data.empty:
        print("No data found.")
        download_choice=input("Download now? (y/n): ").strip().lower()
        if download_choice=='y':
            try:
                symbol,interval=os.path.splitext(os.path.basename(db_path))[0].split('_')
            except ValueError:
                print("Filename invalid.")
                return
            download_binance_data(symbol,interval,db_path)
            try:
                data,is_reverse_chronological,db_filename=load_data(db_path)
                if data.empty:
                    print("DB empty.")
                    return
                else:
                    print("Data loaded.")
            except:
                print("Load fail.")
                return
        else:
            print("No data.")
            return
    try:
        data=compute_all_indicators(data)
    except:
        print("Fail.")
        return
    print("Data and indicators done.")
    try:
        time_interval=determine_time_interval(data)
    except:
        print("Fail.")
        return
    print(f"Interval: {time_interval}")
    print("Preparing data...")
    try:
        X_scaled,feature_names=prepare_data(data)
        print("Data prepared.")
    except:
        print("Prep fail.")
        return
    original_indicators=get_original_indicators(feature_names,data)
    expected_indicators=['FI','ichimoku','KCU_20_2.0','STOCHRSI_14_5_3_slowk','VI+_14']
    original_indicators=handle_missing_indicators(original_indicators,data,expected_indicators)
    if not original_indicators:
        print("No valid indicators.")
        return
    base_csv_filename=os.path.splitext(os.path.basename(db_filename))[0]
    max_lag=len(data)-51
    if max_lag<1:
        print("Insufficient data.")
        return
    try:
        correlations=load_or_calculate_correlations(data,original_indicators,max_lag,is_reverse_chronological,base_csv_filename,db_path=db_path)
    except ValueError as ve:
        print(str(ve))
        return
    except:
        print("Corr fail.")
        return
    try:
        summary_df=generate_statistical_summary(correlations,max_lag)
        summary_csv=os.path.join(reports_dir,f"{timestamp}_{base_csv_filename}_statistical_summary.csv")
        summary_df.to_csv(summary_csv,index=True)
        print(f"Summary: {summary_csv}")
    except:
        print("Fail.")
    try:
        generate_combined_correlation_chart(correlations,max_lag,time_interval,timestamp,base_csv_filename)
    except:
        print("Fail.")
    generate_charts=input("Charts? (y/n): ").strip().lower()=='y'
    generate_heatmaps_flag=input("Heatmaps? (y/n): ").strip().lower()=='y'
    save_correlation_csv=input("Save correlation CSV? (y/n): ").strip().lower()=='y'
    if generate_charts:
        try:
            visualize_data(data,X_scaled,feature_names,timestamp,is_reverse_chronological,time_interval,generate_charts,correlations,calculate_correlation,base_csv_filename)
            print("Viz done.")
        except:
            print("Fail.")
    if generate_heatmaps_flag:
        try:
            generate_heatmaps(data,timestamp,time_interval,generate_heatmaps_flag,correlations,calculate_correlation,base_csv_filename)
            print("Heatmaps done.")
        except:
            print("Fail.")
    try:
        best_indicators_df=generate_best_indicator_table(correlations,max_lag)
        best_indicators_csv=os.path.join(reports_dir,f"{timestamp}_{base_csv_filename}_best_indicators.csv")
        best_indicators_df.to_csv(best_indicators_csv,index=False)
        print(f"Best: {best_indicators_csv}")
    except:
        print("Fail.")
    if save_correlation_csv:
        try:
            generate_correlation_csv(correlations,max_lag,base_csv_filename,reports_dir)
        except TypeError as te:
            print(f"Error: {te}")
        except:
            print("Fail.")
    try:
        data['Date']=pd.to_datetime(data['Date'])
        latest_date_in_data=data['Date'].max()
        print(f"\nLatest: {latest_date_in_data}")
    except:
        print("Fail.")
        return
    current_datetime=datetime.now()
    time_interval_seconds={'1s':1,'1m':60,'5m':300,'15m':900,'30m':1800,'1h':3600,'4h':14400,'1d':86400,'1w':604800}
    if time_interval not in time_interval_seconds:
        print(f"Unsupported '{time_interval}'.")
        return
    time_diff_seconds=(current_datetime - latest_date_in_data).total_seconds()
    lag_periods_behind_current=int(time_diff_seconds/time_interval_seconds[time_interval])
    print("\nRelative time?")
    user_input=input("Future date/time: ").strip()
    try:
        future_datetime=parse_date_time_input(user_input,current_datetime)if user_input else current_datetime
        if user_input:
            print(f"Using {future_datetime}")
        else:
            print(f"Using current {future_datetime}")
    except:
        print("Parse err.")
        return
    lag_seconds=(future_datetime - latest_date_in_data).total_seconds()
    if lag_seconds<=0:
        print("Future after latest!")
        return
    lag_periods=int(lag_seconds/time_interval_seconds[time_interval])
    if lag_periods<1:
        print("Lag <1.")
        return
    if lag_periods>max_lag:
        print(f"Lag ≤ {max_lag}")
        return
    print(f"Lag: {lag_periods}")
    try:
        perform_linear_regression(data,correlations,max_lag,time_interval,timestamp,base_csv_filename,future_datetime,lag_periods)
        print("LR done.")
    except TypeError as te:
        print(f"LR err: {te}")
    except:
        print("LR fail.")
    try:
        advanced_price_prediction(data,correlations,max_lag,time_interval,timestamp,base_csv_filename,future_datetime,lag_periods)
        print("Adv done.")
    except TypeError as te:
        print(f"Adv err: {te}")
    except:
        print("Adv fail.")
    try:
        calculate_and_save_indicator_correlations(data,original_indicators,max_lag,is_reverse_chronological,db_path)
        print("Ind-ind corr done.")
    except:
        print("Fail.")
    print("All done.")

def parse_date_time_input(user_input:str,reference_datetime:datetime)->datetime:
    user_input=user_input.strip()
    if not user_input:
        return reference_datetime
    relative_time_pattern=r'^([+-]\d+)([smhdw])$'
    match=re.match(relative_time_pattern,user_input)
    if match:
        amount=int(match.group(1))
        unit=match.group(2)
        delta_kwargs={'s':'seconds','m':'minutes','h':'hours','d':'days','w':'weeks'}
        delta=timedelta(**{delta_kwargs[unit]:amount})
        return reference_datetime+delta
    for fmt in['%Y%m%d-%H%M','%Y%m%d']:
        try:
            return datetime.strptime(user_input,fmt)
        except ValueError:
            continue
    return dateutil.parser.parse(user_input,fuzzy=True)