# filename: utils.py
import os
import sys
import json
import logging
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
try:
    import dateutil.parser
except ImportError:
    print("Required module 'python-dateutil' not found. Please install it.")
    sys.exit(1)
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
def configure_logging():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',handlers=[logging.FileHandler("application.log"),logging.StreamHandler(sys.stdout)])
def run_backup_cleanup():
    logging.info("Running backup and cleanup operations.")
    pass
def list_database_files(database_dir:str)->List[str]:
    return[f for f in os.listdir(database_dir)if f.endswith('.db')]
def select_existing_database(database_dir:str)->Optional[str]:
    db_files=list_database_files(database_dir)
    if not db_files:
        print("No existing databases found.")
        logging.info("No existing databases found.")
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
            logging.info(f"Selected existing database: {selected_db}")
            return os.path.join(database_dir,selected_db)
        else:
            print("Invalid selection. Please try again.")
def preview_database(db_path:str)->None:
    try:
        data,is_reverse_chronological,_=load_data(db_path)
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
def update_database(db_path:str)->None:
    try:
        base_filename=os.path.basename(db_path)
        symbol,interval=os.path.splitext(base_filename)[0].split('_')
    except ValueError:
        logging.error(f"Database filename '{db_path}' does not follow the 'symbol_interval.db' format.")
        print(f"Database filename '{db_path}' does not follow the 'symbol_interval.db' format.")
        return
    print(f"Updating database for {symbol} with interval {interval}...")
    logging.info(f"Updating database '{db_path}' for symbol '{symbol}' and interval '{interval}'.")
    print("Please enter the date range.")
    start_date_input=input("Enter the start date (YYYY-MM-DD) or press Enter to use the latest date in the database: ").strip()
    end_date_input=input("Enter the end date (YYYY-MM-DD) or press Enter to use today's date: ").strip()
    try:
        data,is_reverse_chronological,_=load_data(db_path)
        if data.empty:
            print("The selected database is empty. Downloading full dataset.")
            logging.warning(f"Database '{db_path}' is empty. Initiating full download.")
            download_binance_data(symbol,interval,db_path)
            return
    except Exception as e:
        logging.error(f"Failed to load data from '{db_path}': {e}")
        print(f"Failed to load data from '{db_path}': {e}")
        return
    if start_date_input:
        try:
            start_datetime=datetime.strptime(start_date_input,'%Y-%m-%d')
        except ValueError:
            print("Invalid start date format.")
            logging.error("Invalid start date format entered.")
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
            logging.error("Invalid end date format entered.")
            return
    else:
        end_datetime=datetime.now()
        print(f"No end date entered. Using {end_datetime}")
    start_time=int(start_datetime.timestamp()*1000)
    end_time=int(end_datetime.timestamp()*1000)
    if start_time>=end_time:
        print("Start date must be before end date.")
        logging.error("Start date is not before end date.")
        return
    try:
        klines=fetch_klines(symbol,interval,start_time,end_time)
        if not klines:
            print("No new data available.")
            logging.info(f"No new data fetched for '{db_path}'.")
            return
        df=process_klines(klines)
        save_to_sqlite(df,db_path)
        print(f"Updated '{os.path.basename(db_path)}' with {len(df)} new records.")
        logging.info(f"Updated database '{db_path}' with {len(df)} new records.")
    except Exception as e:
        logging.error(f"Failed to update database '{db_path}': {e}")
        print(f"Failed to update database '{db_path}': {e}")
def download_new_dataset(database_dir:str,default_interval:str='1d')->Optional[str]:
    return create_new_database(database_dir,default_interval=default_interval)
def create_new_database(database_dir:str,default_interval:str='1d')->Optional[str]:
    symbol=input("Enter the trading symbol (e.g., BTCUSDT): ").strip().upper()
    if not symbol:
        print("Symbol cannot be empty.")
        logging.error("Symbol input was empty.")
        return None
    interval=default_interval
    db_filename=f"{symbol}_{interval}.db"
    db_path=os.path.join(database_dir,db_filename)
    if os.path.exists(db_path):
        print(f"Database '{db_filename}' already exists.")
        logging.warning(f"Database '{db_filename}' already exists.")
        return db_path
    else:
        download_binance_data(symbol,interval,db_path)
        return db_path
def perform_analysis(db_path:str,reports_dir:str,cache_dir:str,timestamp:str)->None:
    try:
        data,is_reverse_chronological,db_filename=load_data(db_path)
        logging.info("Data loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        print(f"Failed to load data: {e}")
        return
    if data.empty:
        logging.warning("Database is empty.")
        print("No data found.")
        download_choice=input("Download now? (y/n): ").strip().lower()
        if download_choice=='y':
            logging.info("User opted to download.")
            try:
                symbol,interval=os.path.splitext(os.path.basename(db_path))[0].split('_')
            except ValueError:
                logging.error(f"Filename '{db_path}' invalid.")
                print("Filename invalid.")
                return
            download_binance_data(symbol,interval,db_path)
            try:
                data,is_reverse_chronological,db_filename=load_data(db_path)
                if data.empty:
                    logging.error("DB still empty.")
                    print("DB empty.")
                    return
                else:
                    logging.info("Data loaded after download.")
                    print("Data loaded.")
            except Exception as e:
                logging.error(f"After download load fail: {e}")
                print(f"Load fail: {e}")
                return
        else:
            logging.info("User declined download.")
            print("No data.")
            return
    try:
        data=compute_all_indicators(data)
        logging.info("Indicators computed.")
    except Exception as e:
        logging.error(f"Compute indicators fail: {e}")
        print(f"Fail: {e}")
        return
    print("Data and indicators done.")
    try:
        time_interval=determine_time_interval(data)
        logging.info(f"Time interval: {time_interval}")
    except Exception as e:
        logging.error(f"Time interval fail: {e}")
        print(f"Fail: {e}")
        return
    print(f"Interval: {time_interval}")
    print("Preparing data...")
    try:
        X_scaled,feature_names=prepare_data(data)
        print("Data prepared.")
        logging.info(f"Features: {feature_names}")
    except Exception as e:
        logging.error(f"Prep fail: {e}")
        print(f"Prep fail: {e}")
        return
    original_indicators=get_original_indicators(feature_names,data)
    expected_indicators=['FI','ichimoku','KCU_20_2.0','STOCHRSI_14_5_3_slowk','VI+_14']
    original_indicators=handle_missing_indicators(original_indicators,data,expected_indicators)
    if not original_indicators:
        logging.error("No valid indicators.")
        print("No valid indicators.")
        return
    base_csv_filename=os.path.splitext(os.path.basename(db_filename))[0]
    cache_filename=os.path.join(cache_dir,f"{base_csv_filename}.json")
    max_lag=len(data)-51
    if max_lag<1:
        logging.error("Insufficient data.")
        print("Insufficient data.")
        return
    try:
        correlations=load_or_calculate_correlations(data,original_indicators,max_lag,is_reverse_chronological,cache_filename,db_path=db_path)
    except ValueError as ve:
        logging.error(str(ve))
        print(str(ve))
        return
    except Exception as e:
        logging.error(f"Corr fail: {e}")
        print(f"Corr fail: {e}")
        return
    try:
        summary_df=generate_statistical_summary(correlations,max_lag)
        summary_csv=os.path.join(reports_dir,f"{timestamp}_{base_csv_filename}_statistical_summary.csv")
        summary_df.to_csv(summary_csv,index=True)
        print(f"Summary: {summary_csv}")
        logging.info(f"Summary saved.")
    except Exception as e:
        logging.error(f"Summary fail: {e}")
        print(f"Fail: {e}")
    try:
        generate_combined_correlation_chart(correlations,max_lag,time_interval,timestamp,base_csv_filename)
    except Exception as e:
        logging.error(f"Comb chart fail: {e}")
        print(f"Fail: {e}")
    generate_charts=input("Charts? (y/n): ").strip().lower()=='y'
    generate_heatmaps_flag=input("Heatmaps? (y/n): ").strip().lower()=='y'
    save_correlation_csv=input("Save correlation CSV? (y/n): ").strip().lower()=='y'
    if generate_charts:
        try:
            visualize_data(data,X_scaled,feature_names,timestamp,is_reverse_chronological,time_interval,generate_charts,correlations,calculate_correlation,base_csv_filename)
            logging.info("Viz done.")
        except Exception as e:
            logging.error(f"Viz fail: {e}")
            print(f"Fail: {e}")
    if generate_heatmaps_flag:
        try:
            generate_heatmaps(data,timestamp,time_interval,generate_heatmaps_flag,correlations,calculate_correlation,base_csv_filename)
            logging.info("Heatmaps done.")
        except Exception as e:
            logging.error(f"Hmap fail: {e}")
            print(f"Fail: {e}")
    try:
        best_indicators_df=generate_best_indicator_table(correlations,max_lag)
        best_indicators_csv=os.path.join(reports_dir,f"{timestamp}_{base_csv_filename}_best_indicators.csv")
        best_indicators_df.to_csv(best_indicators_csv,index=False)
        print(f"Best: {best_indicators_csv}")
        logging.info("Best table done.")
    except Exception as e:
        logging.error(f"Best table fail: {e}")
        print(f"Fail: {e}")
    if save_correlation_csv:
        try:
            generate_correlation_csv(correlations,max_lag,base_csv_filename,reports_dir)
        except TypeError as te:
            logging.error(f"TypeError: {te}")
            print(f"Error: {te}")
        except Exception as e:
            logging.error(f"Corr table fail: {e}")
            print(f"Fail: {e}")
    try:
        data['Date']=pd.to_datetime(data['Date'])
        latest_date_in_data=data['Date'].max()
        print(f"\nLatest: {latest_date_in_data}")
        logging.info(f"Latest: {latest_date_in_data}")
    except Exception as e:
        logging.error(f"Latest fail: {e}")
        print(f"Fail: {e}")
        return
    current_datetime=datetime.now()
    time_interval_seconds={'1s':1,'1m':60,'5m':300,'15m':900,'30m':1800,'1h':3600,'4h':14400,'1d':86400,'1w':604800}
    if time_interval not in time_interval_seconds:
        logging.error(f"Unsupported '{time_interval}'.")
        print(f"Unsupported '{time_interval}'.")
        return
    time_diff_seconds=(current_datetime - latest_date_in_data).total_seconds()
    lag_periods_behind_current=int(time_diff_seconds/time_interval_seconds[time_interval])
    print(f"Latest is {lag_periods_behind_current} behind current.")
    logging.info(f"Behind current: {lag_periods_behind_current}")
    print("\nRelative time?")
    user_input=input("Future date/time: ").strip()
    try:
        future_datetime=parse_date_time_input(user_input,current_datetime)if user_input else current_datetime
        if user_input:
            print(f"Using {future_datetime}")
        else:
            print(f"Using current {future_datetime}")
    except ValueError as e:
        print(f"Parse err: {e}")
        logging.error(f"Parse err: {e}")
        return
    lag_seconds=(future_datetime - latest_date_in_data).total_seconds()
    if lag_seconds<=0:
        print("Future after latest!")
        logging.error("Future < latest.")
        return
    lag_periods=int(lag_seconds/time_interval_seconds[time_interval])
    if lag_periods<1:
        print("Lag <1.")
        logging.error("Lag <1.")
        return
    if lag_periods>max_lag:
        print(f"Lag ≤ {max_lag}")
        logging.error("Lag > max_lag.")
        return
    print(f"Lag: {lag_periods}")
    logging.info(f"Lag: {lag_periods}")
    try:
        perform_linear_regression(data,correlations,max_lag,time_interval,timestamp,base_csv_filename,future_datetime,lag_periods)
        logging.info("LR done.")
    except TypeError as te:
        logging.error(f"TypeError LR: {te}")
        print(f"LR err: {te}")
    except Exception as e:
        logging.error(f"LR fail: {e}")
        print(f"LR fail: {e}")
    try:
        advanced_price_prediction(data,correlations,max_lag,time_interval,timestamp,base_csv_filename,future_datetime,lag_periods)
        logging.info("Adv done.")
    except TypeError as te:
        logging.error(f"TypeError adv: {te}")
        print(f"Adv err: {te}")
    except Exception as e:
        logging.error(f"Adv fail: {e}")
        print(f"Adv fail: {e}")
    try:
        calculate_and_save_indicator_correlations(data,original_indicators,max_lag,is_reverse_chronological,db_path)
        logging.info("Ind-ind corr done.")
    except Exception as e:
        logging.error(f"Ind-ind corr fail: {e}")
        print(f"Fail: {e}")
    logging.info("All done.")
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