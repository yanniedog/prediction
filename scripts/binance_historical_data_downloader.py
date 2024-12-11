# binance_historical_data_downloader.py
import os
import sys
import requests
import pandas as pd
import time
import datetime
from config import DB_PATH
from sqlite_data_manager import save_to_sqlite, create_connection, create_tables
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

BASE_URL = 'https://api.binance.com'

def get_historical_klines(symbol, interval, start_time, end_time):
    limit = 1000
    klines = []
    while True:
        url = f"{BASE_URL}/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }
        logger.debug(f"Requesting URL: {url} with params: {params}")
        response = requests.get(url, params=params)
        if response.status_code != 200:
            logger.error(f"Failed to fetch klines: {response.status_code} - {response.text}")
            break
        data = response.json()
        if not data:
            logger.info("No more data to fetch.")
            break
        klines.extend(data)
        start_time = data[-1][0] + 1
        if len(data) < limit:
            logger.info("Fetched all available data.")
            break
        time.sleep(0.5)
    return klines

def download_binance_data(symbol=None, interval=None, start_date_str=None, end_date_str=None):
    """
    Downloads historical kline data from Binance and saves it to CSV and SQLite.

    Parameters:
    - symbol: Trading symbol (e.g., 'SOLUSDT').
    - interval: Kline interval (e.g., '1w').
    - start_date_str: Start date in 'YYYYMMDD' format or None.
    - end_date_str: End date in 'YYYYMMDD' format or None.
    """
    try:
        if not symbol:
            base_currency = input("Enter the base currency (Default: USDT): ").strip().upper() or 'USDT'
            quote_currency = input("Enter the quote currency (Default: BTC): ").strip().upper() or 'BTC'
            symbol = f"{quote_currency}{base_currency}"
        symbol = symbol.upper()

        if not interval:
            intervals = ['1m','3m','5m','15m','30m','1h','2h','4h','6h','8h','12h','1d','3d','1w','1M']
            print("Available intervals:")
            for idx, intrvl in enumerate(intervals, 1):
                print(f"{idx}. {intrvl}")
            interval_choice = input("Select interval by number (Default: 12 for '1d'): ").strip()
            if interval_choice.isdigit() and 1 <= int(interval_choice) <= len(intervals):
                interval = intervals[int(interval_choice)-1]
            else:
                interval = '1d'
        logger.info(f"Selected symbol: {symbol}, interval: {interval}")

        if not start_date_str:
            start_date_str = input("Enter start date (YYYYMMDD) or blank: ").strip()
        if not end_date_str:
            end_date_str = input("Enter end date (YYYYMMDD) or blank: ").strip()

        start_time = date_to_milliseconds(start_date_str) if start_date_str else get_earliest_valid_timestamp(symbol, interval)
        end_time = date_to_milliseconds(end_date_str) if end_date_str else get_current_timestamp()

        if start_time is None:
            logger.error("Invalid start date format.")
            sys.exit(1)
        if end_time is None:
            logger.error("Failed to retrieve current timestamp from Binance.")
            sys.exit(1)

        klines = get_historical_klines(symbol, interval, start_time, end_time)
        if not klines:
            logger.error("No klines data retrieved.")
            sys.exit(1)

        df = process_klines_to_dataframe(klines)
        os.makedirs('csv', exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        csv_filename = f"csv/{symbol}_{interval}_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        logger.info(f"Saved CSV data to {csv_filename}")

        save_dataframe_to_sqlite(df, DB_PATH, symbol, interval)
        logger.info(f"Saved data to SQLite database at {DB_PATH}")

    except Exception as e:
        logger.exception(f"An error occurred in download_binance_data: {e}")
        sys.exit(1)

def date_to_milliseconds(date_str):
    """
    Converts a date string in 'YYYYMMDD' format to milliseconds since epoch.
    """
    try:
        date_obj = datetime.datetime.strptime(date_str, '%Y%m%d')
        return int(date_obj.timestamp() * 1000)
    except ValueError as ve:
        logger.error(f"Date format error: {ve}")
        return None

def get_earliest_valid_timestamp(symbol, interval):
    """
    Retrieves the earliest valid timestamp for the given symbol and interval.
    """
    try:
        url = f"{BASE_URL}/api/v3/klines"
        params = {'symbol': symbol, 'interval': interval, 'limit': 1, 'startTime': 0}
        response = requests.get(url, params=params)
        if response.status_code != 200:
            logger.error(f"Failed to get earliest timestamp: {response.status_code} - {response.text}")
            return None
        data = response.json()
        if not data:
            logger.error("No data returned when fetching earliest timestamp.")
            return None
        return data[0][0]
    except Exception as e:
        logger.exception(f"Error fetching earliest timestamp: {e}")
        return None

def get_current_timestamp():
    """
    Retrieves the current server time from Binance in milliseconds.
    """
    try:
        url = f"{BASE_URL}/api/v3/time"
        response = requests.get(url)
        if response.status_code != 200:
            logger.error(f"Failed to get current timestamp: {response.status_code} - {response.text}")
            return None
        data = response.json()
        return data['serverTime']
    except Exception as e:
        logger.exception(f"Error fetching current timestamp: {e}")
        return None

def process_klines_to_dataframe(klines):
    """
    Processes raw klines data into a pandas DataFrame.
    """
    try:
        columns = [
            'open_time','open','high','low','close','volume',
            'close_time','quote_asset_volume','number_of_trades',
            'taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'
        ]
        df = pd.DataFrame(klines, columns=columns).drop('ignore', axis=1)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        numeric_cols = [
            'open','high','low','close','volume',
            'quote_asset_volume','taker_buy_base_asset_volume','taker_buy_quote_asset_volume'
        ]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df['number_of_trades'] = df['number_of_trades'].astype(int)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logger.exception(f"Error processing klines to DataFrame: {e}")
        return pd.DataFrame()

def save_dataframe_to_sqlite(df, db_path, symbol, timeframe):
    """
    Saves the DataFrame to a SQLite database.
    """
    try:
        conn = create_connection(db_path)
        if conn is None:
            logger.error("Failed to create database connection.")
            return
        create_tables(conn)
        save_to_sqlite(df, db_path, symbol, timeframe)
        conn.close()
    except Exception as e:
        logger.exception(f"Error saving DataFrame to SQLite: {e}")
