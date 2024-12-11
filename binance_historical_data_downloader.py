# binance_historical_data_downloader.py

import os
import sys
import requests
import pandas as pd
import time
import datetime
import logging
from logging.handlers import RotatingFileHandler

# Import the database path from config.py
from config import DB_PATH

# Import the save_to_sqlite, create_connection, and create_tables functions
from sqlite_data_manager import save_to_sqlite, create_connection, create_tables

# Configure logging
def setup_logging():
    logger = logging.getLogger('binance_historical_data_downloader')
    logger.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler with rotation
    file_handler = RotatingFileHandler('binance_downloader.log', maxBytes=5*1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Initialize logger
logger = setup_logging()

# Binance API base URL
BASE_URL = 'https://api.binance.com'

def get_historical_klines(symbol, interval, start_time, end_time):
    """
    Fetches historical klines from Binance API.

    Args:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
        interval (str): Kline interval (e.g., '1d').
        start_time (int): Start time in milliseconds.
        end_time (int): End time in milliseconds.

    Returns:
        list: List of klines data.
    """
    limit = 1000  # Maximum limit per request
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
        response = requests.get(url, params=params)
        if response.status_code != 200:
            logger.error(f"Error fetching klines: {response.text}")
            break
        data = response.json()
        if not data:
            logger.info("No more klines data to fetch.")
            break
        klines.extend(data)
        start_time = data[-1][0] + 1  # Move to next timestamp
        if len(data) < limit:
            # All data fetched
            break
        time.sleep(0.5)  # To avoid rate limits
        logger.debug(f"Fetched {len(data)} klines in current batch.")
        logger.debug(f"Total klines fetched so far: {len(klines)}")
    return klines

def download_binance_data(symbol=None, interval=None):
    """
    Orchestrates the download of Binance historical data.

    Args:
        symbol (str, optional): Trading pair symbol (e.g., 'BTCUSDT'). If None, prompts the user.
        interval (str, optional): Kline interval (e.g., '1d'). If None, prompts the user.
    """
    logger.info("Starting Binance data download process.")

    # Prompt user for inputs if not provided
    if not symbol:
        base_currency = input("Enter the base currency (Default: USDT): ").strip().upper()
        if not base_currency:
            base_currency = 'USDT'
            logger.debug("Base currency not provided. Defaulting to USDT.")
        quote_currency = input("Enter the quote currency (Default: BTC): ").strip().upper()
        if not quote_currency:
            quote_currency = 'BTC'
            logger.debug("Quote currency not provided. Defaulting to BTC.")
        symbol = f"{quote_currency}{base_currency}"
        logger.debug(f"Constructed symbol: {symbol}")

    if not interval:
        # Available intervals
        intervals = [
            '1m', '3m', '5m', '15m', '30m',
            '1h', '2h', '4h', '6h', '8h', '12h',
            '1d', '3d', '1w', '1M'
        ]
        print("Available intervals:")
        for idx, interval_option in enumerate(intervals, start=1):
            print(f"{idx}. {interval_option}")
        interval_choice = input("Select an interval by number (Default: 12 for '1d'): ").strip()
        if not interval_choice.isdigit() or int(interval_choice) < 1 or int(interval_choice) > len(intervals):
            interval = '1d'
            logger.debug("Interval not selected. Defaulting to '1d'.")
        else:
            interval = intervals[int(interval_choice) - 1]
        logger.debug(f"Selected interval: {interval}")

    # Start and end dates
    start_date_str = input("Enter the start date (YYYYMMDD) or leave blank for earliest available: ").strip()
    end_date_str = input("Enter the end date (YYYYMMDD) or leave blank for latest available: ").strip()

    # Convert date strings to milliseconds
    start_time = date_to_milliseconds(start_date_str) if start_date_str else None
    end_time = date_to_milliseconds(end_date_str) if end_date_str else None

    # Get earliest and latest timestamps if not provided
    if not start_time:
        start_time = get_earliest_valid_timestamp(symbol, interval)
    if not end_time:
        end_time = get_current_timestamp()
    logger.debug(f"Start time: {start_time}, End time: {end_time}")

    logger.info(f"Fetching data for symbol: {symbol}, Interval: {interval}, Start Time: {start_time}, End Time: {end_time}")
    klines = get_historical_klines(symbol, interval, start_time, end_time)
    logger.info(f"Fetched {len(klines)} klines from Binance API.")

    if not klines:
        logger.error("No klines data fetched. Exiting.")
        print("No data fetched from Binance API.")
        sys.exit(1)

    # Process klines into DataFrame
    df = process_klines_to_dataframe(klines)
    logger.info("Klines data successfully processed into DataFrame.")

    # Save DataFrame to CSV
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    csv_filename = f"csv/{symbol}_{interval}_{timestamp}.csv"
    os.makedirs('csv', exist_ok=True)
    df.to_csv(csv_filename, index=False)
    logger.info(f"Data successfully saved to CSV file: {csv_filename}")

    # Save DataFrame to SQLite database
    save_dataframe_to_sqlite(df, DB_PATH, symbol, interval)
    logger.info(f"Inserted {len(df)} records into the SQLite database at {DB_PATH}.")

    print("Binance data download and storage complete.")
    logger.info("Binance data download and storage process completed successfully.")

def date_to_milliseconds(date_str):
    """
    Converts date string to milliseconds.

    Args:
        date_str (str): Date string in 'YYYYMMDD' format.

    Returns:
        int: Time in milliseconds since epoch.
    """
    try:
        date_obj = datetime.datetime.strptime(date_str, '%Y%m%d')
        return int(date_obj.timestamp() * 1000)
    except ValueError:
        logger.error(f"Invalid date format: {date_str}. Expected 'YYYYMMDD'.")
        return None

def get_earliest_valid_timestamp(symbol, interval):
    """
    Gets the earliest valid timestamp for the given symbol and interval.

    Args:
        symbol (str): Trading pair symbol.
        interval (str): Kline interval.

    Returns:
        int: Earliest timestamp in milliseconds.
    """
    logger.info(f"Fetching earliest timestamp for symbol: {symbol}, interval: {interval}")
    url = f"{BASE_URL}/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': 1,
        'startTime': 0
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        logger.error(f"Error fetching earliest timestamp: {response.text}")
        sys.exit(1)
    data = response.json()
    if not data:
        logger.error("No data returned when fetching earliest timestamp.")
        sys.exit(1)
    earliest_timestamp = data[0][0]
    logger.debug(f"Earliest timestamp fetched: {earliest_timestamp}")
    return earliest_timestamp

def get_current_timestamp():
    """
    Gets the current timestamp from Binance server.

    Returns:
        int: Current timestamp in milliseconds.
    """
    logger.info("Fetching latest timestamp from Binance server.")
    url = f"{BASE_URL}/api/v3/time"
    response = requests.get(url)
    if response.status_code != 200:
        logger.error(f"Error fetching server time: {response.text}")
        sys.exit(1)
    data = response.json()
    current_timestamp = data['serverTime']
    logger.debug(f"Latest timestamp fetched: {current_timestamp}")
    return current_timestamp

def process_klines_to_dataframe(klines):
    """
    Processes klines data into a pandas DataFrame.

    Args:
        klines (list): List of klines data.

    Returns:
        pd.DataFrame: DataFrame containing kline data.
    """
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]
    df = pd.DataFrame(klines, columns=columns)
    logger.debug(f"Initial DataFrame shape: {df.shape}")

    # Drop 'ignore' column
    df.drop('ignore', axis=1, inplace=True)
    logger.debug("Dropped 'ignore' column from DataFrame.")

    # Convert timestamps to datetime
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    logger.debug("Converted 'open_time' and 'close_time' to datetime.")

    # Remove timezone information
    df['open_time'] = df['open_time'].dt.tz_localize(None)
    df['close_time'] = df['close_time'].dt.tz_localize(None)
    logger.debug("Removed timezone information from 'open_time' and 'close_time'.")

    # Convert numeric columns
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df['number_of_trades'] = df['number_of_trades'].astype(int)
    logger.debug("Converted numeric columns to appropriate data types.")

    return df

def save_dataframe_to_sqlite(df, db_path, symbol, timeframe):
    """
    Saves DataFrame to SQLite database.

    Args:
        df (pd.DataFrame): DataFrame to save.
        db_path (str): Path to SQLite database.
        symbol (str): Trading symbol.
        timeframe (str): Timeframe of the data.
    """
    try:
        # Ensure the database and tables are initialized
        conn = create_connection(db_path)
        if conn:
            create_tables(conn)
            conn.close()
        else:
            logger.error(f"Cannot connect to the database at {db_path}.")
            sys.exit(1)
    except Exception as e:
        logger.exception(f"Error during database initialization: {e}")
        sys.exit(1)

    # Now, use the save_to_sqlite function
    save_to_sqlite(df, db_path, symbol, timeframe)

if __name__ == "__main__":
    download_binance_data()