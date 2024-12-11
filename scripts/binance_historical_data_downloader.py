import os
import sys
import requests
import pandas as pd
import time
import datetime
from config import DB_PATH
from sqlite_data_manager import save_to_sqlite, create_connection, create_tables

BASE_URL = 'https://api.binance.com'

def get_historical_klines(symbol, interval, start_time, end_time):
    limit = 1000
    klines = []
    while True:
        url = f"{BASE_URL}/api/v3/klines"
        params = {'symbol': symbol, 'interval': interval, 'startTime': start_time, 'endTime': end_time, 'limit': limit}
        response = requests.get(url, params=params)
        if response.status_code != 200:
            break
        data = response.json()
        if not data:
            break
        klines.extend(data)
        start_time = data[-1][0] + 1
        if len(data) < limit:
            break
        time.sleep(0.5)
    return klines

def download_binance_data(symbol=None, interval=None):
    if not symbol:
        base_currency = input("Enter the base currency (Default: USDT): ").strip().upper() or 'USDT'
        quote_currency = input("Enter the quote currency (Default: BTC): ").strip().upper() or 'BTC'
        symbol = f"{quote_currency}{base_currency}"
    symbol = symbol.upper()

    if not interval:
        intervals = ['1m','3m','5m','15m','30m','1h','2h','4h','6h','8h','12h','1d','3d','1w','1M']
        interval_choice = input("Select interval by number (Default: 12 for '1d'): ").strip()
        interval = intervals[int(interval_choice)-1] if interval_choice.isdigit() and 1 <= int(interval_choice) <= len(intervals) else '1d'

    start_date_str = input("Enter start date (YYYYMMDD) or blank: ").strip()
    end_date_str = input("Enter end date (YYYYMMDD) or blank: ").strip()
    start_time = date_to_milliseconds(start_date_str) if start_date_str else None
    end_time = date_to_milliseconds(end_date_str) if end_date_str else None
    if not start_time:
        start_time = get_earliest_valid_timestamp(symbol, interval)
    if not end_time:
        end_time = get_current_timestamp()
    klines = get_historical_klines(symbol, interval, start_time, end_time)
    if not klines:
        sys.exit(1)
    df = process_klines_to_dataframe(klines)
    os.makedirs('csv', exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    csv_filename = f"csv/{symbol}_{interval}_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    save_dataframe_to_sqlite(df, DB_PATH, symbol, interval)

def date_to_milliseconds(date_str):
    try:
        date_obj = datetime.datetime.strptime(date_str,'%Y%m%d')
        return int(date_obj.timestamp()*1000)
    except:
        return None

def get_earliest_valid_timestamp(symbol, interval):
    url = f"{BASE_URL}/api/v3/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': 1, 'startTime': 0}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        sys.exit(1)
    data = response.json()
    if not data:
        sys.exit(1)
    return data[0][0]

def get_current_timestamp():
    url = f"{BASE_URL}/api/v3/time"
    response = requests.get(url)
    if response.status_code != 200:
        sys.exit(1)
    data = response.json()
    return data['serverTime']

def process_klines_to_dataframe(klines):
    columns = ['open_time','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore']
    df = pd.DataFrame(klines, columns=columns).drop('ignore', axis=1)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    numeric_cols = ['open','high','low','close','volume','quote_asset_volume','taker_buy_base_asset_volume','taker_buy_quote_asset_volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df['number_of_trades'] = df['number_of_trades'].astype(int)
    return df

def save_dataframe_to_sqlite(df, db_path, symbol, timeframe):
    conn = create_connection(db_path)
    if conn:
        create_tables(conn)
        conn.close()
    save_to_sqlite(df, db_path, symbol, timeframe)
