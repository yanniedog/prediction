# binance_historical_data_downloader.py
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
    klines = []
    while True:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000
        }
        response = requests.get(f"{BASE_URL}/api/v3/klines", params=params)
        if response.status_code != 200:
            print(f"Error fetching klines: {response.text}")
            break
        data = response.json()
        if not data:
            print("No more klines to fetch.")
            break
        klines.extend(data)
        start_time = data[-1][0] + 1
        if len(data) < 1000:
            break
        time.sleep(0.5)  # To respect API rate limits
    return klines

def download_binance_data(symbol=None, interval=None, db_path=DB_PATH):
    print("Starting Binance data download.")
    if not symbol:
        base = input("Base currency (Default: USDT): ").strip().upper() or 'USDT'
        quote = input("Quote currency (Default: BTC): ").strip().upper() or 'BTC'
        symbol = f"{quote}{base}"
    symbol = symbol.upper()
    if not interval:
        intervals = ['1m','3m','5m','15m','30m','1h','2h','4h','6h','8h','12h','1d','3d','1w','1M']
        print("Available Intervals:")
        for i, iv in enumerate(intervals, 1):
            print(f"{i}. {iv}")
        choice = input("Select interval (Default: 12 for '1d'): ").strip()
        interval = intervals[int(choice)-1] if choice.isdigit() and 1 <= int(choice) <= len(intervals) else '1d'
    start_str = input("Start date (YYYYMMDD) or blank for earliest: ").strip()
    end_str = input("End date (YYYYMMDD) or blank for latest: ").strip()
    start_time = date_to_milliseconds(start_str) if start_str else get_earliest_timestamp(symbol, interval)
    end_time = date_to_milliseconds(end_str) if end_str else get_current_timestamp()
    print(f"Fetching {symbol} {interval} from {start_time} to {end_time}")
    klines = get_historical_klines(symbol, interval, start_time, end_time)
    print(f"Fetched {len(klines)} klines.")
    if not klines:
        print("No data fetched.")
        sys.exit(1)
    df = process_klines(klines)
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    os.makedirs('csv', exist_ok=True)
    csv_file = f"csv/{symbol}_{interval}_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    print(f"Saved to CSV: {csv_file}")
    save_to_sqlite(df, db_path, symbol, interval)
    print(f"Inserted {len(df)} records into DB at {db_path}.")

def date_to_milliseconds(date_str):
    try:
        return int(datetime.datetime.strptime(date_str, '%Y%m%d').timestamp() * 1000)
    except ValueError:
        print(f"Invalid date format: {date_str}. Expected YYYYMMDD.")
        return None

def get_earliest_timestamp(symbol, interval):
    print(f"Fetching earliest timestamp for {symbol} {interval}")
    params = {'symbol': symbol, 'interval': interval, 'limit':1, 'startTime':0}
    response = requests.get(f"{BASE_URL}/api/v3/klines", params=params)
    if response.status_code != 200:
        print(f"Error: {response.text}")
        sys.exit(1)
    data = response.json()
    if not data:
        print("No data returned.")
        sys.exit(1)
    return data[0][0]

def get_current_timestamp():
    print("Fetching current server time.")
    response = requests.get(f"{BASE_URL}/api/v3/time")
    if response.status_code != 200:
        print(f"Error: {response.text}")
        sys.exit(1)
    return response.json()['serverTime']

def process_klines(klines):
    cols = ['open_time','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore']
    df = pd.DataFrame(klines, columns=cols).drop('ignore', axis=1)
    df[['open_time','close_time']] = pd.to_datetime(df[['open_time','close_time']], unit='ms').dt.tz_localize(None)
    num_cols = ['open','high','low','close','volume','quote_asset_volume','taker_buy_base_asset_volume','taker_buy_quote_asset_volume']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    df['number_of_trades'] = df['number_of_trades'].astype(int)
    return df

if __name__ == "__main__":
    download_binance_data()
