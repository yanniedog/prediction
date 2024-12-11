# filename: binance_historical_data_downloader.py
import os
import sys
import requests
import pandas as pd
import time
import datetime
import logging
from config import DB_PATH
from sqlite_data_manager import save_to_sqlite,create_connection,create_tables
BASE_URL='https://api.binance.com'
def get_historical_klines(symbol,interval,start_time,end_time):
    limit=1000
    klines=[]
    while True:
        url=f"{BASE_URL}/api/v3/klines"
        params={'symbol':symbol,'interval':interval,'startTime':start_time,'endTime':end_time,'limit':limit}
        response=requests.get(url,params=params)
        if response.status_code!=200:
            logging.error(f"Error fetching klines: {response.text}")
            break
        data=response.json()
        if not data:
            logging.info("No more klines data to fetch.")
            break
        klines.extend(data)
        start_time=data[-1][0]+1
        if len(data)<limit:
            break
        time.sleep(0.5)
        logging.debug(f"Fetched {len(data)} klines in current batch.")
        logging.debug(f"Total klines fetched so far: {len(klines)}")
    return klines
def download_binance_data(symbol=None,interval=None):
    logging.info("Starting Binance data download process.")
    if not symbol:
        base_currency=input("Enter the base currency (Default: USDT): ").strip().upper()
        if not base_currency:
            base_currency='USDT'
            logging.debug("Base currency not provided. Defaulting to USDT.")
        quote_currency=input("Enter the quote currency (Default: BTC): ").strip().upper()
        if not quote_currency:
            quote_currency='BTC'
            logging.debug("Quote currency not provided. Defaulting to BTC.")
        symbol=f"{quote_currency}{base_currency}"
        logging.debug(f"Constructed symbol: {symbol}")
    if not interval:
        intervals=['1m','3m','5m','15m','30m','1h','2h','4h','6h','8h','12h','1d','3d','1w','1M']
        print("Available intervals:")
        for idx,interval_option in enumerate(intervals, start=1):
            print(f"{idx}. {interval_option}")
        interval_choice=input("Select an interval by number (Default: 12 for '1d'): ").strip()
        if not interval_choice.isdigit()or int(interval_choice)<1 or int(interval_choice)>len(intervals):
            interval='1d'
            logging.debug("Interval not selected. Defaulting to '1d'.")
        else:
            interval=intervals[int(interval_choice)-1]
        logging.debug(f"Selected interval: {interval}")
    start_date_str=input("Enter the start date (YYYYMMDD) or leave blank for earliest available: ").strip()
    end_date_str=input("Enter the end date (YYYYMMDD) or leave blank for latest available: ").strip()
    start_time=date_to_milliseconds(start_date_str)if start_date_str else None
    end_time=date_to_milliseconds(end_date_str)if end_date_str else None
    if not start_time:
        start_time=get_earliest_valid_timestamp(symbol,interval)
    if not end_time:
        end_time=get_current_timestamp()
    logging.debug(f"Start time: {start_time}, End time: {end_time}")
    logging.info(f"Fetching data for symbol: {symbol}, Interval: {interval}, Start Time: {start_time}, End Time: {end_time}")
    klines=get_historical_klines(symbol,interval,start_time,end_time)
    logging.info(f"Fetched {len(klines)} klines from Binance API.")
    if not klines:
        logging.error("No klines data fetched. Exiting.")
        print("No data fetched from Binance API.")
        sys.exit(1)
    df=process_klines_to_dataframe(klines)
    logging.info("Klines data successfully processed into DataFrame.")
    timestamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    csv_filename=f"csv/{symbol}_{interval}_{timestamp}.csv"
    os.makedirs('csv',exist_ok=True)
    df.to_csv(csv_filename,index=False)
    logging.info(f"Data successfully saved to CSV file: {csv_filename}")
    save_dataframe_to_sqlite(df,DB_PATH,symbol,interval)
    logging.info(f"Inserted {len(df)} records into the SQLite database at {DB_PATH}.")
    print("Binance data download and storage complete.")
    logging.info("Binance data download and storage process completed successfully.")
def date_to_milliseconds(date_str):
    if not date_str:
        return None
    try:
        date_obj=datetime.datetime.strptime(date_str,'%Y%m%d')
        return int(date_obj.timestamp()*1000)
    except ValueError:
        logging.error(f"Invalid date format: {date_str}. Expected 'YYYYMMDD'.")
        return None
def get_earliest_valid_timestamp(symbol,interval):
    logging.info(f"Fetching earliest timestamp for symbol: {symbol}, interval: {interval}")
    url=f"{BASE_URL}/api/v3/klines"
    params={'symbol':symbol,'interval':interval,'limit':1,'startTime':0}
    response=requests.get(url,params=params)
    if response.status_code!=200:
        logging.error(f"Error fetching earliest timestamp: {response.text}")
        sys.exit(1)
    data=response.json()
    if not data:
        logging.error("No data returned when fetching earliest timestamp.")
        sys.exit(1)
    earliest_timestamp=data[0][0]
    logging.debug(f"Earliest timestamp fetched: {earliest_timestamp}")
    return earliest_timestamp
def get_current_timestamp():
    logging.info("Fetching latest timestamp from Binance server.")
    url=f"{BASE_URL}/api/v3/time"
    response=requests.get(url)
    if response.status_code!=200:
        logging.error(f"Error fetching server time: {response.text}")
        sys.exit(1)
    data=response.json()
    current_timestamp=data['serverTime']
    logging.debug(f"Latest timestamp fetched: {current_timestamp}")
    return current_timestamp
def process_klines_to_dataframe(klines):
    columns=['open_time','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore']
    df=pd.DataFrame(klines,columns=columns)
    df.drop('ignore',axis=1,inplace=True)
    df['open_time']=pd.to_datetime(df['open_time'],unit='ms')
    df['close_time']=pd.to_datetime(df['close_time'],unit='ms')
    df['open_time']=df['open_time'].dt.tz_localize(None)
    df['close_time']=df['close_time'].dt.tz_localize(None)
    numeric_cols=['open','high','low','close','volume','quote_asset_volume','taker_buy_base_asset_volume','taker_buy_quote_asset_volume']
    df[numeric_cols]=df[numeric_cols].apply(pd.to_numeric,errors='coerce')
    df['number_of_trades']=df['number_of_trades'].astype(int)
    return df
def save_dataframe_to_sqlite(df,db_path,symbol,timeframe):
    try:
        conn=create_connection(db_path)
        if conn:
            create_tables(conn)
            conn.close()
        else:
            logging.error(f"Cannot connect to the database at {db_path}.")
            sys.exit(1)
    except Exception as e:
        logging.exception(f"Error during database initialization: {e}")
        sys.exit(1)
    save_to_sqlite(df,db_path,symbol,timeframe)