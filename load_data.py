# load_data.py
import logging
import os
import pandas as pd
from sqlite_data_manager import create_connection, initialize_database
from config import DB_PATH

logger = logging.getLogger()

def load_data(symbol, timeframe):
    initialize_database(DB_PATH)
    conn = create_connection(DB_PATH)
    query = """SELECT klines.* FROM klines
               JOIN symbols ON klines.symbol_id = symbols.id
               JOIN timeframes ON klines.timeframe_id = timeframes.id
               WHERE symbols.symbol = ? AND timeframes.timeframe = ?
               ORDER BY open_time ASC"""
    try:
        df = pd.read_sql_query(query, conn, params=(symbol, timeframe), parse_dates=['open_time', 'close_time'])
        conn.close()
        if df.empty:
            print("No data found for the specified symbol and timeframe.")
        is_rev = False
        db_fn = str(DB_PATH)
        return df, is_rev, db_fn
    except Exception as e:
        print(f"Error querying database: {e}")
        conn.close()
        return pd.DataFrame(), False, str(DB_PATH)

if __name__ == "__main__":
    symbol = input("Enter symbol (e.g., 'BTCUSDT'): ").strip().upper()
    timeframe = input("Enter timeframe (e.g., '1d'): ").strip()
    df, is_rev, db_fn = load_data(symbol, timeframe)
    if not df.empty:
        print("Data loaded successfully.")
        print(df.head())
    else:
        print("No data available.")
