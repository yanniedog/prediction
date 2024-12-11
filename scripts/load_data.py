# load_data.py
# load_data.py
# load_data.py
import os
import sys
import sqlite3
import logging
import pandas as pd
from config import DB_PATH
from sqlite_data_manager import create_connection, create_tables
def load_data(symbol, timeframe):
    logging.basicConfig(level=logging.INFO)
    db_path = DB_PATH
    if not os.path.exists(db_path):
        conn = create_connection(db_path)
        if conn:
            create_tables(conn)
            conn.close()
        else:
            sys.exit(1)
    try:
        conn = sqlite3.connect(db_path)
    except:
        sys.exit(1)
    query = """
    SELECT klines.* FROM klines
    JOIN symbols ON klines.symbol_id = symbols.id
    JOIN timeframes ON klines.timeframe_id = timeframes.id
    WHERE symbols.symbol = ? AND timeframes.timeframe = ?
    ORDER BY open_time ASC
    """
    try:
        data = pd.read_sql_query(query, conn, params=(symbol, timeframe), parse_dates=['open_time','close_time'])
    except:
        conn.close()
        sys.exit(1)
    conn.close()
    if data.empty:
        return data, False, os.path.basename(db_path)
    is_reverse_chronological = data['open_time'].is_monotonic_decreasing
    if is_reverse_chronological:
        data=data.sort_values('open_time').reset_index(drop=True)
    data.dropna(inplace=True)
    data['open_time']=data['open_time'].dt.tz_localize(None)
    data['close_time']=data['close_time'].dt.tz_localize(None)
    data['TimeDiff']=data['open_time'].diff().dt.total_seconds()
    data['TimeDiff'].fillna(0,inplace=True)
    return data, is_reverse_chronological, os.path.basename(db_path)
