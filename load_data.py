# filename: load_data.py
import os
import sys
import sqlite3
import pandas as pd
from dateutil import parser
import numpy as np
from config import DB_PATH
from sqlite_data_manager import create_connection,create_tables

def load_data(symbol,timeframe):
    db_path=DB_PATH
    if not os.path.exists(db_path):
        conn=create_connection(db_path)
        if conn:
            create_tables(conn)
            conn.close()
        else:
            sys.exit(1)
    try:
        conn=sqlite3.connect(db_path)
    except:
        sys.exit(1)
    query="""
    SELECT klines.* FROM klines
    JOIN symbols ON klines.symbol_id = symbols.id
    JOIN timeframes ON klines.timeframe_id = timeframes.id
    WHERE symbols.symbol = ? AND timeframes.timeframe = ?
    ORDER BY open_time ASC
    """
    try:
        data=pd.read_sql_query(query,conn,params=(symbol,timeframe),parse_dates=['open_time','close_time'])
    except:
        conn.close()
        sys.exit(1)
    conn.close()
    if data.empty:
        is_reverse_chronological=False
        return data,is_reverse_chronological,os.path.basename(db_path)
    else:
        time_column='open_time'
        is_reverse_chronological=data[time_column].is_monotonic_decreasing
        if is_reverse_chronological:
            data=data.sort_values(time_column)
            data.reset_index(drop=True,inplace=True)
        data.dropna(inplace=True)
        data['open_time']=data['open_time'].dt.tz_localize(None)
        data['close_time']=data['close_time'].dt.tz_localize(None)
        if not pd.api.types.is_datetime64_any_dtype(data['open_time']):
            data['open_time']=pd.to_datetime(data['open_time'],errors='coerce')
        data['TimeDiff']=data['open_time'].diff().dt.total_seconds()
        if data['TimeDiff'].isna().all():
            sys.exit(1)
    return data,is_reverse_chronological,os.path.basename(db_path)