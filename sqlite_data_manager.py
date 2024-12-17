# sqlite_data_manager.py
import os
import sys
import sqlite3
import pandas as pd
from pathlib import Path
from config import DB_PATH

def create_connection(db_file):
    conn=None
    try:
        conn=sqlite3.connect(db_file)
    except:
        pass
    return conn

def create_tables(conn):
    try:
        cursor=conn.cursor()
        create_symbols_table="""
        CREATE TABLE IF NOT EXISTS symbols (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE
        );
        """
        cursor.execute(create_symbols_table)
        create_timeframes_table="""
        CREATE TABLE IF NOT EXISTS timeframes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timeframe TEXT UNIQUE
        );
        """
        cursor.execute(create_timeframes_table)
        create_indicators_table="""
        CREATE TABLE IF NOT EXISTS indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        );
        """
        cursor.execute(create_indicators_table)
        create_klines_table="""
        CREATE TABLE IF NOT EXISTS klines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol_id INTEGER,
            timeframe_id INTEGER,
            open_time TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            close_time TEXT,
            quote_asset_volume REAL,
            number_of_trades INTEGER,
            taker_buy_base_asset_volume REAL,
            taker_buy_quote_asset_volume REAL,
            correlation_computed BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (symbol_id) REFERENCES symbols(id),
            FOREIGN KEY (timeframe_id) REFERENCES timeframes(id)
        );
        """
        cursor.execute(create_klines_table)
        create_correlations_table="""
        CREATE TABLE IF NOT EXISTS correlations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol_id INTEGER,
            timeframe_id INTEGER,
            indicator_id INTEGER,
            lag INTEGER,
            correlation_value REAL,
            FOREIGN KEY (symbol_id) REFERENCES symbols(id),
            FOREIGN KEY (timeframe_id) REFERENCES timeframes(id),
            FOREIGN KEY (indicator_id) REFERENCES indicators(id)
        );
        """
        cursor.execute(create_correlations_table)
        indexes=[
            "CREATE INDEX IF NOT EXISTS idx_open_time ON klines (open_time);",
            "CREATE INDEX IF NOT EXISTS idx_close_time ON klines (close_time);",
            "CREATE INDEX IF NOT EXISTS idx_correlation_computed ON klines (correlation_computed);",
            "CREATE INDEX IF NOT EXISTS idx_correlations_symbol_timeframe_indicator_lag ON correlations (symbol_id, timeframe_id, indicator_id, lag);"
        ]
        for index_query in indexes:
            cursor.execute(index_query)
        conn.commit()
    except:
        pass

def save_to_sqlite(df,db_path,symbol,timeframe):
    conn=None
    try:
        os.makedirs(os.path.dirname(db_path),exist_ok=True)
        conn=sqlite3.connect(db_path)
        cursor=conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO symbols (symbol) VALUES (?)",(symbol,))
        cursor.execute("SELECT id FROM symbols WHERE symbol = ?",(symbol,))
        symbol_id=cursor.fetchone()[0]
        cursor.execute("INSERT OR IGNORE INTO timeframes (timeframe) VALUES (?)",(timeframe,))
        cursor.execute("SELECT id FROM timeframes WHERE timeframe = ?",(timeframe,))
        timeframe_id=cursor.fetchone()[0]
        df['open_time']=pd.to_datetime(df['open_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df['close_time']=pd.to_datetime(df['close_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        insert_query="""
            INSERT INTO klines (
                symbol_id, timeframe_id, open_time, open, high, low, close, volume, close_time,
                quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        for index,row in df.iterrows():
            cursor.execute(insert_query,(symbol_id,timeframe_id,row['open_time'],row['open'],row['high'],row['low'],row['close'],row['volume'],row['close_time'],row['quote_asset_volume'],row['number_of_trades'],row['taker_buy_base_asset_volume'],row['taker_buy_quote_asset_volume']))
        conn.commit()
    except:
        pass
    finally:
        if conn:
            conn.close()