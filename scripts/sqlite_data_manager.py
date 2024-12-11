# sqlite_data_manager.py
import os
import sqlite3
import pandas as pd
from config import DB_PATH
def create_connection(db_file):
    try:
        return sqlite3.connect(db_file)
    except:
        return None
def create_tables(conn):
    c=conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS symbols (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT UNIQUE);")
    c.execute("CREATE TABLE IF NOT EXISTS timeframes (id INTEGER PRIMARY KEY AUTOINCREMENT, timeframe TEXT UNIQUE);")
    c.execute("CREATE TABLE IF NOT EXISTS indicators (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE);")
    c.execute("""CREATE TABLE IF NOT EXISTS klines (
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
    );""")
    c.execute("""CREATE TABLE IF NOT EXISTS correlations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol_id INTEGER,
        timeframe_id INTEGER,
        indicator_id INTEGER,
        lag INTEGER,
        correlation_value REAL,
        FOREIGN KEY (symbol_id) REFERENCES symbols(id),
        FOREIGN KEY (timeframe_id) REFERENCES timeframes(id),
        FOREIGN KEY (indicator_id) REFERENCES indicators(id)
    );""")
    indexes=[
        "CREATE INDEX IF NOT EXISTS idx_open_time ON klines (open_time);",
        "CREATE INDEX IF NOT EXISTS idx_close_time ON klines (close_time);",
        "CREATE INDEX IF NOT EXISTS idx_correlation_computed ON klines (correlation_computed);",
        "CREATE INDEX IF NOT EXISTS idx_correlations_symbol_timeframe_indicator_lag ON correlations (symbol_id, timeframe_id, indicator_id, lag);"
    ]
    for q in indexes:
        c.execute(q)
    conn.commit()
def save_to_sqlite(df,db_path,symbol,timeframe):
    os.makedirs(os.path.dirname(db_path),exist_ok=True)
    conn=create_connection(db_path)
    if not conn:
        return
    c=conn.cursor()
    c.execute("INSERT OR IGNORE INTO symbols (symbol) VALUES (?)",(symbol,))
    c.execute("SELECT id FROM symbols WHERE symbol = ?",(symbol,))
    symbol_id=c.fetchone()[0]
    c.execute("INSERT OR IGNORE INTO timeframes (timeframe) VALUES (?)",(timeframe,))
    c.execute("SELECT id FROM timeframes WHERE timeframe = ?",(timeframe,))
    timeframe_id=c.fetchone()[0]
    df['open_time']=pd.to_datetime(df['open_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df['close_time']=pd.to_datetime(df['close_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    insert_query="""
    INSERT INTO klines (symbol_id,timeframe_id,open_time,open,high,low,close,volume,close_time,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
    for _,row in df.iterrows():
        c.execute(insert_query,(symbol_id,timeframe_id,row['open_time'],row['open'],row['high'],row['low'],row['close'],row['volume'],row['close_time'],row['quote_asset_volume'],row['number_of_trades'],row['taker_buy_base_asset_volume'],row['taker_buy_quote_asset_volume']))
    conn.commit()
    conn.close()
