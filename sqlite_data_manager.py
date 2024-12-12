import sqlite3, os
from pathlib import Path

def create_connection(db_file):
    try: return sqlite3.connect(db_file)
    except sqlite3.Error as e: print(f"SQLite connection error: {e}"); return None

def create_tables(conn):
    tables = {
        'symbols': "CREATE TABLE IF NOT EXISTS symbols (id INTEGER PRIMARY KEY, symbol TEXT UNIQUE);",
        'timeframes': "CREATE TABLE IF NOT EXISTS timeframes (id INTEGER PRIMARY KEY, timeframe TEXT UNIQUE);",
        'klines': """CREATE TABLE IF NOT EXISTS klines (
            id INTEGER PRIMARY KEY, symbol_id INTEGER, timeframe_id INTEGER, open_time TEXT, open REAL, high REAL, low REAL,
            close REAL, volume REAL, close_time TEXT, quote_asset_volume REAL, number_of_trades INTEGER,
            taker_buy_base_asset_volume REAL, taker_buy_quote_asset_volume REAL, FOREIGN KEY (symbol_id) REFERENCES symbols(id),
            FOREIGN KEY (timeframe_id) REFERENCES timeframes(id));"""
    }
    try:
        for q in tables.values(): conn.execute(q)
        conn.commit()
    except sqlite3.Error as e: print(f"SQLite table creation error: {e}")

def save_to_sqlite(df, db_path, symbol, timeframe):
    conn, sym_id, tf_id = sqlite3.connect(db_path), None, None
    try:
        conn.execute("INSERT OR IGNORE INTO symbols (symbol) VALUES (?)", (symbol,))
        sym_id = conn.execute("SELECT id FROM symbols WHERE symbol = ?", (symbol,)).fetchone()[0]
        conn.execute("INSERT OR IGNORE INTO timeframes (timeframe) VALUES (?)", (timeframe,))
        tf_id = conn.execute("SELECT id FROM timeframes WHERE timeframe = ?", (timeframe,)).fetchone()[0]
        df['open_time'], df['close_time'] = df['open_time'].dt.strftime('%Y-%m-%d %H:%M:%S'), df['close_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        conn.executemany("""INSERT INTO klines (symbol_id, timeframe_id, open_time, open, high, low, close, volume, close_time,
            quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", [(sym_id, tf_id, row['open_time'], row['open'], row['high'], row['low'], 
            row['close'], row['volume'], row['close_time'], row['quote_asset_volume'], row['number_of_trades'], 
            row['taker_buy_base_asset_volume'], row['taker_buy_quote_asset_volume']) for _, row in df.iterrows()])
        conn.commit()
    except Exception as e: print(f"Error inserting data: {e}")
    finally: conn.close()

def initialize_database(db_path):
    os.makedirs(Path(db_path).parent, exist_ok=True)
    conn = create_connection(db_path)
    if conn: create_tables(conn); conn.close()
