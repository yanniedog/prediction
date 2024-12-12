import os, sqlite3, pandas as pd
from pathlib import Path
from config import DB_PATH

def create_connection(db_file):
    try:
        return sqlite3.connect(db_file)
    except:
        return None

def create_tables(conn):
    try:
        cursor = conn.cursor()
        tables = {
            'symbols': """
                CREATE TABLE IF NOT EXISTS symbols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE
                );""",
            'timeframes': """
                CREATE TABLE IF NOT EXISTS timeframes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timeframe TEXT UNIQUE
                );""",
            'indicators': """
                CREATE TABLE IF NOT EXISTS indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE
                );""",
            'klines': """
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
                );""",
            'correlations': """
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
                );"""
        }
        for table, q in tables.items():
            cursor.execute(q)
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_open_time ON klines (open_time);",
            "CREATE INDEX IF NOT EXISTS idx_close_time ON klines (close_time);",
            "CREATE INDEX IF NOT EXISTS idx_correlation_computed ON klines (correlation_computed);",
            "CREATE INDEX IF NOT EXISTS idx_correlations ON correlations (symbol_id, timeframe_id, indicator_id, lag);"
        ]
        for idx_q in indexes: cursor.execute(idx_q)
        conn.commit()
    except:
        pass

def save_to_sqlite(df, db_path, symbol, timeframe):
    try:
        os.makedirs(Path(db_path).parent, exist_ok=True)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO symbols (symbol) VALUES (?)", (symbol,))
        cursor.execute("SELECT id FROM symbols WHERE symbol = ?", (symbol,))
        sym_id = cursor.fetchone()[0]
        cursor.execute("INSERT OR IGNORE INTO timeframes (timeframe) VALUES (?)", (timeframe,))
        cursor.execute("SELECT id FROM timeframes WHERE timeframe = ?", (timeframe,))
        tf_id = cursor.fetchone()[0]
        df['open_time'] = pd.to_datetime(df['open_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df['close_time'] = pd.to_datetime(df['close_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        insert_q = """
            INSERT INTO klines (symbol_id, timeframe_id, open_time, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.executemany(insert_q, df[['symbol_id','timeframe_id','open_time','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume']].itertuples(index=False, name=None))
        conn.commit()
    except:
        pass
    finally:
        if conn: conn.close()
