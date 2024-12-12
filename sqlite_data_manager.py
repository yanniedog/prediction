import sqlite3
from pathlib import Path
import pandas as pd

def create_connection(db_file):
    try:
        return sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(f"SQLite connection error: {e}")
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
                    FOREIGN KEY (indicator_id) REFERENCES indicators(id),
                    UNIQUE(symbol_id, timeframe_id, indicator_id, lag)
                );"""
        }
        for table, query in tables.items():
            cursor.execute(query)

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_open_time ON klines (open_time);",
            "CREATE INDEX IF NOT EXISTS idx_close_time ON klines (close_time);",
            "CREATE INDEX IF NOT EXISTS idx_correlation_computed ON klines (correlation_computed);",
            "CREATE INDEX IF NOT EXISTS idx_correlations ON correlations (symbol_id, timeframe_id, indicator_id, lag);"
        ]
        for idx_query in indexes:
            cursor.execute(idx_query)

        conn.commit()
    except sqlite3.Error as e:
        print(f"SQLite table creation error: {e}")

def insert_indicator_configs(conn, indicator_name, configs):
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO indicators (name) VALUES (?)", (indicator_name,))
        cursor.execute("SELECT id FROM indicators WHERE name = ?", (indicator_name,))
        indicator_id = cursor.fetchone()[0]

        for config in configs:
            config_name = f"{indicator_name}_{'_'.join([f'{k}{v}' for k, v in config.items()])}"
            cursor.execute("INSERT OR IGNORE INTO indicators (name) VALUES (?)", (config_name,))

        conn.commit()
    except sqlite3.Error as e:
        print(f"SQLite insertion error: {e}")

def insert_klines(conn, df, symbol, timeframe):
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO symbols (symbol) VALUES (?)", (symbol,))
        cursor.execute("SELECT id FROM symbols WHERE symbol = ?", (symbol,))
        symbol_id = cursor.fetchone()[0]

        cursor.execute("INSERT OR IGNORE INTO timeframes (timeframe) VALUES (?)", (timeframe,))
        cursor.execute("SELECT id FROM timeframes WHERE timeframe = ?", (timeframe,))
        timeframe_id = cursor.fetchone()[0]

        df['open_time'] = df['open_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['close_time'] = df['close_time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        insert_query = """
            INSERT INTO klines (
                symbol_id, timeframe_id, open_time, open, high, low, close, volume, close_time,
                quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        data = [
            (
                symbol_id, timeframe_id,
                row['open_time'], row['open'], row['high'], row['low'], row['close'], row['volume'], row['close_time'],
                row['quote_asset_volume'], row['number_of_trades'],
                row['taker_buy_base_asset_volume'], row['taker_buy_quote_asset_volume']
            )
            for _, row in df.iterrows()
        ]

        cursor.executemany(insert_query, data)
        conn.commit()
        print(f"Successfully inserted {len(data)} records into klines.")
    except sqlite3.Error as e:
        print(f"SQLite insertion error: {e}")

def fetch_correlations(conn, symbol, timeframe, indicator_name):
    try:
        cursor = conn.cursor()
        query = """
            SELECT lag, correlation_value
            FROM correlations
            JOIN symbols ON correlations.symbol_id = symbols.id
            JOIN timeframes ON correlations.timeframe_id = timeframes.id
            JOIN indicators ON correlations.indicator_id = indicators.id
            WHERE symbols.symbol = ? AND timeframes.timeframe = ? AND indicators.name = ?
        """
        cursor.execute(query, (symbol, timeframe, indicator_name))
        return cursor.fetchall()
    except sqlite3.Error as e:
        print(f"SQLite fetch error: {e}")
        return []

def main():
    db_path = "database.db"
    conn = create_connection(db_path)
    if conn:
        create_tables(conn)
        conn.close()

if __name__ == "__main__":
    main()
