# sqlite_data_manager.py

import os
import sys
import sqlite3
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler

# Import the database path from config.py
from config import DB_PATH

# Configure logging
def setup_logging():
    logger = logging.getLogger('sqlite_data_manager')
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler with rotation
    file_handler = RotatingFileHandler('sqlite_data_manager.log', maxBytes=5*1024*1024, backupCount=5)  # 5MB per file
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Default console log level
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

setup_logging()
logger = logging.getLogger('sqlite_data_manager')

def create_connection(db_file):
    """
    Creates a connection to the SQLite database.

    Args:
        db_file (str): Path to the SQLite database file.

    Returns:
        sqlite3.Connection: SQLite connection object or None.
    """
    logger.info(f"Attempting to create a connection to the SQLite database at {db_file}.")
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        logger.info(f"Connected to SQLite database: {db_file}")
    except sqlite3.Error as e:
        logger.exception(f"Failed to connect to SQLite database: {e}")
    return conn

def create_tables(conn):
    """
    Creates the necessary tables in the SQLite database.

    Args:
        conn (sqlite3.Connection): SQLite connection object.
    """
    logger.info("Creating tables in the SQLite database.")
    try:
        cursor = conn.cursor()
        logger.debug("Obtained cursor from SQLite connection.")

        # Create the symbols table
        create_symbols_table = """
        CREATE TABLE IF NOT EXISTS symbols (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE
        );
        """
        cursor.execute(create_symbols_table)
        logger.info("Created 'symbols' table successfully.")

        # Create the timeframes table
        create_timeframes_table = """
        CREATE TABLE IF NOT EXISTS timeframes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timeframe TEXT UNIQUE
        );
        """
        cursor.execute(create_timeframes_table)
        logger.info("Created 'timeframes' table successfully.")

        # Create the indicators table
        create_indicators_table = """
        CREATE TABLE IF NOT EXISTS indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        );
        """
        cursor.execute(create_indicators_table)
        logger.info("Created 'indicators' table successfully.")

        # Create the klines table
        create_klines_table = """
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
        logger.info("Created 'klines' table successfully.")

        # Create the correlations table (modified)
        create_correlations_table = """
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
        logger.info("Created 'correlations' table successfully.")

        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_open_time ON klines (open_time);",
            "CREATE INDEX IF NOT EXISTS idx_close_time ON klines (close_time);",
            "CREATE INDEX IF NOT EXISTS idx_correlation_computed ON klines (correlation_computed);",
            "CREATE INDEX IF NOT EXISTS idx_correlations_symbol_timeframe_indicator_lag ON correlations (symbol_id, timeframe_id, indicator_id, lag);"
        ]
        for index_query in indexes:
            cursor.execute(index_query)
            logger.debug(f"Executed index creation query: {index_query.strip()}")

        conn.commit()
        logger.info("Tables and indexes created and committed successfully.")
    except sqlite3.Error as e:
        logger.exception(f"Failed to create tables: {e}")

def save_to_sqlite(df, db_path, symbol, timeframe):
    """
    Saves the DataFrame to an SQLite database.

    Args:
        df (pd.DataFrame): DataFrame to save.
        db_path (str): Path to the SQLite database file.
        symbol (str): Trading symbol.
        timeframe (str): Timeframe of the data.
    """
    logger.info(f"Saving DataFrame to SQLite database at {db_path}.")
    conn = None
    try:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        logger.debug("Connected to SQLite database.")

        # Insert symbol if not exists
        cursor.execute("INSERT OR IGNORE INTO symbols (symbol) VALUES (?)", (symbol,))
        cursor.execute("SELECT id FROM symbols WHERE symbol = ?", (symbol,))
        symbol_id = cursor.fetchone()[0]

        # Insert timeframe if not exists
        cursor.execute("INSERT OR IGNORE INTO timeframes (timeframe) VALUES (?)", (timeframe,))
        cursor.execute("SELECT id FROM timeframes WHERE timeframe = ?", (timeframe,))
        timeframe_id = cursor.fetchone()[0]

        # Ensure 'open_time' and 'close_time' are strings without timezone information
        df['open_time'] = pd.to_datetime(df['open_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df['close_time'] = pd.to_datetime(df['close_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        logger.debug("Converted 'open_time' and 'close_time' to string format without timezone.")

        # Insert data
        insert_query = """
            INSERT INTO klines (
                symbol_id, timeframe_id, open_time, open, high, low, close, volume, close_time,
                quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        for index, row in df.iterrows():
            cursor.execute(insert_query, (
                symbol_id,
                timeframe_id,
                row['open_time'],
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['volume'],
                row['close_time'],
                row['quote_asset_volume'],
                row['number_of_trades'],
                row['taker_buy_base_asset_volume'],
                row['taker_buy_quote_asset_volume']
            ))

        conn.commit()
        logger.info("Data successfully inserted into SQLite database.")
    except Exception as e:
        logger.exception(f"Failed to save DataFrame to SQLite: {e}")
        print(f"Failed to save data to SQLite: {e}")
    finally:
        if conn:
            conn.close()
            logger.debug("Closed SQLite database connection.")

if __name__ == "__main__":
    # Example usage
    db_path = DB_PATH  # Use the database path from config.py

    logger.info("Starting SQLite database setup.")
    conn = create_connection(db_path)
    if conn:
        create_tables(conn)
        # Here you can call save_to_sqlite() or other functions as needed
        conn.close()
        logger.info("SQLite database setup completed.")
    else:
        logger.error("Failed to establish a connection to the SQLite database. Exiting.")
        sys.exit(1)