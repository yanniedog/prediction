# load_data.py

import os
import sys
import sqlite3
import logging
import pandas as pd
from dateutil import parser
import numpy as np

from config import DB_PATH

from sqlite_data_manager import create_connection, create_tables

def load_data(symbol, timeframe):
    """
    Loads and preprocesses data from the SQLite database for a specific symbol and timeframe.

    Args:
        symbol (str): The trading symbol (e.g., 'BTCUSDT').
        timeframe (str): The timeframe (e.g., '1d').

    Returns:
        tuple: A tuple containing the processed DataFrame, a boolean indicating if the data is in reverse chronological order, and the database filename.
    """
    logging.basicConfig(level=logging.INFO)

    db_path = DB_PATH

    if not os.path.exists(db_path):
        logging.warning(f"SQLite database '{db_path}' not found. Creating a new database with all required tables.")
        conn = create_connection(db_path)
        if conn:
            create_tables(conn)
            conn.close()
        else:
            logging.error(f"Failed to create connection to '{db_path}'. Exiting.")
            sys.exit(1)

    try:
        conn = sqlite3.connect(db_path)
        logging.info(f"Connected to SQLite database at: {db_path}")
    except sqlite3.Error as e:
        logging.error(f"Failed to connect to database '{db_path}'. Error: {e}")
        sys.exit(1)

    query = """
    SELECT klines.* FROM klines
    JOIN symbols ON klines.symbol_id = symbols.id
    JOIN timeframes ON klines.timeframe_id = timeframes.id
    WHERE symbols.symbol = ? AND timeframes.timeframe = ?
    ORDER BY open_time ASC
    """

    try:
        data = pd.read_sql_query(query, conn, params=(symbol, timeframe), parse_dates=['open_time', 'close_time'])
        logging.info("Data loaded from SQLite database.")
    except pd.io.sql.DatabaseError as e:
        logging.error(f"Failed to execute query. Error: {e}")
        conn.close()
        sys.exit(1)

    conn.close()

    if data.empty:
        logging.warning("No data available in the database.")
        is_reverse_chronological = False
        return data, is_reverse_chronological, os.path.basename(db_path)
    else:
        time_column = 'open_time'
        is_reverse_chronological = data[time_column].is_monotonic_decreasing

        if is_reverse_chronological:
            data = data.sort_values(time_column)
            data.reset_index(drop=True, inplace=True)
            logging.info("Data sorted in chronological order.")

        data.dropna(inplace=True)

        logging.info("Columns in the data: %s", list(data.columns))
        logging.info("First few rows of the data:\n%s", data.head())

        data['open_time'] = data['open_time'].dt.tz_localize(None)
        data['close_time'] = data['close_time'].dt.tz_localize(None)
        logging.info("Removed timezone information from 'open_time' and 'close_time'.")

        if not pd.api.types.is_datetime64_any_dtype(data['open_time']):
            data['open_time'] = pd.to_datetime(data['open_time'], errors='coerce')

        try:
            data['TimeDiff'] = data['open_time'].diff().dt.total_seconds()
            if data['TimeDiff'].isna().all():
                logging.error("Failed to determine time interval: All computed time differences are NaN.")
                sys.exit(1)
            else:
                logging.info("First few time differences (in seconds):\n%s", data['TimeDiff'].head())
                data['TimeDiff'].fillna(0, inplace=True)
        except Exception as e:
            logging.error(f"Failed to compute time differences: {e}")
            sys.exit(1)

    return data, is_reverse_chronological, os.path.basename(db_path)

if __name__ == "__main__":
    symbol = input("Enter the symbol (e.g., 'BTCUSDT'): ").strip().upper()
    timeframe = input("Enter the timeframe (e.g., '1d'): ").strip()
    df, is_rev, filename = load_data(symbol, timeframe)
