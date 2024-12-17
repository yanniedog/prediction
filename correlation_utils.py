# correlation_utils.py

import os
import pandas as pd
import numpy as np
import sqlite3
from typing import List
from sklearn.preprocessing import StandardScaler
from config import DB_PATH

def calculate_correlation(data: pd.DataFrame, indicator_name: str, lag: int, is_reverse_chronological: bool) -> float:
    """
    Calculate the Pearson correlation between the specified indicator and the 'Close' price,
    shifted by the specified lag.

    Args:
        data (pd.DataFrame): The DataFrame containing indicator and price data.
        indicator_name (str): The name of the indicator column.
        lag (int): The lag period.
        is_reverse_chronological (bool): If True, indicates data is in reverse chronological order.

    Returns:
        float: The Pearson correlation coefficient.
    """
    if indicator_name.lower() == 'close':
        return np.nan

    try:
        shift_value = lag if is_reverse_chronological else -lag
        shifted_col = data[indicator_name].shift(shift_value)

        valid_data = pd.concat([shifted_col, data['Close']], axis=1).dropna()

        if not valid_data.empty:
            corr = valid_data[indicator_name].corr(valid_data['Close'])
            return corr
        else:
            return np.nan
    except:
        return np.nan

def is_valid_indicator(series: pd.Series) -> bool:
    """
    Validate if the indicator series is suitable for correlation calculation.

    Args:
        series (pd.Series): The indicator data series.

    Returns:
        bool: True if valid, False otherwise.
    """
    if series.isna().all():
        return False
    if series.nunique() <= 1:
        return False
    return True

def get_symbol_id(conn: sqlite3.Connection, symbol: str) -> int:
    """
    Retrieve the symbol ID from the database, inserting it if it doesn't exist.

    Args:
        conn (sqlite3.Connection): The SQLite connection object.
        symbol (str): The trading symbol.

    Returns:
        int: The symbol ID.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM symbols WHERE symbol = ?", (symbol,))
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        cursor.execute("INSERT INTO symbols (symbol) VALUES (?)", (symbol,))
        conn.commit()
        return cursor.lastrowid

def get_timeframe_id(conn: sqlite3.Connection, timeframe: str) -> int:
    """
    Retrieve the timeframe ID from the database, inserting it if it doesn't exist.

    Args:
        conn (sqlite3.Connection): The SQLite connection object.
        timeframe (str): The timeframe string.

    Returns:
        int: The timeframe ID.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM timeframes WHERE timeframe = ?", (timeframe,))
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        cursor.execute("INSERT INTO timeframes (timeframe) VALUES (?)", (timeframe,))
        conn.commit()
        return cursor.lastrowid

def get_indicator_id(conn: sqlite3.Connection, indicator_name: str) -> int:
    """
    Retrieve the indicator ID from the database, inserting it if it doesn't exist.

    Args:
        conn (sqlite3.Connection): The SQLite connection object.
        indicator_name (str): The name of the indicator.

    Returns:
        int: The indicator ID.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM indicators WHERE name = ?", (indicator_name,))
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        cursor.execute("INSERT INTO indicators (name) VALUES (?)", (indicator_name,))
        conn.commit()
        return cursor.lastrowid

def load_or_calculate_correlations(
    data: pd.DataFrame,
    original_indicators: List[str],
    max_lag: int,
    is_reverse_chronological: bool,
    symbol: str,
    timeframe: str
) -> None:
    """
    Load existing correlations from the database or calculate them if they don't exist.

    Args:
        data (pd.DataFrame): The DataFrame containing indicator and price data.
        original_indicators (List[str]): List of indicator names.
        max_lag (int): The maximum number of lag periods.
        is_reverse_chronological (bool): If True, data is in reverse chronological order.
        symbol (str): The trading symbol.
        timeframe (str): The timeframe string.

    Returns:
        None
    """
    print(f"[INFO] Calculating correlations for symbol='{symbol}', timeframe='{timeframe}'...")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        symbol_id = get_symbol_id(conn, symbol)
        timeframe_id = get_timeframe_id(conn, timeframe)

        for i, indicator_name in enumerate(original_indicators, start=1):
            print(f"[INFO] Processing indicator {i}/{len(original_indicators)}: '{indicator_name}'")
            indicator_id = get_indicator_id(conn, indicator_name)

            if not is_valid_indicator(data[indicator_name]):
                print(f"[WARN] Indicator '{indicator_name}' is invalid. Skipping.")
                continue

            cursor.execute("""
                SELECT lag FROM correlations
                WHERE symbol_id = ? AND timeframe_id = ? AND indicator_id = ?
            """, (symbol_id, timeframe_id, indicator_id))
            existing_lags = set(row[0] for row in cursor.fetchall())

            lags_to_calculate = [lag for lag in range(1, max_lag + 1) if lag not in existing_lags]
            print(f"[INFO] {len(lags_to_calculate)} lags to calculate for '{indicator_name}'")

            for lag in lags_to_calculate:
                corr_value = calculate_correlation(data, indicator_name, lag, is_reverse_chronological)
                try:
                    cursor.execute("""
                        INSERT INTO correlations (symbol_id, timeframe_id, indicator_id, lag, correlation_value)
                        VALUES (?, ?, ?, ?, ?)
                    """, (symbol_id, timeframe_id, indicator_id, lag, corr_value))
                except sqlite3.Error as e:
                    print(f"[ERROR] Failed to insert correlation for '{indicator_name}', lag {lag}: {e}")

            conn.commit()
            print(f"[INFO] Completed correlations for indicator '{indicator_name}'")

        conn.close()
        print("[INFO] All correlations processed successfully.")
    except Exception as e:
        print(f"[ERROR] Exception occurred during correlation calculations: {e}")
        if conn:
            conn.close()
        raise

def get_all_correlations(
    conn: sqlite3.Connection,
    symbol_id: int,
    timeframe_id: int,
    indicator_id: int,
    max_lag: int
) -> List[float]:
    """
    Retrieve all correlation values for a given indicator and symbol/timeframe up to max_lag.

    Args:
        conn (sqlite3.Connection): The SQLite connection object.
        symbol_id (int): The symbol ID.
        timeframe_id (int): The timeframe ID.
        indicator_id (int): The indicator ID.
        max_lag (int): The maximum lag to retrieve.

    Returns:
        List[float]: List of correlation values ordered by lag.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT lag, correlation_value FROM correlations
        WHERE symbol_id = ? AND timeframe_id = ? AND indicator_id = ?
        AND lag BETWEEN 1 AND ?
        ORDER BY lag ASC
    """, (symbol_id, timeframe_id, indicator_id, max_lag))
    rows = cursor.fetchall()
    correlations = [row[1] for row in rows]
    print(f"[DEBUG] get_all_correlations: Retrieved {len(correlations)} correlations for indicator_id={indicator_id}")
    return correlations