# correlation_utils.py

import os
import logging
import pandas as pd
import numpy as np
from typing import List
from sklearn.preprocessing import StandardScaler
import sqlite3

# Import the database path from config.py
from config import DB_PATH

def calculate_correlation(
    data: pd.DataFrame,
    indicator_name: str,
    lag: int,
    is_reverse_chronological: bool
) -> float:
    """
    Calculates the correlation between a shifted indicator and the 'close' price.

    Args:
        data (pd.DataFrame): The dataset containing indicators and 'close' price.
        indicator_name (str): The indicator column name.
        lag (int): The lag period.
        is_reverse_chronological (bool): Indicates if data is in reverse chronological order.

    Returns:
        float: The correlation coefficient or NaN if invalid.
    """
    logging.debug(f"Starting correlation for '{indicator_name}' at lag {lag}.")
    if indicator_name == 'close':
        logging.warning(f"Skipping correlation calculation for '{indicator_name}' at lag {lag} as it is the target variable.")
        return np.nan
    try:
        shift_value = lag if is_reverse_chronological else -lag
        shifted_col = data[indicator_name].shift(shift_value)
        valid_data = pd.concat([shifted_col, data['close']], axis=1).dropna()
        if not valid_data.empty:
            corr = valid_data[indicator_name].corr(valid_data['close'])
            logging.debug(f"Correlation for '{indicator_name}' at lag {lag}: {corr}")
            return corr
        logging.warning(f"No valid data for '{indicator_name}' at lag {lag}'. Returning NaN.")
        return np.nan
    except Exception as e:
        logging.error(f"Error calculating correlation for {indicator_name} at lag {lag}: {e}")
        return np.nan

def is_valid_indicator(series: pd.Series) -> bool:
    """
    Checks if the indicator series is valid for correlation calculation.

    Args:
        series (pd.Series): The indicator data series.

    Returns:
        bool: True if valid, False otherwise.
    """
    # Check for sufficient non-NaN values
    if series.isna().all():
        logging.warning("Indicator series contains only NaN values.")
        return False
    # Check for variability (non-constant)
    if series.nunique() <= 1:
        logging.warning("Indicator series is constant.")
        return False
    return True

def get_symbol_id(conn, symbol):
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM symbols WHERE symbol = ?", (symbol,))
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        # Insert the symbol into the database
        cursor.execute("INSERT INTO symbols (symbol) VALUES (?)", (symbol,))
        conn.commit()
        return cursor.lastrowid

def get_timeframe_id(conn, timeframe):
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM timeframes WHERE timeframe = ?", (timeframe,))
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        # Insert the timeframe into the database
        cursor.execute("INSERT INTO timeframes (timeframe) VALUES (?)", (timeframe,))
        conn.commit()
        return cursor.lastrowid

def get_indicator_id(conn, indicator_name):
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM indicators WHERE name = ?", (indicator_name,))
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        # Insert the indicator into the database
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
    Loads existing correlations from the database or calculates them if not present.

    Args:
        data (pd.DataFrame): The dataset.
        original_indicators (List[str]): List of indicators to calculate correlations for.
        max_lag (int): Maximum lag period.
        is_reverse_chronological (bool): Data ordering flag.
        symbol (str): Trading symbol.
        timeframe (str): Timeframe of the data.
    """
    # Connect to the database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get symbol_id and timeframe_id
    symbol_id = get_symbol_id(conn, symbol)
    timeframe_id = get_timeframe_id(conn, timeframe)

    for indicator_name in original_indicators:
        # Get indicator_id
        indicator_id = get_indicator_id(conn, indicator_name)

        # Check which lags are already calculated
        cursor.execute(
            "SELECT lag FROM correlations WHERE symbol_id = ? AND timeframe_id = ? AND indicator_id = ?",
            (symbol_id, timeframe_id, indicator_id)
        )
        existing_lags = set([row[0] for row in cursor.fetchall()])

        # Determine which lags need to be calculated
        lags_to_calculate = [lag for lag in range(1, max_lag + 1) if lag not in existing_lags]

        if not lags_to_calculate:
            logging.info(f"All correlations for indicator '{indicator_name}' already calculated. Skipping.")
            continue

        # Validate the indicator before processing
        if not is_valid_indicator(data[indicator_name]):
            logging.warning(f"Indicator '{indicator_name}' is invalid. Skipping correlation calculations.")
            continue

        logging.info(f"Calculating correlations for indicator '{indicator_name}'.")

        try:
            # Calculate correlations sequentially
            for lag in lags_to_calculate:
                calculate_correlation_and_insert(
                    data,
                    indicator_name,
                    lag,
                    is_reverse_chronological,
                    symbol_id,
                    timeframe_id,
                    indicator_id
                )
            logging.info(f"Calculated correlations for '{indicator_name}'.")
        except Exception as e:
            logging.error(f"Failed to calculate correlations for '{indicator_name}': {e}")

    conn.close()

def calculate_correlation_and_insert(
    data,
    indicator_name,
    lag,
    is_reverse_chronological,
    symbol_id,
    timeframe_id,
    indicator_id
):
    """
    Calculates the correlation and inserts it into the database.
    """
    corr_value = calculate_correlation(data, indicator_name, lag, is_reverse_chronological)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO correlations (
            symbol_id, timeframe_id, indicator_id, lag, correlation_value
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (symbol_id, timeframe_id, indicator_id, lag, corr_value)
    )
    conn.commit()
    conn.close()