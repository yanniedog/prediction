# correlation_utils.py

import sqlite3
import numpy as np
import pandas as pd
from typing import List
from config import DB_PATH
from tqdm import tqdm
import logging
from correlation_database import CorrelationDatabase
from visualization_utils import generate_individual_indicator_chart
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_correlation(data: pd.DataFrame, indicator_name: str, lag: int, is_reverse_chronological: bool) -> float:
    """
    Calculate the Pearson correlation coefficient between the indicator and Close price at a given lag.

    Parameters:
    - data: DataFrame containing the data.
    - indicator_name: Name of the indicator column.
    - lag: Lag value.
    - is_reverse_chronological: Boolean indicating if the data is in reverse chronological order.

    Returns:
    - Correlation coefficient as a float.
    """
    if indicator_name.lower() == 'close':
        return np.nan

    shift_value = lag if is_reverse_chronological else -lag
    shifted_col = data[indicator_name].shift(shift_value)
    valid_data = pd.concat([shifted_col, data['Close']], axis=1).dropna()

    if valid_data.empty:
        return np.nan

    return valid_data[indicator_name].corr(valid_data['Close'])

def is_valid_indicator(series: pd.Series) -> bool:
    """
    Check if the indicator series is valid (not all NaNs and has more than one unique value).

    Parameters:
    - series: Pandas Series of the indicator.

    Returns:
    - Boolean indicating validity.
    """
    return not series.isna().all() and series.nunique() > 1

def load_or_calculate_correlations(
    data: pd.DataFrame, 
    original_indicators: List[str], 
    max_lag: int, 
    is_reverse_chronological: bool, 
    symbol: str, 
    timeframe: str
) -> None:
    """
    Load existing correlations from the database or calculate and insert them if they do not exist.
    After calculating all correlations for an indicator, generate an individual indicator chart.

    Parameters:
    - data: DataFrame containing the data.
    - original_indicators: List of indicator column names.
    - max_lag: Maximum lag to calculate correlations for.
    - is_reverse_chronological: Boolean indicating if the data is in reverse chronological order.
    - symbol: Trading symbol (e.g., 'SOLUSDT').
    - timeframe: Timeframe interval (e.g., '1w').
    """
    correlation_db = CorrelationDatabase(DB_PATH)

    indicator_correlations = {}

    for indicator in tqdm(original_indicators, desc="Calculating correlations", unit="indicator"):
        if not is_valid_indicator(data[indicator]):
            logging.warning(f"Indicator '{indicator}' is not valid. Skipping.")
            continue

        corr_values = []

        for lag in range(1, max_lag + 1):
            corr_value = calculate_correlation(data, indicator, lag, is_reverse_chronological)
            corr_values.append(corr_value)

            try:
                correlation_db.insert_correlation(symbol, timeframe, indicator, lag, corr_value)
            except Exception as e:
                logging.error(f"Failed to insert correlation for {indicator} at lag {lag}: {e}")
                continue

        indicator_correlations[indicator] = corr_values

        try:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            base_csv_filename = f"{symbol}_{timeframe}"
            generate_individual_indicator_chart(
                indicator_name=indicator,
                correlations=corr_values,
                max_lag=max_lag,
                timestamp=timestamp,
                base_csv_filename=base_csv_filename
            )
            logging.info(f"Generated individual indicator chart for {indicator}.")
        except Exception as e:
            logging.error(f"Failed to generate chart for {indicator}: {e}")
            continue

    correlation_db.close()

def get_all_correlations(
    conn: sqlite3.Connection, 
    symbol_id: int, 
    timeframe_id: int, 
    indicator_id: int, 
    max_lag: int
) -> List[float]:
    """
    Retrieve all correlation values for a specific symbol, timeframe, and indicator up to max_lag.

    Parameters:
    - conn: SQLite3 connection object.
    - symbol_id: ID of the symbol in the database.
    - timeframe_id: ID of the timeframe in the database.
    - indicator_id: ID of the indicator in the database.
    - max_lag: Maximum lag to retrieve correlations for.

    Returns:
    - List of correlation values ordered by lag ascending.
    """
    cursor = conn.cursor()
    query = """
        SELECT correlation_value 
        FROM correlations 
        WHERE symbol_id=? AND timeframe_id=? AND indicator_id=? AND lag BETWEEN 1 AND ? 
        ORDER BY lag ASC
    """
    cursor.execute(query, (symbol_id, timeframe_id, indicator_id, max_lag))
    rows = cursor.fetchall()
    return [row[0] for row in rows]