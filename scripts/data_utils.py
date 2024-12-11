# scripts/data_utils.py

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import numpy as np
from load_data import load_data
from indicators import compute_all_indicators
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clear_screen() -> None:
    """
    Clears the terminal screen.
    """
    os.system('cls' if os.name == 'nt' else 'clear')

def prepare_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare data by scaling numerical features.

    Ensures that 'Volume', 'Open', 'High', 'Low' are retained for downstream processes.
    
    Returns:
        X_scaled_df: Scaled features DataFrame.
        numeric_features: List of feature column names.
    """
    excluded_columns = ['date', 'Date', 'Close']
    feature_columns = [col for col in data.columns if col not in excluded_columns and col.lower() not in ['close']]
    
    X = data[feature_columns]
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_features]
    
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled_array, columns=numeric_features, index=data.index)
    
    logging.info("Data prepared and scaled successfully.")
    
    return X_scaled_df, numeric_features

def determine_time_interval(data: pd.DataFrame) -> str:
    """
    Determine the time interval of the data based on 'Date' or 'open_time' column.

    Returns:
        Time interval as a string ('second', 'minute', 'hour', 'day', 'week').
    """
    date_col = 'Date' if 'Date' in data.columns else 'open_time' if 'open_time' in data.columns else None
    if date_col is None:
        raise ValueError("No 'Date' or 'open_time' column found.")
    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    data.dropna(subset=[date_col], inplace=True)
    if not data[date_col].is_monotonic_increasing:
        raise ValueError(f"'{date_col}' column not monotonically increasing.")
    time_diffs = data[date_col].diff().dropna().dt.total_seconds()
    if time_diffs.empty:
        raise ValueError("No time differences found.")
    diff = time_diffs.mode().iloc[0]
    if diff < 60:
        interval = 'second'
    elif diff < 3600:
        interval = 'minute'
    elif diff < 86400:
        interval = 'hour'
    elif diff < 604800:
        interval = 'day'
    else:
        interval = 'week'
    
    logging.info(f"Determined time interval: {interval}")
    return interval

def get_original_indicators(feature_names: List[str], data: pd.DataFrame) -> List[str]:
    """
    Retrieve original indicator names from feature names.

    Parameters:
        feature_names: List of feature column names.
        data: DataFrame containing the data.

    Returns:
        List of original indicator names.
    """
    original_indicators = [
        col for col in feature_names 
        if col.lower() not in ['open','high','low','close','volume'] 
        and data[col].notna().any() 
        and data[col].var() > 1e-6
    ]
    logging.info(f"Identified {len(original_indicators)} original indicators.")
    return original_indicators

def handle_missing_indicators(original_indicators: List[str], data: pd.DataFrame, expected_indicators: List[str]) -> List[str]:
    """
    Handle indicators that might be missing after processing.

    Parameters:
        original_indicators: List of original indicator names.
        data: DataFrame containing the data.
        expected_indicators: List of expected indicator names.

    Returns:
        Updated list of original indicators after handling missing indicators.
    """
    missing_indicators = [indicator for indicator in expected_indicators if indicator not in data.columns]
    if missing_indicators:
        logging.warning(f"Missing indicators detected: {missing_indicators}. They will be removed from processing.")
        original_indicators = [col for col in original_indicators if col not in missing_indicators]
    else:
        logging.info("No missing indicators detected.")
    return original_indicators
