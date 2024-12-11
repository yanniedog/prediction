import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import numpy as np
from load_data import load_data
from indicators import compute_all_indicators

def clear_screen()->None:
    os.system('cls' if os.name=='nt' else 'clear')

def prepare_data(data: pd.DataFrame) -> Tuple[pd.DataFrame,List[str]]:
    feature_columns = data.columns.difference(['date','open','high','low','close','volume','Date','Open','High','Low','Close','Volume'])
    X = data[feature_columns]
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_features]
    scaler = StandardScaler()
    X_scaled_df = pd.DataFrame(scaler.fit_transform(X), columns=numeric_features, index=data.index)
    return X_scaled_df, numeric_features

def determine_time_interval(data: pd.DataFrame) -> str:
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
    if diff < 60: return 'second'
    elif diff < 3600: return 'minute'
    elif diff < 86400: return 'hour'
    elif diff < 604800: return 'day'
    else: return 'week'

def get_original_indicators(feature_names: List[str], data: pd.DataFrame) -> List[str]:
    return [col for col in feature_names if col.lower() not in ['open','high','low','close','volume'] and data[col].notna().any() and data[col].var()>1e-6]

def handle_missing_indicators(original_indicators: List[str], data: pd.DataFrame, expected_indicators: List[str]) -> List[str]:
    return [col for col in original_indicators if col not in expected_indicators or col in data.columns]
