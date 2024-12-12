# data_utils.py
import os, sys, pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
from datetime import datetime
from joblib import Parallel, delayed
from load_data import load_data
from indicators import compute_all_indicators

def clear_screen():
    os.system('cls' if os.name=='nt' else 'clear')

def prepare_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    features = data.columns.difference(['date', 'open', 'high', 'low', 'close', 'volume'])
    X = data[features].select_dtypes(include=[np.number])
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=data.index)
    return X_scaled, X.columns.tolist()

def determine_time_interval(data: pd.DataFrame) -> str:
    date_col = 'Date' if 'Date' in data else 'open_time' if 'open_time' in data else None
    if not date_col:
        raise ValueError("No 'Date' or 'open_time' column.")
    data[date_col] = pd.to_datetime(data[date_col], errors='coerce').dropna()
    if not data[date_col].is_monotonic_increasing:
        raise ValueError(f"'{date_col}' not increasing.")
    diff = data[date_col].diff().dt.total_seconds().mode()[0]
    if diff < 60:
        return 'second'
    elif diff < 3600:
        return 'minute'
    elif diff < 86400:
        return 'hour'
    elif diff < 604800:
        return 'day'
    else:
        return 'week'

def get_original_indicators(feature_names: List[str], data: pd.DataFrame) -> List[str]:
    return [
        col for col in feature_names
        if col.lower() not in ['open', 'high', 'low', 'close', 'volume']
        and data[col].notna().any()
        and data[col].var() > 1e-6
    ]

def handle_missing_indicators(original_inds: List[str], data: pd.DataFrame, expected: List[str]) -> List[str]:
    missing = [ind for ind in expected if ind not in data.columns]
    return [col for col in original_inds if col not in missing]
