import os, sys, pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
from datetime import datetime
from joblib import Parallel, delayed
from load_data import load_data
from indicators import compute_all_indicators

def clear_screen():
    os.system('cls' if os.name=='nt' else 'clear')

import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_data(data):
    numeric_features = data.select_dtypes(include=[np.number])
    feature_cols = numeric_features.columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_features)
    return X_scaled, feature_cols

def determine_time_interval(data: pd.DataFrame) -> str:
    date_col = 'Date' if 'Date' in data else 'open_time' if 'open_time' in data else None
    if not date_col: raise ValueError("No 'Date' or 'open_time' column.")
    data[date_col] = pd.to_datetime(data[date_col], errors='coerce').dropna()
    if not data[date_col].is_monotonic_increasing: raise ValueError(f"'{date_col}' not increasing.")
    diff = data[date_col].diff().dt.total_seconds().mode()[0]
    return 'second' if diff <60 else 'minute' if diff <3600 else 'hour' if diff <86400 else 'day' if diff <604800 else 'week'

def get_original_indicators(feature_names: List[str], data: pd.DataFrame) -> List[str]:
    return [col for col in feature_names if col.lower() not in ['open','high','low','close','volume'] and data[col].notna().any() and data[col].var() >1e-6]

def handle_missing_indicators(original_inds: List[str], data: pd.DataFrame, expected: List[str]) -> List[str]:
    missing = [ind for ind in expected if ind not in data]
    return [col for col in original_inds if col not in missing]
