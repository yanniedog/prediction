# filename: data_utils.py
import os
import sys
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from typing import Tuple,List
from datetime import datetime
from joblib import Parallel,delayed
import numpy as np
from load_data import load_data
from indicators import compute_all_indicators
def clear_screen()->None:
    os.system('cls' if os.name=='nt' else 'clear')
def prepare_data(data:pd.DataFrame)->Tuple[pd.DataFrame,List[str]]:
    feature_columns=data.columns.difference(['date','open','high','low','close','volume'])
    X=data[feature_columns]
    numeric_features=X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_features=set(feature_columns)-set(numeric_features)
    if non_numeric_features:
        logging.warning(f"Non-numeric features detected and will be excluded: {non_numeric_features}")
        feature_columns=numeric_features
        X=X[feature_columns]
    scaler=StandardScaler()
    X_scaled_array=scaler.fit_transform(X)
    X_scaled_df=pd.DataFrame(X_scaled_array,columns=feature_columns,index=data.index)
    logging.info("Data has been scaled.")
    return X_scaled_df,feature_columns
def determine_time_interval(data:pd.DataFrame)->str:
    date_col='Date' if 'Date' in data.columns else 'open_time' if 'open_time' in data.columns else None
    if date_col is None:
        raise ValueError("No 'Date' or 'open_time' column found in data.")
    data[date_col]=pd.to_datetime(data[date_col],errors='coerce')
    data.dropna(subset=[date_col],inplace=True)
    if not data[date_col].is_monotonic_increasing:
        raise ValueError(f"The '{date_col}' column is not monotonically increasing.")
    time_diffs=data[date_col].diff().dropna().dt.total_seconds()
    logging.info("Determined time differences between dates.")
    if time_diffs.empty:
        raise ValueError("No time differences found.")
    logging.info(f"First few time differences (in seconds):\n{time_diffs.head()}")
    most_common_diff=time_diffs.mode().iloc[0]
    if most_common_diff<60:
        return 'second'
    elif most_common_diff<3600:
        return 'minute'
    elif most_common_diff<86400:
        return 'hour'
    elif most_common_diff<604800:
        return 'day'
    else:
        return 'week'
def get_original_indicators(feature_names:List[str],data:pd.DataFrame)->List[str]:
    original_indicators=[col for col in feature_names if col.lower()not in['open','high','low','close','volume']and data[col].notna().any()and data[col].var()>1e-6]
    logging.info(f"Original indicators identified: {original_indicators}")
    return original_indicators
def handle_missing_indicators(original_indicators:List[str],data:pd.DataFrame,expected_indicators:List[str])->List[str]:
    missing_indicators=[indicator for indicator in expected_indicators if indicator not in data.columns]
    for indicator in missing_indicators:
        logging.warning(f"Indicator '{indicator}' is missing from the data. It will be excluded.")
        original_indicators=[col for col in original_indicators if col!=indicator]
    logging.info(f"Final list of indicators for analysis: {original_indicators}")
    return original_indicators