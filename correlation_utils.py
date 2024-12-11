# filename: correlation_utils.py
import os
import pandas as pd
import numpy as np
import sqlite3
from typing import List
from sklearn.preprocessing import StandardScaler
from config import DB_PATH

def calculate_correlation(data:pd.DataFrame,indicator_name:str,lag:int,is_reverse_chronological:bool)->float:
    if indicator_name=='close':
        return np.nan
    try:
        shift_value=lag if is_reverse_chronological else -lag
        shifted_col=data[indicator_name].shift(shift_value)
        valid_data=pd.concat([shifted_col,data['close']],axis=1).dropna()
        if not valid_data.empty:
            corr=valid_data[indicator_name].corr(valid_data['close'])
            return corr
        return np.nan
    except:
        return np.nan

def is_valid_indicator(series:pd.Series)->bool:
    if series.isna().all():
        return False
    if series.nunique()<=1:
        return False
    return True

def get_symbol_id(conn,symbol):
    cursor=conn.cursor()
    cursor.execute("SELECT id FROM symbols WHERE symbol = ?",(symbol,))
    result=cursor.fetchone()
    if result:
        return result[0]
    else:
        cursor.execute("INSERT INTO symbols (symbol) VALUES (?)",(symbol,))
        conn.commit()
        return cursor.lastrowid

def get_timeframe_id(conn,timeframe):
    cursor=conn.cursor()
    cursor.execute("SELECT id FROM timeframes WHERE timeframe = ?",(timeframe,))
    result=cursor.fetchone()
    if result:
        return result[0]
    else:
        cursor.execute("INSERT INTO timeframes (timeframe) VALUES (?)",(timeframe,))
        conn.commit()
        return cursor.lastrowid

def get_indicator_id(conn,indicator_name):
    cursor=conn.cursor()
    cursor.execute("SELECT id FROM indicators WHERE name = ?",(indicator_name,))
    result=cursor.fetchone()
    if result:
        return result[0]
    else:
        cursor.execute("INSERT INTO indicators (name) VALUES (?)",(indicator_name,))
        conn.commit()
        return cursor.lastrowid

def load_or_calculate_correlations(data:pd.DataFrame,original_indicators:List[str],max_lag:int,is_reverse_chronological:bool,symbol:str,timeframe:str)->None:
    conn=sqlite3.connect(DB_PATH)
    cursor=conn.cursor()
    symbol_id=get_symbol_id(conn,symbol)
    timeframe_id=get_timeframe_id(conn,timeframe)
    for indicator_name in original_indicators:
        indicator_id=get_indicator_id(conn,indicator_name)
        cursor.execute("SELECT lag FROM correlations WHERE symbol_id = ? AND timeframe_id = ? AND indicator_id = ?",(symbol_id,timeframe_id,indicator_id))
        existing_lags=set([row[0]for row in cursor.fetchall()])
        lags_to_calculate=[lag for lag in range(1,max_lag+1) if lag not in existing_lags]
        if not lags_to_calculate:
            continue
        if not is_valid_indicator(data[indicator_name]):
            continue
        for lag in lags_to_calculate:
            calculate_correlation_and_insert(data,indicator_name,lag,is_reverse_chronological,symbol_id,timeframe_id,indicator_id)
    conn.close()

def calculate_correlation_and_insert(data,indicator_name,lag,is_reverse_chronological,symbol_id,timeframe_id,indicator_id):
    corr_value=calculate_correlation(data,indicator_name,lag,is_reverse_chronological)
    conn=sqlite3.connect(DB_PATH)
    cursor=conn.cursor()
    cursor.execute(
        """
        INSERT INTO correlations (
            symbol_id, timeframe_id, indicator_id, lag, correlation_value
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (symbol_id,timeframe_id,indicator_id,lag,corr_value)
    )
    conn.commit()
    conn.close()