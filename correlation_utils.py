import sqlite3
import numpy as np
import pandas as pd
from typing import List
from config import DB_PATH

def calculate_correlation(data: pd.DataFrame, indicator_name: str, lag: int, is_reverse_chronological: bool) -> float:
    if indicator_name.lower() == 'close':
        return np.nan
    shift_value = lag if is_reverse_chronological else -lag
    shifted_col = data[indicator_name].shift(shift_value)
    valid_data = pd.concat([shifted_col, data['Close']], axis=1).dropna()
    return valid_data[indicator_name].corr(valid_data['Close']) if not valid_data.empty else np.nan

def is_valid_indicator(series: pd.Series) -> bool:
    return not series.isna().all() and series.nunique() > 1

def get_symbol_id(conn: sqlite3.Connection, symbol: str) -> int:
    c = conn.cursor()
    c.execute("SELECT id FROM symbols WHERE symbol = ?", (symbol,))
    r = c.fetchone()
    if r: return r[0]
    c.execute("INSERT INTO symbols (symbol) VALUES (?)", (symbol,))
    conn.commit()
    return c.lastrowid

def get_timeframe_id(conn: sqlite3.Connection, timeframe: str) -> int:
    c = conn.cursor()
    c.execute("SELECT id FROM timeframes WHERE timeframe = ?", (timeframe,))
    r = c.fetchone()
    if r: return r[0]
    c.execute("INSERT INTO timeframes (timeframe) VALUES (?)", (timeframe,))
    conn.commit()
    return c.lastrowid

def get_indicator_id(conn: sqlite3.Connection, indicator_name: str) -> int:
    c = conn.cursor()
    c.execute("SELECT id FROM indicators WHERE name = ?", (indicator_name,))
    r = c.fetchone()
    if r: return r[0]
    c.execute("INSERT INTO indicators (name) VALUES (?)", (indicator_name,))
    conn.commit()
    return c.lastrowid

def load_or_calculate_correlations(data: pd.DataFrame, original_indicators: List[str], max_lag: int, is_reverse_chronological: bool, symbol: str, timeframe: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    symbol_id = get_symbol_id(conn, symbol)
    timeframe_id = get_timeframe_id(conn, timeframe)
    for indicator_name in original_indicators:
        if not is_valid_indicator(data[indicator_name]):
            continue
        c.execute("SELECT lag FROM correlations WHERE symbol_id=? AND timeframe_id=? AND indicator_id=?",(symbol_id, timeframe_id, get_indicator_id(conn, indicator_name)))
        existing_lags = {row[0] for row in c.fetchall()}
        lags_to_calculate = [lag for lag in range(1, max_lag+1) if lag not in existing_lags]
        for lag in lags_to_calculate:
            corr_value = calculate_correlation(data, indicator_name, lag, is_reverse_chronological)
            try:
                c.execute("INSERT INTO correlations (symbol_id,timeframe_id,indicator_id,lag,correlation_value) VALUES (?,?,?,?,?)",
                          (symbol_id,timeframe_id,get_indicator_id(conn, indicator_name),lag,corr_value))
            except:
                pass
        conn.commit()
    conn.close()

def get_all_correlations(conn: sqlite3.Connection, symbol_id: int, timeframe_id: int, indicator_id: int, max_lag: int) -> List[float]:
    c = conn.cursor()
    c.execute("SELECT correlation_value FROM correlations WHERE symbol_id=? AND timeframe_id=? AND indicator_id=? AND lag BETWEEN 1 AND ? ORDER BY lag ASC",(symbol_id,timeframe_id,indicator_id,max_lag))
    rows = c.fetchall()
    return [r[0] for r in rows]