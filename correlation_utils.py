# correlation_utils.py
import pandas as pd, numpy as np, sqlite3
from typing import List
from sklearn.preprocessing import StandardScaler
from config import DB_PATH

def calculate_correlation(data: pd.DataFrame, indicator: str, lag: int, reverse: bool) -> float:
    if indicator.lower() == 'close': return np.nan
    try:
        shifted = data[indicator].shift(lag if reverse else -lag)
        valid = pd.concat([shifted, data['Close']], axis=1).dropna()
        return valid[indicator].corr(valid['Close']) if not valid.empty else np.nan
    except:
        return np.nan

def is_valid_indicator(series: pd.Series) -> bool:
    return not series.isna().all() and series.nunique() > 1

def get_symbol_id(conn: sqlite3.Connection, symbol: str) -> int:
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM symbols WHERE symbol = ?", (symbol,))
    res = cursor.fetchone()
    if res: return res[0]
    cursor.execute("INSERT INTO symbols (symbol) VALUES (?)", (symbol,))
    conn.commit()
    return cursor.lastrowid

def get_timeframe_id(conn: sqlite3.Connection, timeframe: str) -> int:
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM timeframes WHERE timeframe = ?", (timeframe,))
    res = cursor.fetchone()
    if res: return res[0]
    cursor.execute("INSERT INTO timeframes (timeframe) VALUES (?)", (timeframe,))
    conn.commit()
    return cursor.lastrowid

def get_indicator_id(conn: sqlite3.Connection, indicator: str) -> int:
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM indicators WHERE name = ?", (indicator,))
    res = cursor.fetchone()
    if res: return res[0]
    cursor.execute("INSERT INTO indicators (name) VALUES (?)", (indicator,))
    conn.commit()
    return cursor.lastrowid

def load_or_calculate_correlations(data: pd.DataFrame, indicators: List[str], max_lag: int, reverse: bool, symbol: str, timeframe: str):
    print(f"Calculating correlations for {symbol} {timeframe}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    sym_id = get_symbol_id(conn, symbol)
    tf_id = get_timeframe_id(conn, timeframe)
    for i, ind in enumerate(indicators, 1):
        print(f"Processing {i}/{len(indicators)}: {ind}")
        if not is_valid_indicator(data[ind]):
            print(f"Invalid indicator '{ind}'. Skipping.")
            continue
        cursor.execute("SELECT lag FROM correlations WHERE symbol_id=? AND timeframe_id=? AND indicator_id=?", (sym_id, tf_id, get_indicator_id(conn, ind)))
        existing = {row[0] for row in cursor.fetchall()}
        to_calc = [lag for lag in range(1, max_lag+1) if lag not in existing]
        print(f"Calculating {len(to_calc)} lags for '{ind}'")
        for lag in to_calc:
            corr = calculate_correlation(data, ind, lag, reverse)
            try:
                cursor.execute("""
                    INSERT INTO correlations (symbol_id, timeframe_id, indicator_id, lag, correlation_value)
                    VALUES (?, ?, ?, ?, ?);""", (sym_id, tf_id, get_indicator_id(conn, ind), lag, corr))
            except sqlite3.Error as e:
                print(f"Insert error for '{ind}' lag {lag}: {e}")
        conn.commit()
        print(f"Completed '{ind}'")
    conn.close()
    print("All correlations processed.")

def get_all_correlations(conn: sqlite3.Connection, sym_id: int, tf_id: int, ind_id: int, max_lag: int) -> List[float]:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT correlation_value FROM correlations
        WHERE symbol_id=? AND timeframe_id=? AND indicator_id=? AND lag BETWEEN 1 AND ?
        ORDER BY lag ASC;""", (sym_id, tf_id, ind_id, max_lag))
    return [row[0] for row in cursor.fetchall()]
