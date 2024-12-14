# correlation_utils.py
import logging
import pandas as pd
import numpy as np
import json
from typing import List
from config import DB_PATH
from correlation_database import CorrelationDatabase

logger = logging.getLogger()

def calculate_correlation(data: pd.DataFrame, indicator: str, lag: int, reverse: bool) -> float:
    if indicator.lower() == 'close': return np.nan
    try:
        shifted = data[indicator].shift(lag if reverse else -lag)
        valid = pd.concat([shifted, data['close']], axis=1).dropna()
        return valid[indicator].corr(valid['close']) if not valid.empty else np.nan
    except:
        return np.nan

def is_valid_indicator(series: pd.Series) -> bool:
    return not series.isna().all() and series.nunique() > 1

def load_or_calculate_correlations(data: pd.DataFrame, indicators: List[str], max_lag: int, reverse: bool, symbol: str, timeframe: str):
    print(f"Calculating correlations for {symbol} {timeframe}")
    db = CorrelationDatabase(DB_PATH)
    cursor = db.conn.cursor()
    for indicator in indicators:
        cursor.execute("""
            SELECT ic.id, ic.config FROM indicator_configs ic
            JOIN indicators i ON ic.indicator_id = i.id
            WHERE i.name = ?;""", (indicator,))
        configs = cursor.fetchall()
        for config_id, config_json in configs:
            config = json.loads(config_json)
            config_name = f"{indicator}_config_{config_id}"
            if config_name not in data.columns:
                logger.error(f"Indicator column '{config_name}' not found in data.")
                continue
            if not is_valid_indicator(data[config_name]):
                logger.warning(f"Invalid indicator '{config_name}'. Skipping.")
                continue
            cursor.execute("""
                SELECT lag FROM correlations 
                WHERE symbol_id=(SELECT id FROM symbols WHERE symbol=?) 
                AND timeframe_id=(SELECT id FROM timeframes WHERE timeframe=?) 
                AND indicator_config_id=?;""", (symbol, timeframe, config_id))
            existing_lags = {row[0] for row in cursor.fetchall()}
            to_calc = [lag for lag in range(1, max_lag+1) if lag not in existing_lags]
            for lag in to_calc:
                corr = calculate_correlation(data, config_name, lag, reverse)
                try:
                    db.insert_correlation(symbol, timeframe, {'indicator_name': indicator, 'params': config}, lag, corr)
                except Exception as e:
                    logger.error(f"Error inserting correlation for {config_name} lag {lag}: {e}")
    db.close()
    print("All correlations processed.")
