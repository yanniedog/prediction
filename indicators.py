# indicators.py
import logging
import pandas as pd
import numpy as np
import talib as ta
import pandas_ta as pta
import json
from typing import List
from indicator_config_parser import parse_indicators_json
from sqlite_data_manager import create_connection, fetch_indicator_configs
from config import DB_PATH

logger = logging.getLogger(__name__)

def z_score(x: np.ndarray) -> float:
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0
    return (x[-1] - mean) / std

def compute_custom_indicator(data: pd.DataFrame, indicator_name: str, params: dict, input_columns: List[str]) -> pd.DataFrame:
    try:
        if indicator_name in ["obv_price_divergence", "EyeX MFV Volume", "EyeX MFV S/R Bull", "EyeX MFV S/R Bear"]:
            if indicator_name == "obv_price_divergence":
                method = params.get("method", "Difference")
                obv_method = params.get("obv_method", "SMA")
                obv_period = params.get("obv_period", 14)
                price_input_type = params.get("price_input_type", "OHLC/4")
                price_method = params.get("price_method", "SMA")
                price_period = params.get("price_period", 14)
                smoothing = params.get("smoothing", 0.01)

                price_map = {
                    "close": data['close'],
                    "open": data['open'],
                    "high": data['high'],
                    "low": data['low'],
                    "hl/2": (data['high'] + data['low']) / 2,
                    "ohlc/4": (data[['open','high','low','close']].sum(axis=1) / 4)
                }
                selected_price = price_map.get(price_input_type.lower(), (data['open'] + data['high'] + data['low'] + data['close']) / 4)

                obv = ta.OBV(data['close'], data['volume'])
                obv_ma = ta.SMA(obv, timeperiod=obv_period) if obv_method.upper() == "SMA" else ta.EMA(obv, timeperiod=obv_period)
                price_ma = ta.SMA(selected_price, timeperiod=price_period) if price_method.upper() == "SMA" else ta.EMA(selected_price, timeperiod=price_period)

                obv_change = (obv_ma - obv_ma.shift(1)) / obv_ma.shift(1) * 100
                price_change = (price_ma - price_ma.shift(1)) / price_ma.shift(1) * 100

                if method == "Difference":
                    metric = obv_change - price_change
                elif method == "Ratio":
                    metric = obv_change / np.maximum(smoothing, np.abs(price_change))
                else:
                    metric = np.log(np.maximum(smoothing, np.abs(obv_change)) / np.maximum(smoothing, np.abs(price_change)))

                data['obv_price_divergence'] = metric

            elif indicator_name == "EyeX MFV Volume":
                ranges = params.get("ranges", [50, 75, 100, 200])

                mf_multiplier = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
                mf_multiplier = mf_multiplier.replace([np.inf, -np.inf], 0).fillna(0)
                mf_volume = mf_multiplier * data['volume']

                combined_mfv = sum([
                    (mf_volume.rolling(window=br, min_periods=1).sum() - mf_volume.shift(br).fillna(0))
                    .rolling(window=br, min_periods=1)
                    .apply(z_score, raw=True) * 10
                    for br in ranges
                ]).clip(-400, 400)

                data['EyeX MFV Volume'] = combined_mfv

            elif indicator_name == "EyeX MFV S/R Bull":
                ranges = params.get("ranges", [50, 75, 100, 200])
                pivot_lookback = params.get("pivot_lookback", 5)
                price_proximity = params.get("price_proximity", 0.00001)

                mf_multiplier = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
                mf_multiplier = mf_multiplier.replace([np.inf, -np.inf], 0).fillna(0)
                mf_volume = mf_multiplier * data['volume']

                combined_mfv = sum([
                    (mf_volume.rolling(window=br, min_periods=1).sum() - mf_volume.shift(br).fillna(0))
                    .rolling(window=br, min_periods=1)
                    .apply(z_score, raw=True) * 10
                    for br in ranges
                ])

                pivot_high = data['high'][(data['high'] == data['high'].rolling(window=pivot_lookback*2+1, center=True).max())]
                pivot_low = data['low'][(data['low'] == data['low'].rolling(window=pivot_lookback*2+1, center=True).min())]

                resistance_levels, support_levels = [], []
                bull_attack, bear_attack = [], []
                max_levels = 10

                for i in range(len(data)):
                    if i in pivot_high.index:
                        resistance_levels.insert(0, data['high'].iloc[i])
                        resistance_levels = resistance_levels[:max_levels]
                    if i in pivot_low.index:
                        support_levels.insert(0, data['low'].iloc[i])
                        support_levels = support_levels[:max_levels]

                    close = data['close'].iloc[i]
                    near_res = any(abs(close - res)/res <= price_proximity for res in resistance_levels)
                    near_sup = any(abs(close - sup)/sup <= price_proximity for sup in support_levels)

                    bull_attack.append(near_res and combined_mfv.iloc[i] > 0)
                    bear_attack.append(near_sup and combined_mfv.iloc[i] < 0)

                data['EyeX MFV S/R Bull'] = bull_attack
                data['EyeX MFV S/R Bear'] = bear_attack

        elif indicator_name.upper() in ta.get_functions():
            try:
                clean_params = {k: v for k, v in params.items() if k != 'input_columns'}
                inputs = [data[col] for col in input_columns]
                ta_func = getattr(ta, indicator_name.upper())
                result = ta_func(*inputs, **clean_params)
                if isinstance(result, tuple):
                    for idx, res in enumerate(result):
                        column_name = f"{indicator_name}_{idx}"
                        data[column_name] = res
                else:
                    data[indicator_name] = result
            except Exception as e:
                logger.error(f"Error computing indicator '{indicator_name}': {e}")
        elif indicator_name.lower() in [i.lower() for i in pta.available_indicators()]:
            try:
                clean_params = {k: v for k, v in params.items() if k != 'input_columns'}
                pta_func = getattr(pta, indicator_name.lower())
                inputs = {key: data[key] for key in input_columns if key in data.columns}
                result = pta_func(**clean_params, **inputs)
                if isinstance(result, pd.DataFrame):
                    for col in result.columns:
                        data[col] = result[col]
                else:
                    data[indicator_name] = result
            except Exception as e:
                logger.error(f"Error computing indicator '{indicator_name}': {e}")
        else:
            logger.warning(f"Indicator '{indicator_name}' not recognized in TA-Lib or pandas_ta. Skipping.")
    except Exception as e:
        logger.error(f"Error processing custom indicator '{indicator_name}': {e}")
    return data

def compute_configured_indicators(data: pd.DataFrame, indicators_list: List[str], db_path: str = DB_PATH, indicator_params_path: str = 'indicator_params.json') -> pd.DataFrame:
    try:
        with open(indicator_params_path, 'r') as f:
            indicator_params = json.load(f)
    except Exception as e:
        logger.error(f"Error loading indicator parameters: {e}")
        return data

    conn = create_connection(db_path)
    if not conn:
        logger.error("Failed to connect to the database.")
        return data
    cursor = conn.cursor()
    for indicator_name in indicators_list:
        try:
            cursor.execute("""
                SELECT config FROM indicator_configs 
                JOIN indicators ON indicator_configs.indicator_id = indicators.id
                WHERE indicators.name = ?
            """, (indicator_name,))
            rows = cursor.fetchall()
            configs = [json.loads(row[0]) for row in rows]
            input_columns = indicator_params.get(indicator_name, {}).get("input_columns", ["close"])
            for config in configs:
                data = compute_custom_indicator(data, indicator_name, config, input_columns)
        except Exception as e:
            logger.error(f"Error computing indicator '{indicator_name}': {e}")
    conn.close()
    data.dropna(inplace=True)
    return data

def compute_all_indicators(data: pd.DataFrame, db_path: str = DB_PATH, indicator_params_path: str = 'indicator_params.json') -> pd.DataFrame:
    try:
        with open(indicator_params_path, 'r') as f:
            indicators_list = json.load(f)
        indicators_keys = list(indicators_list.keys())
        for indicator in indicators_keys:
            params = indicators_list[indicator]
            input_columns = params.get("input_columns", ["close"])
            data = compute_custom_indicator(data, indicator, params, input_columns)
    except Exception as e:
        logger.error(f"Error computing all indicators: {e}")
    return data
