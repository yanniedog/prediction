# indicators.py

import logging
import pandas as pd
import numpy as np
import talib as ta
import pandas_ta as pta
import json
from typing import List
from indicator_config_parser import parse_indicators_json

logger = logging.getLogger(__name__)

def compute_obv_price_divergence(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Compute the OBV Price Divergence indicator based on provided parameters.

    Args:
        data (pd.DataFrame): DataFrame containing 'open', 'high', 'low', 'close', 'volume'.
        params (dict): Parameters for the OBV Price Divergence indicator.

    Returns:
        pd.DataFrame: DataFrame with the 'obv_price_divergence' column added.
    """
    method = params.get("method", "Difference")
    obv_method = params.get("obv_method", "SMA")
    obv_period = params.get("obv_period", 14)
    price_input_type = params.get("price_input_type", "OHLC/4")
    price_method = params.get("price_method", "SMA")
    price_period = params.get("price_period", 14)
    bearish_threshold = params.get("bearish_threshold", -0.8)
    bullish_threshold = params.get("bullish_threshold", 0.8)
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
    return data

def compute_eyeX_MFV_volume(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Compute the EyeX MFV Volume indicator based on provided parameters.

    Args:
        data (pd.DataFrame): DataFrame containing 'high', 'low', 'close', 'volume'.
        params (dict): Parameters for the EyeX MFV Volume indicator.

    Returns:
        pd.DataFrame: DataFrame with the 'EyeX MFV Volume' column added.
    """
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
    return data

def compute_eyeX_MFV_support_resistance(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Compute the EyeX MFV Support/Resistance Bull indicator based on provided parameters.

    Args:
        data (pd.DataFrame): DataFrame containing 'high', 'low', 'close', 'volume'.
        params (dict): Parameters for the EyeX MFV S/R Bull indicator.

    Returns:
        pd.DataFrame: DataFrame with 'EyeX MFV S/R Bull' and 'EyeX MFV S/R Bear' columns added.
    """
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
    return data

def z_score(x: np.ndarray) -> float:
    """
    Calculate the Z-score of the last element in the array.

    Args:
        x (np.ndarray): Array of numerical values.

    Returns:
        float: Z-score of the last element.
    """
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0
    return (x[-1] - mean) / std

def compute_configured_indicators(data: pd.DataFrame, indicators_list: List[str], indicator_params_path: str = 'indicator_params.json') -> pd.DataFrame:
    """
    Compute configured indicators based on the provided list and parameters from indicator_params.json.

    Args:
        data (pd.DataFrame): The input data containing 'open', 'high', 'low', 'close', 'volume'.
        indicators_list (List[str]): List of indicator names to compute.
        indicator_params_path (str): Path to indicator_params.json

    Returns:
        pd.DataFrame: The data with configured indicators added.
    """
    with open(indicator_params_path, 'r') as f:
        indicator_params = json.load(f)
    
    for indicator_name in indicators_list:
        if indicator_name not in indicator_params:
            logger.warning(f"No parameters found for '{indicator_name}'. Skipping configuration.")
            continue
        params = indicator_params[indicator_name]
        
        try:
            if indicator_name == "obv_price_divergence":
                data = compute_obv_price_divergence(data, params)
            elif indicator_name == "EyeX MFV Volume":
                data = compute_eyeX_MFV_volume(data, params)
            elif indicator_name == "EyeX MFV S/R Bull":
                data = compute_eyeX_MFV_support_resistance(data, params)
            else:
                # Dynamically handle standard indicators
                # Ensure the indicator exists in TA-Lib or pandas_ta
                # For simplicity, use pandas_ta if available
                if hasattr(ta, indicator_name.upper()):
                    ta_func = getattr(ta, indicator_name.upper())
                    # Extract parameters excluding 'name'
                    ta_params = {k: v for k, v in params.items()}
                    data[indicator_name] = ta_func(data['close'], **ta_params)
                elif indicator_name.lower() in pta.indicators():
                    data[indicator_name] = pta.ta(indicator_name.lower(), close=data['close'], **params)
                else:
                    logger.warning(f"Indicator '{indicator_name}' not recognized in TA-Lib or pandas_ta. Skipping.")
        except Exception as e:
            logger.error(f"Error computing indicator '{indicator_name}': {e}")
    
    data.dropna(inplace=True)
    return data

def compute_all_indicators(data: pd.DataFrame, indicator_params_path: str = 'indicator_params.json') -> pd.DataFrame:
    """
    Compute all indicators based on the parameters defined in indicator_params.json.

    Args:
        data (pd.DataFrame): The input data containing 'open', 'high', 'low', 'close', 'volume'.
        indicator_params_path (str): Path to indicator_params.json

    Returns:
        pd.DataFrame: The data with all indicators added.
    """
    with open(indicator_params_path, 'r') as f:
        indicator_params = json.load(f)
    
    indicators_list = list(indicator_params.keys())
    data = compute_configured_indicators(data, indicators_list, indicator_params_path)
    return data
