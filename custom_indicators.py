# custom_indicators.py
import pandas as pd
import numpy as np
import talib as ta
import logging
from typing import List # Added List import

logger = logging.getLogger(__name__)

# --- Column Name Constants ---
OBV_PRICE_DIVERGENCE = 'obv_price_divergence'
VOLUME_OSCILLATOR = 'volume_osc'
VWAP = 'vwap'
PVI = 'pvi'
NVI = 'nvi'

# --- Helper for Required Columns ---
def _check_required_cols(data: pd.DataFrame, required: List[str], indicator_name: str) -> bool:
    """Checks if DataFrame contains required columns."""
    missing = [col for col in required if col not in data.columns]
    if missing:
        logger.error(f"Missing required columns for {indicator_name}: {missing}. Skipping.")
        return False
    return True

# --- Custom Indicator Functions ---

def compute_obv_price_divergence(data: pd.DataFrame, method: str ="Difference", obv_method: str ="SMA", obv_period: int =14,
                                 price_input_type: str ="OHLC/4", price_method: str ="SMA", price_period: int =14,
                                 smoothing: float =0.01) -> pd.DataFrame:
    """Calculates OBV/Price divergence."""
    col_name = OBV_PRICE_DIVERGENCE
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not _check_required_cols(data, required_cols, col_name):
        data[col_name] = np.nan
        return data

    logger.debug(f"Computing {col_name}: method={method}, obv={obv_method}/{obv_period}, price={price_input_type}/{price_method}/{price_period}")

    try:
        # Select Price Series
        price_map = {
            "close": data['close'], "open": data['open'], "high": data['high'], "low": data['low'],
            "hl/2": (data['high'] + data['low']) / 2,
            "ohlc/4": (data['open'] + data['high'] + data['low'] + data['close']) / 4
        }
        selected_price = price_map.get(price_input_type.lower())
        if selected_price is None: raise ValueError(f"Unsupported price input type: {price_input_type}")

        # Calculate OBV and smoothed OBV
        obv = ta.OBV(data['close'], data['volume'])
        obv_ma = obv
        if obv_method == "SMA": obv_ma = ta.SMA(obv, timeperiod=obv_period)
        elif obv_method == "EMA": obv_ma = ta.EMA(obv, timeperiod=obv_period)
        elif obv_method is not None and obv_method.lower() != "none": raise ValueError(f"Unsupported obv_method: {obv_method}")

        # Calculate Smoothed Price
        price_ma = selected_price
        if price_method == "SMA": price_ma = ta.SMA(selected_price, timeperiod=price_period)
        elif price_method == "EMA": price_ma = ta.EMA(selected_price, timeperiod=price_period)
        elif price_method is not None and price_method.lower() != "none": raise ValueError(f"Unsupported price_method: {price_method}")

        # Calculate Percentage Changes robustly
        obv_change_percent = obv_ma.pct_change().multiply(100).replace([np.inf, -np.inf], np.nan)
        price_change_percent = price_ma.pct_change().multiply(100).replace([np.inf, -np.inf], np.nan)

        # Calculate Divergence Metric
        metric = pd.Series(np.nan, index=data.index)
        if method == "Difference":
            metric = obv_change_percent - price_change_percent
        elif method == "Ratio":
             denominator = price_change_percent.abs().add(max(1e-9, smoothing))
             metric = obv_change_percent.divide(denominator)
        elif method == "Log Ratio":
             obv_ratio = obv_ma.divide(obv_ma.shift(1).replace(0, np.nan)).clip(lower=1e-9)
             price_ratio = price_ma.divide(price_ma.shift(1).replace(0, np.nan)).clip(lower=1e-9)
             metric = np.log(obv_ratio.divide(price_ratio))
        else: raise ValueError(f"Unsupported divergence method: {method}")

        data[col_name] = metric.replace([np.inf, -np.inf], np.nan)

    except Exception as e:
        logger.error(f"Error calculating {col_name}: {e}", exc_info=True)
        data[col_name] = np.nan
    return data


def compute_volume_oscillator(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculates Volume Oscillator."""
    col_name = VOLUME_OSCILLATOR
    if not _check_required_cols(data, ['volume'], col_name):
        data[col_name] = np.nan; return data
    if window < 2:
        logger.error(f"Window ({window}) must be >= 2 for {col_name}. Skipping.")
        data[col_name] = np.nan; return data

    logger.debug(f"Computing {col_name}: window={window}")
    try:
        vol_ma = data['volume'].rolling(window=window, min_periods=max(1, window // 2)).mean()
        denominator = vol_ma.replace(0, np.nan)
        data[col_name] = (data['volume'] - vol_ma).divide(denominator).replace([np.inf, -np.inf], np.nan)
    except Exception as e:
        logger.error(f"Error calculating {col_name}: {e}", exc_info=True)
        data[col_name] = np.nan
    return data


def compute_vwap(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates Volume Weighted Average Price (VWAP) cumulatively."""
    col_name = VWAP
    if not _check_required_cols(data, ['close', 'volume'], col_name):
        data[col_name] = np.nan; return data

    logger.debug(f"Computing {col_name}")
    try:
        safe_volume = data['volume'].clip(lower=0)
        cum_vol = safe_volume.cumsum()
        cum_vol_price = (data['close'] * safe_volume).cumsum()
        data[col_name] = cum_vol_price.divide(cum_vol.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    except Exception as e:
        logger.error(f"Error calculating {col_name}: {e}", exc_info=True)
        data[col_name] = np.nan
    return data


def _compute_volume_index(data: pd.DataFrame, col_name: str, volume_condition: callable) -> pd.DataFrame:
    """Helper for PVI and NVI calculation."""
    if not _check_required_cols(data, ['close', 'volume'], col_name):
        data[col_name] = np.nan; return data

    logger.debug(f"Computing {col_name}")
    try:
        vol_diff = data['volume'].diff()
        price_change_ratio = data['close'].pct_change() # NaN at index 0

        index_series = pd.Series(np.nan, index=data.index, dtype=float)
        index_series.iloc[0] = 1000.0 # Start value

        # Iterative approach for PVI/NVI logic
        last_valid_index = 1000.0
        for i in range(1, len(data)):
            # Start with the previous valid index value
            current_index = index_series.iloc[i-1] if pd.notna(index_series.iloc[i-1]) else last_valid_index

            vol_cond_met = pd.notna(vol_diff.iloc[i]) and volume_condition(vol_diff.iloc[i])
            price_change = price_change_ratio.iloc[i]

            if vol_cond_met:
                if pd.notna(price_change):
                    if abs(price_change) < 10: # Avoid extreme changes
                        current_index = current_index * (1.0 + price_change)
                    else:
                        logger.warning(f"{col_name} calc: Skipping extreme price change ({price_change:.2f}) at index {i}. Using previous {col_name}.")
                        # current_index remains the previous value
                # else: price change is NaN, index remains previous value

            # Update the series and last valid index
            if pd.notna(current_index):
                 last_valid_index = current_index
            index_series.iloc[i] = current_index

        # Forward fill initial NaNs if any, and handle potential infinities
        data[col_name] = index_series.ffill().replace([np.inf, -np.inf], np.nan)

    except Exception as e:
        logger.error(f"Error calculating {col_name}: {e}", exc_info=True)
        data[col_name] = np.nan
    return data

def compute_pvi(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates Positive Volume Index (PVI)."""
    return _compute_volume_index(data, PVI, lambda vol_diff: vol_diff > 0)

def compute_nvi(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates Negative Volume Index (NVI)."""
    return _compute_volume_index(data, NVI, lambda vol_diff: vol_diff < 0)


# --- Apply All Custom Indicators ---
def apply_all_custom_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all defined custom indicators using their default parameters.
    """
    logger.info("Applying custom indicators (using internal defaults)...")
    # Create a copy to avoid modifying the original DataFrame passed to this function
    data_copy = data.copy()
    # Apply functions using their defaults
    data_copy = compute_obv_price_divergence(data_copy)
    data_copy = compute_volume_oscillator(data_copy)
    data_copy = compute_vwap(data_copy)
    data_copy = compute_pvi(data_copy)
    data_copy = compute_nvi(data_copy)
    logger.info("Finished applying custom indicators.")
    return data_copy
