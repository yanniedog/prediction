# custom_indicators.py
import pandas as pd
import numpy as np
import talib as ta
import logging
from typing import List, Optional # Added Optional import

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
                                 smoothing: float =0.01) -> Optional[pd.DataFrame]:
    """Calculates OBV/Price divergence. Returns a DataFrame with the result or None."""
    col_name = OBV_PRICE_DIVERGENCE
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not _check_required_cols(data, required_cols, col_name):
        return None # Return None on failure

    logger.debug(f"Computing {col_name}: method={method}, obv={obv_method}/{obv_period}, price={price_input_type}/{price_method}/{price_period}")
    result_df = pd.DataFrame(index=data.index) # Create new DataFrame

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
        elif obv_method is not None and str(obv_method).lower() != "none": raise ValueError(f"Unsupported obv_method: {obv_method}") # Added str()

        # Calculate Smoothed Price
        price_ma = selected_price
        if price_method == "SMA": price_ma = ta.SMA(selected_price, timeperiod=price_period)
        elif price_method == "EMA": price_ma = ta.EMA(selected_price, timeperiod=price_period)
        elif price_method is not None and str(price_method).lower() != "none": raise ValueError(f"Unsupported price_method: {price_method}") # Added str()

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

        result_df[col_name] = metric.replace([np.inf, -np.inf], np.nan) # Add to new DF
        return result_df

    except Exception as e:
        logger.error(f"Error calculating {col_name}: {e}", exc_info=True)
        # data[col_name] = np.nan # Don't modify input
        return None # Return None on failure


def compute_volume_oscillator(data: pd.DataFrame, window: int = 20) -> Optional[pd.DataFrame]:
    """Calculates Volume Oscillator. Returns a DataFrame with the result or None."""
    col_name = VOLUME_OSCILLATOR
    if not _check_required_cols(data, ['volume'], col_name):
        return None
    if window < 2:
        logger.error(f"Window ({window}) must be >= 2 for {col_name}. Skipping.")
        return None

    logger.debug(f"Computing {col_name}: window={window}")
    result_df = pd.DataFrame(index=data.index) # Create new DataFrame
    try:
        vol_ma = data['volume'].rolling(window=window, min_periods=max(1, window // 2)).mean()
        denominator = vol_ma.replace(0, np.nan)
        result_df[col_name] = (data['volume'] - vol_ma).divide(denominator).replace([np.inf, -np.inf], np.nan) # Add to new DF
        return result_df
    except Exception as e:
        logger.error(f"Error calculating {col_name}: {e}", exc_info=True)
        # data[col_name] = np.nan # Don't modify input
        return None


def compute_vwap(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Calculates Volume Weighted Average Price (VWAP) cumulatively. Returns a DataFrame with the result or None."""
    col_name = VWAP
    if not _check_required_cols(data, ['close', 'volume'], col_name):
        return None

    logger.debug(f"Computing {col_name}")
    result_df = pd.DataFrame(index=data.index) # Create new DataFrame
    try:
        safe_volume = data['volume'].clip(lower=0)
        cum_vol = safe_volume.cumsum()
        cum_vol_price = (data['close'] * safe_volume).cumsum()
        result_df[col_name] = cum_vol_price.divide(cum_vol.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan) # Add to new DF
        return result_df
    except Exception as e:
        logger.error(f"Error calculating {col_name}: {e}", exc_info=True)
        # data[col_name] = np.nan # Don't modify input
        return None


def _compute_volume_index(data: pd.DataFrame, col_name: str, volume_condition: callable) -> Optional[pd.DataFrame]:
    """Helper for PVI and NVI calculation. Returns a DataFrame with the result or None."""
    if not _check_required_cols(data, ['close', 'volume'], col_name):
        return None

    logger.debug(f"Computing {col_name}")
    result_df = pd.DataFrame(index=data.index) # Create new DataFrame
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
        result_df[col_name] = index_series.ffill().replace([np.inf, -np.inf], np.nan) # Add to new DF
        return result_df

    except Exception as e:
        logger.error(f"Error calculating {col_name}: {e}", exc_info=True)
        # data[col_name] = np.nan # Don't modify input
        return None

def compute_pvi(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Calculates Positive Volume Index (PVI)."""
    return _compute_volume_index(data, PVI, lambda vol_diff: vol_diff > 0)

def compute_nvi(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Calculates Negative Volume Index (NVI)."""
    return _compute_volume_index(data, NVI, lambda vol_diff: vol_diff < 0)


# --- Apply All Custom Indicators ---
# Note: This function might be less relevant now, as indicator computation
# is primarily driven by configurations in main.py. Keep for testing?
def apply_all_custom_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all defined custom indicators using their default parameters.
    Returns a NEW DataFrame with original data + new columns.
    """
    logger.info("Applying custom indicators (using internal defaults)...")
    data_copy = data.copy() # Work on a copy
    all_custom_dfs = []

    # Call each function, they should now return a DataFrame or None
    df_obv_div = compute_obv_price_divergence(data_copy)
    if df_obv_div is not None: all_custom_dfs.append(df_obv_div)

    df_vol_osc = compute_volume_oscillator(data_copy)
    if df_vol_osc is not None: all_custom_dfs.append(df_vol_osc)

    df_vwap = compute_vwap(data_copy)
    if df_vwap is not None: all_custom_dfs.append(df_vwap)

    df_pvi = compute_pvi(data_copy)
    if df_pvi is not None: all_custom_dfs.append(df_pvi)

    df_nvi = compute_nvi(data_copy)
    if df_nvi is not None: all_custom_dfs.append(df_nvi)

    if all_custom_dfs:
        logger.info(f"Concatenating {len(all_custom_dfs)} custom indicator results.")
        try:
            # Concatenate results onto the original data copy
            data_with_custom = pd.concat([data_copy] + all_custom_dfs, axis=1)
            logger.info(f"Finished applying custom indicators. Shape: {data_with_custom.shape}")
            return data_with_custom
        except Exception as e:
            logger.error(f"Error concatenating custom indicators: {e}", exc_info=True)
            return data_copy # Return original copy on error
    else:
        logger.warning("No custom indicators successfully computed.")
        return data_copy