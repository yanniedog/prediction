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
# These functions now return a DataFrame with ONLY the new column(s) or None on failure.

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
        # Handle potential None or empty string for methods
        safe_obv_method = str(obv_method).upper() if obv_method else "NONE"
        if safe_obv_method == "SMA": obv_ma = ta.SMA(obv, timeperiod=obv_period)
        elif safe_obv_method == "EMA": obv_ma = ta.EMA(obv, timeperiod=obv_period)
        elif safe_obv_method != "NONE": raise ValueError(f"Unsupported obv_method: {obv_method}")

        # Calculate Smoothed Price
        price_ma = selected_price
        safe_price_method = str(price_method).upper() if price_method else "NONE"
        if safe_price_method == "SMA": price_ma = ta.SMA(selected_price, timeperiod=price_period)
        elif safe_price_method == "EMA": price_ma = ta.EMA(selected_price, timeperiod=price_period)
        elif safe_price_method != "NONE": raise ValueError(f"Unsupported price_method: {price_method}")

        # Calculate Percentage Changes robustly
        obv_change_percent = obv_ma.pct_change().multiply(100).replace([np.inf, -np.inf], np.nan)
        price_change_percent = price_ma.pct_change().multiply(100).replace([np.inf, -np.inf], np.nan)

        # Calculate Divergence Metric
        metric = pd.Series(np.nan, index=data.index)
        safe_method = str(method).capitalize() if method else ""
        if safe_method == "Difference":
            metric = obv_change_percent - price_change_percent
        elif safe_method == "Ratio":
             # Add epsilon to denominator to avoid division by zero
             denominator = price_change_percent.abs().add(max(1e-9, smoothing))
             metric = obv_change_percent.divide(denominator)
        elif safe_method == "Log Ratio":
             # Clip ratios to avoid log(0) or log(negative) issues
             obv_shifted = obv_ma.shift(1).replace(0, np.nan)
             price_shifted = price_ma.shift(1).replace(0, np.nan)
             obv_ratio = obv_ma.divide(obv_shifted).clip(lower=1e-9)
             price_ratio = price_ma.divide(price_shifted).clip(lower=1e-9)
             # Use np.log1p on (ratio - 1) for better precision near 1? No, simple log is fine.
             metric = np.log(obv_ratio.divide(price_ratio.replace(0, np.nan))) # Avoid div by zero in final step too
        else: raise ValueError(f"Unsupported divergence method: {method}")

        result_df[col_name] = metric.replace([np.inf, -np.inf], np.nan) # Add to new DF
        return result_df

    except Exception as e:
        logger.error(f"Error calculating {col_name}: {e}", exc_info=True)
        return None # Return None on failure


def compute_volume_oscillator(data: pd.DataFrame, window: int = 20) -> Optional[pd.DataFrame]:
    """Calculates Volume Oscillator. Returns a DataFrame with the result or None."""
    col_name = VOLUME_OSCILLATOR
    if not _check_required_cols(data, ['volume'], col_name):
        return None
    if not isinstance(window, int) or window < 2:
        logger.error(f"Window ({window}) must be an integer >= 2 for {col_name}. Skipping.")
        return None

    logger.debug(f"Computing {col_name}: window={window}")
    result_df = pd.DataFrame(index=data.index) # Create new DataFrame
    try:
        # Use min_periods=1 to avoid NaN at the start if window > length
        vol_ma = data['volume'].rolling(window=window, min_periods=1).mean()
        # Avoid division by zero or near-zero
        denominator = vol_ma.replace(0, np.nan)
        result_df[col_name] = (data['volume'] - vol_ma).divide(denominator).replace([np.inf, -np.inf], np.nan) # Add to new DF
        return result_df
    except Exception as e:
        logger.error(f"Error calculating {col_name}: {e}", exc_info=True)
        return None


def compute_vwap(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Calculates Volume Weighted Average Price (VWAP) cumulatively. Returns a DataFrame with the result or None."""
    col_name = VWAP
    if not _check_required_cols(data, ['close', 'volume'], col_name):
        return None

    logger.debug(f"Computing {col_name}")
    result_df = pd.DataFrame(index=data.index) # Create new DataFrame
    try:
        # Ensure volume is non-negative
        safe_volume = data['volume'].clip(lower=0)
        # Calculate cumulative terms
        cum_vol = safe_volume.cumsum()
        cum_vol_price = (data['close'] * safe_volume).cumsum()
        # Avoid division by zero
        result_df[col_name] = cum_vol_price.divide(cum_vol.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan) # Add to new DF
        return result_df
    except Exception as e:
        logger.error(f"Error calculating {col_name}: {e}", exc_info=True)
        return None


def _compute_volume_index(data: pd.DataFrame, col_name: str, volume_condition: callable) -> Optional[pd.DataFrame]:
    """Helper for PVI and NVI calculation. Returns a DataFrame with the result or None."""
    if not _check_required_cols(data, ['close', 'volume'], col_name):
        return None

    logger.debug(f"Computing {col_name}")
    result_df = pd.DataFrame(index=data.index) # Create new DataFrame
    try:
        # Calculate differences and percentage changes
        vol_diff = data['volume'].diff()
        price_change_ratio = data['close'].pct_change().fillna(0.0) # Fill first NaN with 0 change

        index_series = pd.Series(np.nan, index=data.index, dtype=float)
        index_series.iloc[0] = 1000.0 # Start value

        # Iterative approach for PVI/NVI logic
        # Using vectorized operations where possible is faster, but the logic is inherently sequential.
        # The loop remains the clearest way to implement this specific logic.
        for i in range(1, len(data)):
            prev_index = index_series.iloc[i-1]
            if pd.isna(prev_index): # Should not happen after setting iloc[0]
                 logger.error(f"Previous {col_name} is NaN at index {i-1}. Cannot proceed."); return None

            # Check volume condition
            current_vol_diff = vol_diff.iloc[i]
            if pd.notna(current_vol_diff) and volume_condition(current_vol_diff):
                # Apply price change if volume condition met
                current_price_change = price_change_ratio.iloc[i]
                # Avoid extreme changes that might skew the index excessively
                if pd.notna(current_price_change) and abs(current_price_change) < 10.0: # Limit change factor
                    index_series.iloc[i] = prev_index * (1.0 + current_price_change)
                else:
                    # If change is extreme or NaN, keep previous index value
                    if pd.notna(current_price_change):
                         logger.warning(f"{col_name} calc: Skipping extreme price change ({current_price_change:.2f}) at index {i}. Using previous {col_name}.")
                    index_series.iloc[i] = prev_index
            else:
                # If volume condition not met, index remains unchanged
                index_series.iloc[i] = prev_index

        # Handle potential infinities just in case, although checks should prevent them
        result_df[col_name] = index_series.replace([np.inf, -np.inf], np.nan) # Add to new DF
        # Forward fill any remaining NaNs (should only be at the very start if price calc failed)
        result_df[col_name] = result_df[col_name].ffill()

        return result_df

    except Exception as e:
        logger.error(f"Error calculating {col_name}: {e}", exc_info=True)
        return None

def compute_pvi(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Calculates Positive Volume Index (PVI)."""
    return _compute_volume_index(data, PVI, lambda vol_diff: vol_diff > 0)

def compute_nvi(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Calculates Negative Volume Index (NVI)."""
    return _compute_volume_index(data, NVI, lambda vol_diff: vol_diff < 0)


# --- Removed apply_all_custom_indicators function ---
# This function is no longer needed as main.py ensures custom indicator
# defaults are added to the processing list if necessary.