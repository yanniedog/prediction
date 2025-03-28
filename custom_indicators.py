# custom_indicators.py
import pandas as pd
import numpy as np
import talib as ta
import logging

logger = logging.getLogger(__name__)

# --- compute_obv_price_divergence remains the same ---
def compute_obv_price_divergence(data, method="Difference", obv_method="SMA", obv_period=14,
                                 price_input_type="OHLC/4", price_method="SMA", price_period=14,
                                 smoothing=0.01):
    """Calculates OBV/Price divergence. Adds 'obv_price_divergence' column."""
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in data.columns for col in required_cols):
        logger.error("Missing required columns for OBV Price Divergence. Skipping.")
        data['obv_price_divergence'] = np.nan
        return data

    logger.debug(f"Computing OBV Price Divergence: method={method}, obv_method={obv_method}/{obv_period}, price={price_input_type}/{price_method}/{price_period}")
    try:
        if price_input_type.lower() == "close": selected_price = data['close']
        elif price_input_type.lower() == "open": selected_price = data['open']
        elif price_input_type.lower() == "high": selected_price = data['high']
        elif price_input_type.lower() == "low": selected_price = data['low']
        elif price_input_type.lower() == "hl/2": selected_price = (data['high'] + data['low']) / 2
        elif price_input_type.lower() == "ohlc/4": selected_price = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        else: raise ValueError(f"Unsupported price input type: {price_input_type}")

        obv = ta.OBV(data['close'], data['volume'])
        if obv_method == "SMA": obv_ma = ta.SMA(obv, timeperiod=obv_period)
        elif obv_method == "EMA": obv_ma = ta.EMA(obv, timeperiod=obv_period)
        else: obv_ma = obv

        if price_method == "SMA": price_ma = ta.SMA(selected_price, timeperiod=price_period)
        elif price_method == "EMA": price_ma = ta.EMA(selected_price, timeperiod=price_period)
        else: price_ma = selected_price

        obv_ma_shifted = obv_ma.shift(1).replace(0, np.nan)
        price_ma_shifted = price_ma.shift(1).replace(0, np.nan)
        obv_change_percent = ((obv_ma - obv_ma_shifted) / obv_ma_shifted) * 100
        price_change_percent = ((price_ma - price_ma_shifted) / price_ma_shifted) * 100

        metric = np.nan
        if method == "Difference": metric = obv_change_percent - price_change_percent
        elif method == "Ratio": denominator = np.maximum(smoothing, np.abs(price_change_percent)); metric = obv_change_percent / denominator
        elif method == "Log Ratio":
            numerator = np.maximum(smoothing, np.abs(obv_change_percent)); denominator = np.maximum(smoothing, np.abs(price_change_percent))
            safe_numerator = np.where(numerator <= 0, np.nan, numerator); safe_denominator = np.where(denominator <= 0, np.nan, denominator)
            metric = np.log(safe_numerator / safe_denominator)
        else: raise ValueError(f"Unsupported divergence method: {method}")
        data['obv_price_divergence'] = metric
    except Exception as e: logger.error(f"Error calculating OBV Price Divergence: {e}", exc_info=True); data['obv_price_divergence'] = np.nan
    return data

# --- compute_volume_oscillator remains the same ---
def compute_volume_oscillator(data, window=20):
    """Calculates Volume Oscillator. Adds 'volume_osc' column."""
    logger.debug(f"Computing Volume Oscillator: window={window}")
    if 'volume' not in data.columns: logger.error("Missing 'volume' column for Vol Osc. Skipping."); data['volume_osc'] = np.nan; return data
    try:
        vol_ma = data['volume'].rolling(window=window).mean(); denominator = vol_ma.replace(0, np.nan)
        data['volume_osc'] = (data['volume'] - vol_ma) / denominator
    except Exception as e: logger.error(f"Error calculating Volume Oscillator: {e}", exc_info=True); data['volume_osc'] = np.nan
    return data

# --- compute_vwap remains the same ---
def compute_vwap(data):
    """Calculates Volume Weighted Average Price. Adds 'vwap' column."""
    logger.debug("Computing VWAP")
    if not all(col in data.columns for col in ['close', 'volume']): logger.error("Missing 'close' or 'volume' for VWAP. Skipping."); data['vwap'] = np.nan; return data
    try:
        cum_vol = data['volume'].cumsum(); cum_vol_price = (data['close'] * data['volume']).cumsum()
        data['vwap'] = cum_vol_price / cum_vol.replace(0, np.nan)
    except Exception as e: logger.error(f"Error calculating VWAP: {e}", exc_info=True); data['vwap'] = np.nan
    return data

# --- UPDATED compute_pvi ---
def compute_pvi(data):
    """Calculates Positive Volume Index. Adds 'pvi' column. More robust."""
    logger.debug("Computing PVI")
    if not all(col in data.columns for col in ['close', 'volume']):
        logger.error("Missing 'close' or 'volume' for PVI. Skipping.")
        data['pvi'] = np.nan
        return data
    try:
        vol_diff = data['volume'].diff()
        price_change_ratio = data['close'].pct_change() # Note: This will have NaN at index 0

        pvi = pd.Series(np.nan, index=data.index)
        pvi.iloc[0] = 1000.0 # Start with float

        # Iterate and calculate, using previous *valid* PVI if current calculation yields NaN
        for i in range(1, len(data)):
            last_valid_pvi = pvi.iloc[i-1] # Get previous value first

            current_pvi = last_valid_pvi # Default to previous value
            if pd.notna(vol_diff.iloc[i]) and vol_diff.iloc[i] > 0:
                # Only update if volume increased
                if pd.notna(price_change_ratio.iloc[i]) and pd.notna(last_valid_pvi):
                    current_pvi = last_valid_pvi * (1.0 + price_change_ratio.iloc[i])

            # Assign the calculated (or carried forward) value
            pvi.iloc[i] = current_pvi

        # Forward fill any remaining NaNs (e.g., if price_change_ratio was NaN)
        pvi.ffill(inplace=True)
        data['pvi'] = pvi.replace([np.inf, -np.inf], np.nan)

    except Exception as e:
        logger.error(f"Error calculating PVI: {e}", exc_info=True)
        data['pvi'] = np.nan
    return data

# --- UPDATED compute_nvi ---
def compute_nvi(data):
    """Calculates Negative Volume Index. Adds 'nvi' column. More robust."""
    logger.debug("Computing NVI")
    if not all(col in data.columns for col in ['close', 'volume']):
        logger.error("Missing 'close' or 'volume' for NVI. Skipping.")
        data['nvi'] = np.nan
        return data
    try:
        vol_diff = data['volume'].diff()
        price_change_ratio = data['close'].pct_change() # NaN at index 0

        nvi = pd.Series(np.nan, index=data.index)
        nvi.iloc[0] = 1000.0 # Start with float

        # Iterate and calculate
        for i in range(1, len(data)):
            last_valid_nvi = nvi.iloc[i-1]

            current_nvi = last_valid_nvi # Default to previous value
            if pd.notna(vol_diff.iloc[i]) and vol_diff.iloc[i] < 0:
                # Only update if volume decreased
                if pd.notna(price_change_ratio.iloc[i]) and pd.notna(last_valid_nvi):
                    current_nvi = last_valid_nvi * (1.0 + price_change_ratio.iloc[i])

            nvi.iloc[i] = current_nvi

        # Forward fill any remaining NaNs
        nvi.ffill(inplace=True)
        data['nvi'] = nvi.replace([np.inf, -np.inf], np.nan)

    except Exception as e:
        logger.error(f"Error calculating NVI: {e}", exc_info=True)
        data['nvi'] = np.nan
    return data

# --- apply_all_custom_indicators remains the same ---
def apply_all_custom_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Applies all defined custom indicators to the DataFrame."""
    logger.info("Applying custom indicators...")
    # Call each custom function. Pass data, it adds column inplace (or returns modified df)
    # The function calls internally use the parameters defined within them (defaults).
    # If JSON parameters were meant to be passed, this would need modification.
    data = compute_obv_price_divergence(data)
    data = compute_volume_oscillator(data)
    data = compute_vwap(data)
    data = compute_pvi(data)
    data = compute_nvi(data)
    logger.info("Finished applying custom indicators.")
    return data