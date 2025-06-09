# custom_indicators.py
import pandas as pd
import numpy as np
import talib as ta
import logging
from typing import List, Optional, Callable, Union, Any

logger = logging.getLogger(__name__)

# --- Column Name Constants ---
OBV_PRICE_DIVERGENCE = 'obv_price_divergence'
VOLUME_OSCILLATOR = 'volume_osc'
VWAP = 'vwap'
PVI = 'pvi'
NVI = 'nvi'

class IndicatorError(Exception):
    """Base exception for indicator calculation errors."""
    pass

class MissingColumnsError(IndicatorError):
    """Raised when required columns are missing."""
    pass

class InvalidParameterError(IndicatorError):
    """Raised when parameters are invalid."""
    pass

class UnsupportedMethodError(IndicatorError):
    """Raised when an unsupported method is specified."""
    pass

# --- Helper for Required Columns ---
def _check_required_cols(data: pd.DataFrame, required_cols: List[str], indicator_name: str) -> None:
    """Check if required columns are present in the data.
    
    Args:
        data: DataFrame to check
        required_cols: List of required column names
        indicator_name: Name of the indicator for error message
        
    Raises:
        MissingColumnsError: If any required columns are missing
    """
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise MissingColumnsError(f"Missing required columns for {indicator_name}: {missing}")

# --- Custom Indicator Functions ---
# These functions now return a DataFrame with ONLY the new column(s) or None on failure.

def compute_obv_price_divergence(data: pd.DataFrame, method: str ="Difference", obv_method: str ="SMA", obv_period: int =14,
                                price_input_type: str ="OHLC/4", price_method: str ="SMA", price_period: int =14,
                                smoothing: float =0.01) -> pd.DataFrame:
    """Calculates OBV/Price divergence.
    
    Args:
        data: DataFrame with OHLCV data
        method: Divergence calculation method ("Difference", "Ratio", "Log Ratio")
        obv_method: OBV smoothing method ("SMA", "EMA", "NONE")
        obv_period: Period for OBV smoothing
        price_input_type: Price series to use ("close", "open", "high", "low", "hl/2", "ohlc/4")
        price_method: Price smoothing method ("SMA", "EMA", "NONE")
        price_period: Period for price smoothing
        smoothing: Smoothing factor for ratio calculations
        
    Returns:
        DataFrame with OBV/Price divergence column
        
    Raises:
        MissingColumnsError: If required columns are missing
        InvalidParameterError: If parameters are invalid
        UnsupportedMethodError: If method is not supported
    """
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    _check_required_cols(data, required_cols, "OBV/Price Divergence")
    
    if not isinstance(obv_period, int) or obv_period < 1:
        raise InvalidParameterError(f"Invalid obv_period: {obv_period}")
    if not isinstance(price_period, int) or price_period < 1:
        raise InvalidParameterError(f"Invalid price_period: {price_period}")
    if not isinstance(smoothing, (int, float)) or smoothing <= 0:
        raise InvalidParameterError(f"Invalid smoothing: {smoothing}")

    logger.debug(f"Computing OBV/Price Divergence: method={method}, obv={obv_method}/{obv_period}, price={price_input_type}/{price_method}/{price_period}")
    result_df = pd.DataFrame(index=data.index)

    try:
        # Select Price Series
        price_map = {
            "close": data['close'], "open": data['open'], "high": data['high'], "low": data['low'],
            "hl/2": (data['high'] + data['low']) / 2,
            "ohlc/4": (data['open'] + data['high'] + data['low'] + data['close']) / 4
        }
        selected_price = price_map.get(price_input_type.lower())
        if selected_price is None:
            raise UnsupportedMethodError(f"Unsupported price input type: {price_input_type}")

        # Calculate OBV and smoothed OBV
        obv = ta.OBV(data['close'], data['volume'])
        obv_ma = obv
        safe_obv_method = str(obv_method).upper() if obv_method else "NONE"
        if safe_obv_method == "SMA":
            obv_ma = ta.SMA(obv, timeperiod=obv_period)
        elif safe_obv_method == "EMA":
            obv_ma = ta.EMA(obv, timeperiod=obv_period)
        elif safe_obv_method != "NONE":
            raise UnsupportedMethodError(f"Unsupported obv_method: {obv_method}")

        # Calculate Smoothed Price
        price_ma = selected_price
        safe_price_method = str(price_method).upper() if price_method else "NONE"
        if safe_price_method == "SMA":
            price_ma = ta.SMA(selected_price, timeperiod=price_period)
        elif safe_price_method == "EMA":
            price_ma = ta.EMA(selected_price, timeperiod=price_period)
        elif safe_price_method != "NONE":
            raise UnsupportedMethodError(f"Unsupported price_method: {price_method}")

        # Calculate Percentage Changes robustly
        obv_change_percent = obv_ma.pct_change(fill_method=None).multiply(100).replace([np.inf, -np.inf], np.nan)
        price_change_percent = price_ma.pct_change(fill_method=None).multiply(100).replace([np.inf, -np.inf], np.nan)

        # Calculate Divergence Metric
        metric = pd.Series(np.nan, index=data.index)
        safe_method = str(method).strip().lower() if method else ""
        
        if safe_method == "difference":
            metric = obv_change_percent - price_change_percent
        elif safe_method == "ratio":
            denominator = price_change_percent.abs().add(max(1e-9, smoothing))
            metric = obv_change_percent.divide(denominator)
        elif safe_method == "log ratio":
            obv_shifted = obv_ma.shift(1).replace(0, np.nan)
            price_shifted = price_ma.shift(1).replace(0, np.nan)
            obv_ratio = obv_ma.divide(obv_shifted).clip(lower=1e-9)
            price_ratio = price_ma.divide(price_shifted).clip(lower=1e-9)
            metric = np.log(obv_ratio.divide(price_ratio.replace(0, np.nan)))
        else:
            raise UnsupportedMethodError(f"Unsupported divergence method: {method}")

        result_df[OBV_PRICE_DIVERGENCE] = metric.replace([np.inf, -np.inf], np.nan)
        return result_df

    except Exception as e:
        if isinstance(e, (MissingColumnsError, InvalidParameterError, UnsupportedMethodError)):
            raise
        logger.error(f"Error calculating OBV/Price Divergence: {e}", exc_info=True)
        raise IndicatorError(f"Failed to calculate OBV/Price Divergence: {e}")

def compute_volume_oscillator(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculates Volume Oscillator.
    
    Args:
        data: DataFrame with volume data
        window: Window size for moving average
        
    Returns:
        DataFrame with Volume Oscillator column
        
    Raises:
        MissingColumnsError: If volume column is missing
        InvalidParameterError: If window is invalid
    """
    _check_required_cols(data, ['volume'], "Volume Oscillator")
    
    if not isinstance(window, int) or window < 2:
        raise InvalidParameterError(f"Window ({window}) must be an integer >= 2")

    logger.debug(f"Computing Volume Oscillator: window={window}")
    result_df = pd.DataFrame(index=data.index)
    
    try:
        vol_ma = data['volume'].rolling(window=window, min_periods=1).mean()
        zero_mask = (data['volume'] == 0) | (vol_ma == 0)
        result_df[VOLUME_OSCILLATOR] = pd.Series(np.nan, index=data.index)
        non_zero_mask = ~zero_mask
        if non_zero_mask.any():
            result_df.loc[non_zero_mask, VOLUME_OSCILLATOR] = (
                (data.loc[non_zero_mask, 'volume'] - vol_ma[non_zero_mask])
                .divide(vol_ma[non_zero_mask])
                .replace([np.inf, -np.inf], np.nan)
            )
        return result_df

    except Exception as e:
        if isinstance(e, (MissingColumnsError, InvalidParameterError)):
            raise
        logger.error(f"Error calculating Volume Oscillator: {e}", exc_info=True)
        raise IndicatorError(f"Failed to calculate Volume Oscillator: {e}")


def compute_vwap(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates Volume Weighted Average Price (VWAP) cumulatively.
    Returns a DataFrame with the result.

    Args:
        data: DataFrame with OHLCV data
    Returns:
        DataFrame with VWAP column
    Raises:
        MissingColumnsError: If required columns are missing
        IndicatorError: For other calculation errors
    """
    required_cols = ['high', 'low', 'close', 'volume']
    _check_required_cols(data, required_cols, 'VWAP')

    logger.debug(f"Computing VWAP")
    result_df = pd.DataFrame(index=data.index)
    try:
        # Use typical price for VWAP
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        safe_volume = data['volume'].clip(lower=0)
        # Calculate cumulative terms
        cum_vol = safe_volume.cumsum()
        cum_vol_price = (typical_price * safe_volume).cumsum()
        # Avoid division by zero
        vwap = cum_vol_price.divide(cum_vol).replace([np.inf, -np.inf], np.nan)
        # Set VWAP to NaN where current volume is zero
        vwap[data['volume'] == 0] = np.nan
        result_df[VWAP] = vwap
        return result_df
    except Exception as e:
        if isinstance(e, MissingColumnsError):
            raise
        logger.error(f"Error calculating VWAP: {e}", exc_info=True)
        raise IndicatorError(f"Failed to calculate VWAP: {e}")


def _compute_volume_index(data: pd.DataFrame, col_name: str, volume_condition: Callable[[float], bool]) -> Optional[pd.DataFrame]:
    """Helper function to compute volume-based indices (PVI/NVI).
    
    Args:
        data: DataFrame with OHLCV data
        col_name: Name of the index column to compute
        volume_condition: Function that takes a volume difference and returns bool
    Returns:
        DataFrame with the computed index column, or None if calculation fails
    """
    logger.debug(f"Computing {col_name}")
    result_df = pd.DataFrame(index=data.index)
    try:
        # If all volume is zero, return all NaN
        if 'volume' in data.columns and (data['volume'] == 0).all():
            result_df[col_name] = np.nan
            return result_df

        # Calculate differences and percentage changes
        vol_diff = data['volume'].diff()
        price_change_ratio = data['close'].pct_change().fillna(0.0)

        index_series = pd.Series(np.nan, index=data.index, dtype=float)
        index_series.iloc[0] = 1000.0  # Start value

        for i in range(1, len(data)):
            prev_index = index_series.iloc[i-1]
            if pd.isna(prev_index):
                logger.error(f"Previous {col_name} is NaN at index {i-1}. Cannot proceed.")
                return None

            current_vol_diff: Union[float, Any] = vol_diff.iloc[i]
            if pd.notna(current_vol_diff) and isinstance(current_vol_diff, (int, float)) and volume_condition(float(current_vol_diff)):
                current_price_change: Union[float, Any] = price_change_ratio.iloc[i]
                if pd.notna(current_price_change) and isinstance(current_price_change, (int, float)) and abs(float(current_price_change)) < 10.0:
                    index_series.iloc[i] = prev_index * (1.0 + float(current_price_change))
                else:
                    if pd.notna(current_price_change) and isinstance(current_price_change, (int, float)):
                        logger.warning(f"{col_name} calc: Skipping extreme price change ({float(current_price_change):.2f}) at index {i}. Using previous {col_name}.")
                    index_series.iloc[i] = prev_index
            else:
                index_series.iloc[i] = prev_index

        result_df[col_name] = index_series.replace([np.inf, -np.inf], np.nan)
        result_df[col_name] = result_df[col_name].ffill()
        return result_df

    except Exception as e:
        logger.error(f"Error calculating {col_name}: {e}", exc_info=True)
        return None

def compute_pvi(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates Positive Volume Index (PVI) cumulatively.
    Returns a DataFrame with the result.

    Args:
        data: DataFrame with OHLCV data
    Returns:
        DataFrame with PVI column
    Raises:
        MissingColumnsError: If required columns are missing
        IndicatorError: For other calculation errors
    """
    required_cols = ['close', 'volume']
    _check_required_cols(data, required_cols, 'PVI')

    logger.debug(f"Computing PVI")
    result_df = _compute_volume_index(data, PVI, lambda x: x > 0)
    if result_df is None:
        raise IndicatorError("Failed to calculate PVI")
    return result_df

def compute_nvi(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates Negative Volume Index (NVI) cumulatively.
    Returns a DataFrame with the result.

    Args:
        data: DataFrame with OHLCV data
    Returns:
        DataFrame with NVI column
    Raises:
        MissingColumnsError: If required columns are missing
        IndicatorError: For other calculation errors
    """
    required_cols = ['close', 'volume']
    _check_required_cols(data, required_cols, 'NVI')

    logger.debug(f"Computing NVI")
    result_df = _compute_volume_index(data, NVI, lambda x: x < 0)
    if result_df is None:
        raise IndicatorError("Failed to calculate NVI")
    return result_df

def compute_returns(data: pd.DataFrame, period: int = 1) -> pd.Series:
    """Compute simple returns for the given period on the 'close' column."""
    # Accept period as int or dict (from JSON params)
    if isinstance(period, dict):
        period = int(period.get('default', 1))
    _check_required_cols(data, ['close'], 'Returns')
    if not isinstance(period, int) or period < 1:
        raise InvalidParameterError(f"Invalid period: {period}")
    if len(data) < period + 1:
        # Not enough data, return a Series of NaN with correct index
        returns = pd.Series([np.nan] * len(data), index=data.index, name=f"returns_{period}")
        return returns
    returns = data['close'].pct_change(periods=period)
    returns.name = f"returns_{period}"
    return returns

def compute_volatility(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """Compute rolling volatility (stddev of returns) for the given period on the 'close' column."""
    if isinstance(period, dict):
        period = int(period.get('default', 20))
    _check_required_cols(data, ['close'], 'Volatility')
    if not isinstance(period, int) or period < 1:
        raise InvalidParameterError(f"Invalid period: {period}")
    if len(data) < period + 1:
        # Not enough data, return a Series of NaN with correct index
        volatility = pd.Series([np.nan] * len(data), index=data.index, name=f"volatility_{period}")
        return volatility
    returns = data['close'].pct_change()
    volatility = returns.rolling(window=period).std()
    volatility.name = f"volatility_{period}"
    return volatility

# --- Removed apply_all_custom_indicators function ---
# This function is no longer needed as main.py ensures custom indicator
# defaults are added to the processing list if necessary.

_custom_indicator_registry = {}

def register_custom_indicator(name, func):
    _custom_indicator_registry[name] = func

def custom_rsi(data, period=14):
    import pandas as pd
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return pd.DataFrame({'CUSTOM_RSI': rsi})