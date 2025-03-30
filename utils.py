# utils.py
import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
import re
import hashlib
import json
from datetime import datetime, timedelta, timezone
import math

try:
    from dateutil.relativedelta import relativedelta
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False
    # Less critical warning now, mainly needed for date calculations
    logging.getLogger(__name__).info("python-dateutil missing. Month timeframe calculations ('1M') may fail.")

logger = logging.getLogger(__name__)

def clear_screen() -> None:
    """Clears the terminal screen."""
    # Add basic error handling for environments where clear might fail
    try:
        os.system('cls' if os.name == 'nt' else 'clear')
    except Exception:
        pass # Ignore if clear fails

def parse_indicator_column_name(col_name: str) -> Optional[Tuple[str, int, Optional[str]]]:
    """
    Parses indicator column names like 'MACD_10_0', 'AD_29', 'KC_50_LOWER'.
    Assumes Config ID is the LAST numeric part preceded by '_'.
    Handles cases like 'ADX_14' or 'RSI_14'.
    """
    # Updated regex to be more robust:
    # ^(.*?)             # Group 1: Base name (non-greedy)
    # _(\d+)             # Group 2: Config ID (must have underscore before)
    # (?:_([^_]*))?       # Group 3: Optional suffix (underscore + non-underscore chars)
    # $                   # End of string
    match = re.match(r'^(.*?)_(\d+)(?:_([^_]+))?$', col_name)
    if match:
        base_name, config_id_str, output_suffix = match.groups()
        # Handle cases where base_name might end with digits due to non-greedy match
        # Example: indicator10_123 -> base should be indicator10, not indicator
        # This case is less likely with the current naming scheme but good to consider
        # if naming convention changes. For now, the regex is likely sufficient.

        try:
            return base_name, int(config_id_str), output_suffix
        except ValueError:
            logger.error(f"Internal Error: Regex matched digits, but conversion failed for '{col_name}'")
            return None
    else:
        # Handle simple cases like 'RSI_14' where there's no suffix group
        match_simple = re.match(r'^(.*?)_(\d+)$', col_name)
        if match_simple:
            base_name, config_id_str = match_simple.groups()
            try:
                return base_name, int(config_id_str), None # No suffix
            except ValueError:
                 logger.error(f"Internal Error: Regex matched digits, but conversion failed for '{col_name}'")
                 return None

        logger.debug(f"Column name '{col_name}' did not match pattern `base_name_configID[_suffix]` or `base_name_configID`.")
        return None


def get_config_identifier(indicator_name: str, config_id: int, output_index_or_suffix: Optional[Any]) -> str:
    """Creates a consistent identifier string for an indicator output column."""
    clean_name = str(indicator_name).strip().replace(' ','_') # Sanitize name
    name = f"{clean_name}_{config_id}"
    if output_index_or_suffix is not None:
        # Sanitize suffix: replace spaces, convert to string
        clean_suffix = str(output_index_or_suffix).strip().replace(' ','_')
        if clean_suffix:
            # Avoid double underscores if suffix already starts with one
            if clean_suffix.startswith('_'):
                name += clean_suffix
            else:
                name += f"_{clean_suffix}"
    return name

def get_config_hash(params: Dict[str, Any]) -> str:
    """Generates a stable SHA256 hash for a parameter dictionary."""
    try:
        # Ensure consistent float representation (e.g., avoid 1.0 vs 1)
        # We can round floats to a reasonable precision before hashing
        def round_floats(obj):
            if isinstance(obj, float):
                return round(obj, 8) # Round floats to 8 decimal places for hashing
            if isinstance(obj, dict):
                return {k: round_floats(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [round_floats(elem) for elem in obj]
            return obj

        params_rounded = round_floats(params)
        config_str = json.dumps(params_rounded, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(config_str.encode('utf-8')).hexdigest()
    except TypeError as e:
        logger.error(f"Error creating JSON for hashing parameters: {params}. Error: {e}")
        try:
            # Fallback hashing (less reliable) - attempt sorting items robustly
            items_repr = []
            for k, v in sorted(params.items()):
                 # Try to create a stable representation for common types
                 if isinstance(v, (dict, list, tuple)):
                     try: items_repr.append(f"{k}:{json.dumps(v, sort_keys=True)}")
                     except: items_repr.append(f"{k}:{repr(v)}") # Fallback repr
                 else: items_repr.append(f"{k}:{repr(v)}")
            fallback_str = ";".join(items_repr)

            logger.warning(f"Using fallback hashing method for params: {params}")
            return hashlib.sha256(fallback_str.encode('utf-8')).hexdigest()
        except Exception as fallback_e:
            logger.critical(f"CRITICAL: Failed fallback hashing for params {params}: {fallback_e}")
            raise TypeError(f"Cannot hash parameters: {params}") from e

def compare_param_dicts(dict1: Optional[Dict], dict2: Optional[Dict]) -> bool:
    """Compares two parameter dictionaries for equality, handling float precision."""
    if dict1 is None and dict2 is None: return True
    if dict1 is None or dict2 is None: return False
    if not isinstance(dict1, dict) or not isinstance(dict2, dict): return False
    if dict1.keys() != dict2.keys(): return False

    for key in dict1:
        val1, val2 = dict1[key], dict2[key]
        if isinstance(val1, float) and isinstance(val2, float):
            # Use numpy isclose for robust float comparison
            if not np.isclose(val1, val2, rtol=1e-6, atol=1e-9): return False # Slightly tighter tolerance
        elif isinstance(val1, (np.floating, np.integer)) or isinstance(val2, (np.floating, np.integer)):
             # Handle numpy types by converting to standard Python types if possible
             try:
                 py_val1 = val1.item() if hasattr(val1, 'item') else val1
                 py_val2 = val2.item() if hasattr(val2, 'item') else val2
                 if isinstance(py_val1, float) and isinstance(py_val2, float):
                      if not np.isclose(py_val1, py_val2, rtol=1e-6, atol=1e-9): return False
                 elif py_val1 != py_val2: return False
             except: # Fallback if conversion fails
                 if val1 != val2: return False
        elif type(val1) != type(val2):
             # Allow comparing None or int/float that might look different
             if (val1 is None and val2 is not None) or (val1 is not None and val2 is None): return False
             if val1 is None and val2 is None: continue # Both None is equal
             try: # Attempt numeric comparison if one is int/float
                 num1 = float(val1); num2 = float(val2)
                 if not np.isclose(num1, num2, rtol=1e-6, atol=1e-9): return False
             except (ValueError, TypeError): # If conversion fails, types are truly different
                 return False
        elif val1 != val2: return False # Direct comparison for other types
    return True

# --- Time Formatting Utility ---
def format_duration(duration: timedelta) -> str:
    """Formats a timedelta duration into a human-readable string."""
    total_seconds = duration.total_seconds()

    if total_seconds < 0:
        return f"-{format_duration(-duration)}"

    if total_seconds < 60:
        return f"{total_seconds:.1f}s"

    days = duration.days
    seconds_remaining = total_seconds - (days * 86400)
    hours = int(seconds_remaining // 3600)
    seconds_remaining -= hours * 3600
    minutes = int(seconds_remaining // 60)
    seconds = int(seconds_remaining % 60)

    parts = []
    if days > 0: parts.append(f"{days}d")
    if hours > 0: parts.append(f"{hours}h")
    if minutes > 0: parts.append(f"{minutes}m")
    if seconds > 0 or not parts: # Always show seconds if no other parts or if non-zero
        parts.append(f"{seconds}s")

    return " ".join(parts)

# --- Timeframe Calculation Helper Functions ---
def _get_seconds_per_period(timeframe: str) -> Optional[float]:
    """Helper to get approximate seconds per period for common Binance timeframes."""
    tf_lower = timeframe.lower()
    # Handle cases like '1M', '1w', '3d', '12h', '5m'
    match = re.match(r'(\d+)([mhdwy])$', tf_lower) # Added 'y' for year potential? (Though Binance doesn't use)
    if match:
        try:
            value = int(match.group(1))
            unit = match.group(2)
            if value <= 0: raise ValueError("Timeframe value must be positive")

            if unit == 'y': return float(value * 31536000) # Approx seconds in a year
            elif unit == 'w': return float(value * 604800) # Seconds in a week
            elif unit == 'd': return float(value * 86400) # Seconds in a day
            elif unit == 'h': return float(value * 3600) # Seconds in an hour
            elif unit == 'm':
                 # Binance specific '1M' (Month) vs '1m' (minute)
                 if timeframe == '1M': # Check original case for Binance month
                     return None # Indicate month needs special handling
                 return float(value * 60) # Seconds in a minute
            else: raise ValueError(f"Unknown unit {unit}") # Should not happen with regex

        except (ValueError, KeyError) as e:
             logger.error(f"Invalid value/unit in timeframe '{timeframe}': {e}")
             return None
    elif timeframe == '1M': # Explicitly catch Binance month timeframe again
         return None
    else:
        logger.error(f"Unrecognized timeframe format: {timeframe}")
        return None

def calculate_periods_between_dates(start_date: datetime, end_date: datetime, timeframe: str) -> Optional[int]:
    """Estimates the number of periods between two dates based on the timeframe."""
    if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
        logger.error("Invalid date types for period calculation."); return None
    # Ensure timezone aware comparison (assume UTC if naive)
    start_utc = start_date if start_date.tzinfo else start_date.replace(tzinfo=timezone.utc)
    end_utc = end_date if end_date.tzinfo else end_date.replace(tzinfo=timezone.utc)

    if end_utc <= start_utc: return 0

    if timeframe == '1M': # Handle Month ('1M')
        if not DATEUTIL_AVAILABLE: logger.error("Cannot calc month periods: python-dateutil missing."); return None
        try:
            # Use relativedelta for accurate month calculation
            delta = relativedelta(end_utc, start_utc)
            # Consider partial months as a full period for ceiling effect
            total_months = delta.years * 12 + delta.months
            # If there's any leftover time beyond full months, add one period
            if delta.days > 0 or delta.hours > 0 or delta.minutes > 0 or delta.seconds > 0 or delta.microseconds > 0:
                 periods = total_months + 1
            else:
                 periods = total_months
            # Ensure at least 1 period if end > start but difference < 1 month
            if periods == 0 and end_utc > start_utc:
                periods = 1
            logger.debug(f"Calculated {periods} month periods between {start_utc} and {end_utc}.")
            return periods
        except Exception as e: logger.error(f"Error calc month periods: {e}", exc_info=True); return None

    # Handle other timeframes using seconds
    period_seconds = _get_seconds_per_period(timeframe)
    if period_seconds is None: logger.error(f"Cannot get seconds/period for '{timeframe}'."); return None
    total_seconds = (end_utc - start_utc).total_seconds()
    if total_seconds <= 0: return 0
    # Use ceiling to include partial periods at the end
    periods = math.ceil(total_seconds / period_seconds)
    logger.debug(f"Calculated {periods} periods for '{timeframe}' ({total_seconds:.2f}s / {period_seconds}s/prd).")
    return periods

def estimate_days_in_periods(periods: int, timeframe: str) -> Optional[float]:
    """Estimates approximate calendar days covered by N periods."""
    if not isinstance(periods, int) or periods < 0: logger.error(f"Invalid periods ({periods})"); return None
    if periods == 0: return 0.0

    if timeframe == '1M':
        avg_days_month = 365.2425 / 12.0 # Average days in a month
        est_days = periods * avg_days_month
        logger.debug(f"Estimated {est_days:.2f} days for {periods} months.")
        return est_days

    period_seconds = _get_seconds_per_period(timeframe)
    if period_seconds is None: logger.error(f"Cannot get seconds/period for '{timeframe}'."); return None
    # Calculate total seconds and convert to days
    est_days = (periods * period_seconds) / 86400.0
    logger.debug(f"Estimated {est_days:.2f} days for {periods} periods of '{timeframe}'.")
    return est_days

def estimate_future_date(start_date: datetime, periods: int, timeframe: str) -> Optional[datetime]:
    """Estimates the future date after N periods."""
    if not isinstance(start_date, datetime): logger.error("Invalid start_date."); return None
    if not isinstance(periods, int) or periods < 0: logger.error(f"Invalid periods ({periods})."); return None
    if periods == 0: return start_date

    # Ensure start date is timezone aware (assume UTC if naive)
    tz_info = start_date.tzinfo if start_date.tzinfo else timezone.utc
    start_date_aware = start_date.replace(tzinfo=tz_info)

    try:
        if timeframe == '1M': # Use relativedelta for accurate month addition
            if not DATEUTIL_AVAILABLE: logger.error("Cannot estimate future month date: dateutil missing."); return None
            future_date = start_date_aware + relativedelta(months=periods)
            logger.debug(f"Estimated future date for {periods} months: {future_date}"); return future_date

        # Other timeframes use seconds
        period_seconds = _get_seconds_per_period(timeframe)
        if period_seconds is None: logger.error(f"Cannot get seconds/period for '{timeframe}'."); return None
        future_date = start_date_aware + timedelta(seconds=periods * period_seconds)
        logger.debug(f"Estimated future date for {periods} periods of '{timeframe}': {future_date}")
        return future_date
    except OverflowError: logger.error(f"Date calculation overflow: {periods} periods of {timeframe} from {start_date}."); return None
    except Exception as e: logger.error(f"Error calculating future date: {e}", exc_info=True); return None