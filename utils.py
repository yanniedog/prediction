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
    logging.warning("python-dateutil missing. Month timeframe calculations ('1M') may fail.")

logger = logging.getLogger(__name__)

def clear_screen() -> None:
    """Clears the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def parse_indicator_column_name(col_name: str) -> Optional[Tuple[str, int, Optional[str]]]:
    """
    Parses indicator column names like 'MACD_10_0', 'AD_29', 'KC_50_LOWER'.
    Assumes Config ID is the LAST numeric part preceded by '_'.
    """
    match = re.match(r'^(.*?)_(\d+)(?:_(.+))?$', col_name)
    if match:
        base_name, config_id_str, output_suffix = match.groups()
        try:
            return base_name, int(config_id_str), output_suffix
        except ValueError:
            logger.error(f"Internal Error: Regex matched digits, but conversion failed for '{col_name}'")
            return None
    else:
        logger.debug(f"Column name '{col_name}' did not match pattern `base_name_configID[_suffix]`.")
        return None

def get_config_identifier(indicator_name: str, config_id: int, output_index_or_suffix: Optional[Any]) -> str:
    """Creates a consistent identifier string for an indicator output column."""
    clean_name = str(indicator_name).strip()
    name = f"{clean_name}_{config_id}"
    if output_index_or_suffix is not None:
        clean_suffix = str(output_index_or_suffix).strip()
        if clean_suffix:
            name += f"_{clean_suffix}"
    return name

def get_config_hash(params: Dict[str, Any]) -> str:
    """Generates a stable SHA256 hash for a parameter dictionary."""
    try:
        config_str = json.dumps(params, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(config_str.encode('utf-8')).hexdigest()
    except TypeError as e:
        logger.error(f"Error creating JSON for hashing parameters: {params}. Error: {e}")
        try:
            # Fallback hashing (less reliable)
            fallback_str = str(sorted(params.items()))
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
            if not np.isclose(val1, val2, rtol=1e-5, atol=1e-8): return False
        elif type(val1) != type(val2):
             # Allow None comparisons specifically
            if (val1 is None and val2 is not None) or (val1 is not None and val2 is None): return False
            if val1 is None and val2 is None: continue # Both None is equal
            return False # Different types otherwise
        elif val1 != val2: return False # Direct comparison for non-floats
    return True

# --- Timeframe Calculation Helper Functions ---
def _get_seconds_per_period(timeframe: str) -> Optional[float]:
    """Helper to get approximate seconds per period for common Binance timeframes."""
    match = re.match(r'(\d+)([mhdw])$', timeframe.lower())
    if match:
        try:
            value = int(match.group(1)); unit = match.group(2)
            if value <= 0: raise ValueError("Timeframe value must be positive")
            multipliers = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800}
            return float(value * multipliers[unit])
        except (ValueError, KeyError):
             logger.error(f"Invalid value/unit in timeframe '{timeframe}'")
             return None
    elif timeframe == '1M': return None # Month handled separately
    else: logger.error(f"Unrecognized timeframe format: {timeframe}"); return None

def calculate_periods_between_dates(start_date: datetime, end_date: datetime, timeframe: str) -> Optional[int]:
    """Estimates the number of periods between two dates based on the timeframe."""
    if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
        logger.error("Invalid date types for period calculation."); return None
    if start_date.tzinfo is None: start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None: end_date = end_date.replace(tzinfo=timezone.utc)
    if end_date <= start_date: return 0

    if timeframe == '1M': # Handle Month ('1M')
        if not DATEUTIL_AVAILABLE: logger.error("Cannot calc month periods: python-dateutil missing."); return None
        try:
            delta = relativedelta(end_date, start_date)
            total_months = delta.years * 12 + delta.months
            # If any remaining time, count as next period
            has_remainder = delta.days > 0 or delta.hours > 0 or delta.minutes > 0 or delta.seconds > 0 or delta.microseconds > 0
            periods = total_months + 1 if has_remainder else total_months
            # Handle case where difference is less than a month but non-zero
            if periods == 0 and total_months == 0 and has_remainder: periods = 1
            logger.debug(f"Calculated {periods} month periods between {start_date} and {end_date}.")
            return max(1, periods) if end_date > start_date else 0
        except Exception as e: logger.error(f"Error calc month periods: {e}", exc_info=True); return None

    # Handle other timeframes using seconds
    period_seconds = _get_seconds_per_period(timeframe)
    if period_seconds is None: logger.error(f"Cannot get seconds/period for '{timeframe}'."); return None
    total_seconds = (end_date - start_date).total_seconds()
    if total_seconds <= 0: return 0
    periods = math.ceil(total_seconds / period_seconds) # Use ceiling division
    logger.debug(f"Calculated {periods} periods for '{timeframe}' ({total_seconds:.2f}s / {period_seconds}s/prd).")
    return periods

def estimate_days_in_periods(periods: int, timeframe: str) -> Optional[float]:
    """Estimates approximate calendar days covered by N periods."""
    if not isinstance(periods, int) or periods < 0: logger.error(f"Invalid periods ({periods})"); return None
    if periods == 0: return 0.0

    if timeframe == '1M':
        avg_days_month = 365.2425 / 12.0; est_days = periods * avg_days_month
        logger.debug(f"Estimated {est_days:.2f} days for {periods} months.")
        return est_days

    period_seconds = _get_seconds_per_period(timeframe)
    if period_seconds is None: logger.error(f"Cannot get seconds/period for '{timeframe}'."); return None
    est_days = (periods * period_seconds) / 86400.0
    logger.debug(f"Estimated {est_days:.2f} days for {periods} periods of '{timeframe}'.")
    return est_days

def estimate_future_date(start_date: datetime, periods: int, timeframe: str) -> Optional[datetime]:
    """Estimates the future date after N periods."""
    if not isinstance(start_date, datetime): logger.error("Invalid start_date."); return None
    if not isinstance(periods, int) or periods < 0: logger.error(f"Invalid periods ({periods})."); return None
    if periods == 0: return start_date

    tz_info = start_date.tzinfo if start_date.tzinfo else timezone.utc
    start_date_aware = start_date.replace(tzinfo=tz_info)

    try:
        if timeframe == '1M': # Use relativedelta for months
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
