# utils.py
import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
import re # Import re
import hashlib # Import hashlib
import json # Import json

logger = logging.getLogger(__name__)

def clear_screen() -> None:
    """Clears the terminal screen."""
    # Use standard library method for cross-platform compatibility
    # Works on Linux/macOS/Windows (in modern terminals)
    os.system('cls' if os.name == 'nt' else 'clear')

def determine_time_interval(data: pd.DataFrame) -> Optional[str]:
    """Determines the most common time interval in the data based on 'date' column."""
    if 'date' not in data.columns:
        logger.error("No 'date' column found for determining time interval.")
        return None
    if len(data) < 2:
        logger.warning("Insufficient data (less than 2 rows) to determine time interval.")
        return None

    # Ensure date column is suitable
    date_col = data['date']
    if not pd.api.types.is_datetime64_any_dtype(date_col):
        try:
            date_col = pd.to_datetime(date_col, errors='coerce')
        except Exception as e:
             logger.error(f"Error converting date column to datetime: {e}")
             return None

    date_col = date_col.dropna()
    if len(date_col) < 2:
        logger.warning("Not enough valid dates after dropna to determine interval.")
        return None

    # Calculate time differences in seconds
    # Sort first to ensure correct diff calculation if data wasn't pre-sorted
    time_diffs = date_col.sort_values().diff().dt.total_seconds().dropna()

    if time_diffs.empty:
        logger.warning("No time differences found after processing.")
        return None

    try:
        # Use median for robustness against outliers
        median_diff = time_diffs.median()
        logger.debug(f"Median time difference: {median_diff} seconds.")

        if median_diff <= 0:
            logger.warning(f"Median time difference is not positive ({median_diff}). Cannot determine interval.")
            return None

        # Define thresholds (adjust slightly for edge cases, e.g., 59.9s)
        # Using common intervals
        if median_diff < 60: return 'seconds'          # Less than 1 min
        if median_diff < 3570: return 'minutes'        # Less than 59.5 min
        if median_diff < 86100: return 'hours'         # Less than 23.9 hours
        if median_diff < 603000: return 'days'         # Less than 6.98 days (handle weekly slightly off)
        if median_diff < 2580000: return 'weeks'       # Less than ~29.8 days
        if median_diff < 31500000: return 'months'     # Less than ~364 days
        return 'years'

    except Exception as e:
        logger.error(f"Error determining time interval median: {e}", exc_info=True)
        return None

def parse_indicator_column_name(col_name: str) -> Optional[Tuple[str, int, Optional[str]]]:
    """
    Parses indicator column names like 'MACD_10_0', 'AD_29', 'KC_50_LOWER', 'VORTEX_70_PLUS'.
    Handles base names that might contain underscores.

    Returns: Tuple(base_indicator_name, config_id, output_suffix|None)
             Returns None if parsing fails.
             output_suffix can be numeric ('0', '1') or string ('LOWER', 'PLUS').
    """
    # Regex tries to capture the pattern: (Anything)_(Digits)(Optional_[AnythingElse])
    # It assumes the config_id is the *last* group of digits preceded by an underscore.
    # (.*?)       Group 1: Base Name (non-greedy)
    # _           Literal underscore
    # (\d+)       Group 2: Config ID (digits)
    # (?:         Optional non-capturing group for suffix
    #   _         Literal underscore
    #   (.*)      Group 3: Suffix (anything following the underscore after config_id)
    # )?          End optional group
    # $           End of string
    match = re.match(r'^(.*?)_(\d+)(?:_(.*))?$', col_name)

    if match:
        base_name = match.group(1)
        config_id_str = match.group(2)
        output_suffix = match.group(3) # This could be None, a number string, or a text suffix

        try:
            config_id = int(config_id_str)
            # Check if base_name itself ends with _<digits> which might be part of the *actual* indicator name
            # e.g., if an indicator was genuinely named 'MY_INDICATOR_10' and config ID is 5 -> 'MY_INDICATOR_10_5'
            # This is tricky. For now, assume the last _<digits> is the config ID. Refine if needed.
            return base_name, config_id, output_suffix # output_suffix can be None
        except ValueError:
            # This case should be rare if the regex matched digits for config_id
            logger.error(f"Could not convert supposed config_id '{config_id_str}' to int for column '{col_name}'")
            return None
    else:
        # If the primary pattern fails, log and return None
        # logger.debug(f"Could not parse indicator column name format: {col_name}") # Reduce noise
        return None

def get_config_identifier(indicator_name: str, config_id: int, output_index_or_suffix: Optional[Any]) -> str:
    """
    Creates a consistent identifier string for an indicator configuration output.
    Handles numeric index or string suffix.
    """
    name = f"{indicator_name}_{config_id}"
    if output_index_or_suffix is not None:
        # Append the index or suffix directly
        name += f"_{output_index_or_suffix}"
    return name

def get_config_hash(params: Dict[str, Any]) -> str:
    """Generates a stable SHA256 hash for a parameter dictionary."""
    # Convert params to a canonical JSON string (sorted keys, no whitespace)
    config_str = json.dumps(params, sort_keys=True, separators=(',', ':'))
    # Encode to bytes and hash
    return hashlib.sha256(config_str.encode('utf-8')).hexdigest()

def compare_param_dicts(dict1: Dict, dict2: Dict) -> bool:
    """Compares two parameter dictionaries, handling float precision."""
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        val1 = dict1[key]
        val2 = dict2[key]
        if isinstance(val1, float) and isinstance(val2, float):
            # Use numpy's isclose for robust float comparison
            if not np.isclose(val1, val2, rtol=1e-5, atol=1e-8):
                return False
        elif type(val1) != type(val2): # Check types if not float
             return False
        elif val1 != val2: # Direct comparison for non-float types
            return False
    return True