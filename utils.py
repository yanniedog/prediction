# utils.py
import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
import re # Import re

logger = logging.getLogger(__name__)

def clear_screen() -> None:
    """Clears the terminal screen."""
    # Use standard library method for cross-platform compatibility
    print("\033[H\033[J", end="") # ANSI escape code for clearing screen

def determine_time_interval(data: pd.DataFrame) -> Optional[str]:
    """Determines the most common time interval in the data."""
    if 'date' not in data.columns:
        logger.error("No 'date' column found for determining time interval.")
        return None
    if len(data) < 2:
        logger.warning("Insufficient data to determine time interval.")
        return None

    # Ensure date column is suitable
    try:
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
             data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data = data.dropna(subset=['date']).sort_values('date')
        if data.empty: return None # No valid dates
    except Exception as e:
        logger.error(f"Error processing date column for interval determination: {e}")
        return None


    if not data['date'].is_monotonic_increasing:
        # Duplicates might have been introduced if sorting wasn't strict before
        initial_len = len(data)
        data = data.drop_duplicates(subset=['date'], keep='first')
        if len(data) < initial_len:
            logger.warning(f"Dropped {initial_len - len(data)} duplicate dates for interval calculation.")
        if not data['date'].is_monotonic_increasing:
             logger.error("'date' column still not monotonic after dropping duplicates.")
             return None # Cannot proceed if time doesn't move forward consistently
    if len(data) < 2: return None # Not enough data after potential drops


    time_diffs = data['date'].diff().dt.total_seconds().dropna()
    if time_diffs.empty:
        logger.warning("No time differences found after processing.")
        return None

    try:
        # Use median as it's more robust to outliers than mode for time differences
        median_diff = time_diffs.median()
        logger.debug(f"Median time difference: {median_diff} seconds.")

        if median_diff <= 0: # Ignore non-positive differences
            logger.warning("Median time difference is not positive. Cannot determine interval.")
            return None

        # Define thresholds (slightly adjusted for clarity, e.g., 59.9s is still seconds)
        if median_diff < 60: return 'seconds'
        if median_diff < 3600: return 'minutes'
        if median_diff < 86400: return 'hours'
        if median_diff < 604800: return 'days'
        return 'weeks' # Or potentially 'months' if needed
    except Exception as e:
        logger.error(f"Error determining time interval median: {e}", exc_info=True)
        return None

def parse_indicator_column_name(col_name: str) -> Optional[Tuple[str, int, Optional[int]]]:
    """
    Parses indicator column names like 'MACD_10_0', 'AD_29', 'KC_50_LOWER'.
    Returns (base_indicator_name, config_id, output_index|None)
    """
    # Regex Breakdown:
    # ^                     Start of string
    # (.*?)                 Group 1: Base name (non-greedy match of any character)
    # _                     Literal underscore separating base name from config ID
    # (\d+)                 Group 2: Config ID (one or more digits)
    # (?:                   Start of non-capturing group for optional suffix
    #   _                   Literal underscore separating config ID from suffix
    #   (\d+)               Group 3: Optional numeric output index (one or more digits)
    #   |                   OR
    #   _                   Literal underscore separating config ID from suffix
    #   ([a-zA-Z].*)       Group 4: Optional non-numeric suffix (starts with letter, then anything)
    # )?                    End of non-capturing group, make the whole suffix optional
    # $                     End of string
    match = re.match(r'^(.*?)_(\d+)(?:_(\d+)|_([a-zA-Z].*))?$', col_name)

    if match:
        base_name = match.group(1)
        config_id = int(match.group(2))
        output_index_str = match.group(3)
        output_suffix = match.group(4) # e.g., 'LOWER', 'PLUS', 'TSIs_13_25_13'

        if output_index_str is not None:
             # Standard numeric index (e.g., BBANDS_1_0)
             return base_name, config_id, int(output_index_str)
        elif output_suffix is not None:
             # Non-numeric suffix added (e.g., KC_66_LOWER, VORTEX_70_PLUS)
             # We treat this as a unique output, but don't assign a numeric index.
             # The combination of base_name + config_id + column name is unique.
             return base_name, config_id, None
        else:
             # Only base_name and config_id (e.g., SMA_11)
             return base_name, config_id, None
    else:
        # Fallback: Check if it matches NAME_ID format (for indicators without underscores in their base name)
        match_simple = re.match(r'^([a-zA-Z]+)_(\d+)$', col_name)
        if match_simple:
             base_name = match_simple.group(1)
             config_id = int(match_simple.group(2))
             return base_name, config_id, None

    # If no pattern matches
    # logger.debug(f"Could not parse indicator column name format: {col_name}") # Reduce noise if many columns are checked
    return None

def get_config_identifier(indicator_name: str, config_id: int, output_index: Optional[int]) -> str:
    """Creates a consistent identifier string for an indicator configuration output."""
    name = f"{indicator_name}_{config_id}"
    if output_index is not None:
        # Only add numeric index if provided (handles single output and non-numeric suffix cases)
        name += f"_{output_index}"
    return name