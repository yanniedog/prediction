# utils.py

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Set # Added Set
import logging
import re
import hashlib
import json
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path # Added Path
import shutil # Import shutil for file operations

# Added imports for potentially needed modules if functions were moved here
# (Adjust based on actual function locations if refactored)
import sqlite_manager
import leaderboard_manager
import visualization_generator
import config as app_config # Use app_config alias to avoid conflict

try:
    from dateutil.relativedelta import relativedelta
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False
    logging.getLogger(__name__).info("python-dateutil missing. Month timeframe calculations ('1M') may fail.")

logger = logging.getLogger(__name__)

# --- Function: cleanup_previous_content ---
def cleanup_previous_content(
    clean_reports: bool = True,
    clean_logs: bool = True,
    clean_db: bool = False, # Default to NOT cleaning database files
    # Add exclusions if needed, e.g., leaderboard DB, params file
    exclude_files: List[str] = [app_config.LEADERBOARD_DB_PATH.name, app_config.INDICATOR_PARAMS_PATH.name, '.gitignore']
) -> None:
    """Deletes generated content from previous runs."""
    logger.info("--- Starting Cleanup ---")

    dirs_to_clean = []
    if clean_reports:
        # Ensure all report subdirectories are included
        dirs_to_clean.extend([
            app_config.REPORTS_DIR, # Clean root reports dir first
            app_config.HEATMAPS_DIR,
            app_config.LINE_CHARTS_DIR,
            app_config.COMBINED_CHARTS_DIR
        ])
        # Remove duplicates just in case config paths overlap
        dirs_to_clean = list(set(dirs_to_clean))

    if clean_logs:
        dirs_to_clean.append(app_config.LOG_DIR)
    if clean_db:
        logger.warning("Database cleanup enabled. THIS WILL DELETE DOWNLOADED DATA FILES.")
        dirs_to_clean.append(app_config.DB_DIR)

    for target_dir in dirs_to_clean:
        if not target_dir.exists():
            logger.info(f"Cleanup: Directory '{target_dir}' not found, skipping.")
            continue
        # Skip cleaning if it's the project root itself (safety check)
        if target_dir.resolve() == app_config.PROJECT_ROOT.resolve():
             logger.warning(f"Cleanup: Skipping attempt to clean project root directory: {target_dir}")
             continue

        logger.info(f"Cleaning directory: {target_dir}")
        cleaned_count = 0
        skipped_count = 0
        try:
            # Ensure we only iterate if it's actually a directory
            if not target_dir.is_dir():
                logger.warning(f"Cleanup: Target '{target_dir}' is not a directory. Skipping.")
                continue

            for item_path in target_dir.iterdir():
                item_name = item_path.name
                # Check exclusions relative to the base DB_DIR or PROJECT_ROOT if necessary
                is_excluded = item_name in exclude_files
                # Special check for leaderboard DB if cleaning DB_DIR
                if clean_db and target_dir == app_config.DB_DIR and item_path == app_config.LEADERBOARD_DB_PATH:
                     is_excluded = True
                # Special check for params file if somehow in a cleaned dir (unlikely but safe)
                if item_path == app_config.INDICATOR_PARAMS_PATH:
                    is_excluded = True

                if is_excluded:
                    logger.debug(f"Skipping excluded file/dir: {item_name}")
                    skipped_count += 1
                    continue

                try:
                    if item_path.is_file():
                        item_path.unlink()
                        logger.debug(f"Deleted file: {item_path}")
                        cleaned_count += 1
                    elif item_path.is_dir():
                        # Double-check before deleting directory (safety)
                        if item_path.resolve() != app_config.PROJECT_ROOT.resolve():
                             shutil.rmtree(item_path)
                             logger.debug(f"Deleted directory: {item_path}")
                             cleaned_count += 1
                        else:
                             logger.warning(f"Cleanup: Safety prevented deletion of project root via subdirectory reference: {item_path}")
                             skipped_count +=1

                except Exception as e:
                    logger.error(f"Error deleting item {item_path}: {e}", exc_info=True)
            logger.info(f"Finished cleaning '{target_dir}'. Deleted: {cleaned_count}, Skipped: {skipped_count}.")
        except Exception as e:
            logger.error(f"Error iterating through directory {target_dir}: {e}", exc_info=True)

    logger.info("--- Cleanup Finished ---")
# --- End of Cleanup Function ---


# --- Existing Utils ---

def clear_screen() -> None:
    """Clears the console screen."""
    try: os.system('cls' if os.name == 'nt' else 'clear')
    except Exception: pass # Ignore if clearing fails

def parse_indicator_column_name(col_name: str) -> Optional[Tuple[str, int, Optional[str]]]:
    """Parses indicator column names like 'RSI_123_FASTK' or 'ADX_456'."""
    # Regex to match: BaseName_ConfigID OR BaseName_ConfigID_Suffix
    match = re.match(r'^(.*?)_(\d+)(?:_([^_].*|_+))?$', col_name) # Adjusted suffix capture
    if match:
        base_name, config_id_str, output_suffix = match.groups()
        try:
            cleaned_suffix = output_suffix.strip('_') if output_suffix else None
            return base_name, int(config_id_str), cleaned_suffix
        except ValueError:
            logger.error(f"Internal Error parsing config ID from column '{col_name}'")
            return None
    else:
        match_simple = re.match(r'^(.*?)_(\d+)$', col_name)
        if match_simple:
            base_name, config_id_str = match_simple.groups()
            try:
                return base_name, int(config_id_str), None
            except ValueError:
                logger.error(f"Internal Error parsing config ID from column '{col_name}'")
                return None
        return None

def get_config_identifier(indicator_name: str, config_id: int, output_index_or_suffix: Optional[Any]) -> str:
    """Constructs a standardized column name for an indicator configuration."""
    clean_name = str(indicator_name).strip().replace(' ','_')
    name = f"{clean_name}_{config_id}"
    if output_index_or_suffix is not None:
        clean_suffix = str(output_index_or_suffix).strip().replace(' ','_')
        if clean_suffix:
            name += f"_{clean_suffix}"
    return name

# Helper for consistent hashing/JSON serialization
def round_floats_for_hashing(obj):
    """Recursively rounds floats in nested data structures for consistent hashing."""
    if isinstance(obj, float):
        return round(obj, 8) # Use a fixed precision
    if isinstance(obj, dict):
        return {k: round_floats_for_hashing(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats_for_hashing(elem) for elem in obj]
    return obj

def get_config_hash(params: Dict[str, Any]) -> str:
    """Generates a stable SHA256 hash for a parameter dictionary, rounding floats."""
    try:
        params_rounded = round_floats_for_hashing(params)
        config_str = json.dumps(params_rounded, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(config_str.encode('utf-8')).hexdigest()
    except TypeError as e:
        logger.error(f"Error hashing params (likely non-serializable type): {params}. Error: {e}")
        try:
            items_repr = [f"{k}:{json.dumps(v, sort_keys=True)}" if isinstance(v, (dict, list, tuple)) else f"{k}:{repr(v)}" for k, v in sorted(params.items())]
            fallback_str = ";".join(items_repr)
            logger.warning(f"Using fallback hashing method for params: {params}")
            return hashlib.sha256(fallback_str.encode('utf-8')).hexdigest()
        except Exception as fallback_e:
            logger.critical(f"CRITICAL: Failed fallback hashing for params: {params}. Error: {fallback_e}")
            raise TypeError(f"Cannot hash parameters: {params}") from e


def compare_param_dicts(dict1: Optional[Dict], dict2: Optional[Dict]) -> bool:
    """Compares two parameter dictionaries, handling None and float precision."""
    if dict1 is None and dict2 is None: return True
    if dict1 is None or dict2 is None: return False
    if not isinstance(dict1, dict) or not isinstance(dict2, dict): return False
    if dict1.keys() != dict2.keys(): return False

    for key in dict1:
        val1, val2 = dict1[key], dict2[key]
        if isinstance(val1, float) and isinstance(val2, float):
            if not np.isclose(val1, val2, rtol=1e-6, atol=1e-9): return False
        elif isinstance(val1, (np.floating, np.integer)) or isinstance(val2, (np.floating, np.integer)):
             try:
                 py_val1 = val1.item() if hasattr(val1, 'item') else val1
                 py_val2 = val2.item() if hasattr(val2, 'item') else val2
                 if isinstance(py_val1, float) and isinstance(py_val2, float):
                      if not np.isclose(py_val1, py_val2, rtol=1e-6, atol=1e-9): return False
                 elif py_val1 != py_val2:
                      return False
             except (AttributeError, TypeError):
                 if val1 != val2: return False
        elif type(val1) != type(val2):
             try:
                 num1 = float(val1); num2 = float(val2)
                 if not np.isclose(num1, num2, rtol=1e-6, atol=1e-9): return False
             except (ValueError, TypeError):
                 return False
        elif val1 != val2:
            return False
    return True


def format_duration(duration: timedelta) -> str:
    """Formats a timedelta duration into a human-readable string (e.g., 1d 2h 3m 4s)."""
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
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts) if parts else "0s"

def estimate_price_precision(price: float) -> int:
    """Estimates appropriate number of decimal places based on price magnitude."""
    if not isinstance(price, (int, float, np.number)) or not np.isfinite(price) or price == 0:
        return 4
    abs_price = abs(price)
    if abs_price >= 10000: return 1
    if abs_price >= 1000: return 2
    if abs_price >= 10: return 3
    if abs_price >= 1: return 4
    if abs_price >= 0.01: return 6
    if abs_price >= 0.0001: return 7
    return 8


# --- Timeframe Calculation Helpers ---
def _get_seconds_per_period(timeframe: str) -> Optional[float]:
    """Converts a timeframe string (e.g., '1h', '3d') to seconds."""
    tf_lower = str(timeframe).lower()
    if tf_lower == '1m': return 60.0
    if tf_lower.endswith('m') and not tf_lower.endswith('mo'):
        match = re.match(r'(\d+)m$', tf_lower)
        if match:
            try: value = int(match.group(1)); return float(value * 60) if value > 0 else None
            except ValueError: return None
    elif tf_lower.endswith('h'):
        match = re.match(r'(\d+)h$', tf_lower)
        if match:
            try: value = int(match.group(1)); return float(value * 3600) if value > 0 else None
            except ValueError: return None
    elif tf_lower.endswith('d'):
        match = re.match(r'(\d+)d$', tf_lower)
        if match:
            try: value = int(match.group(1)); return float(value * 86400) if value > 0 else None
            except ValueError: return None
    elif tf_lower.endswith('w'):
        match = re.match(r'(\d+)w$', tf_lower)
        if match:
            try: value = int(match.group(1)); return float(value * 604800) if value > 0 else None
            except ValueError: return None
    elif tf_lower == '1M': return None # Variable length
    elif tf_lower.endswith('y'):
         match = re.match(r'(\d+)y$', tf_lower)
         if match:
             try: value = int(match.group(1)); return float(value * 31536000) if value > 0 else None
             except ValueError: return None
    logger.error(f"Unrecognized or unsupported timeframe format: {timeframe}")
    return None

def calculate_periods_between_dates(start_date: datetime, end_date: datetime, timeframe: str) -> Optional[int]:
    """Calculates the number of full periods between two dates for a given timeframe."""
    if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
        logger.error("Invalid date types provided for period calculation."); return None
    start_utc = start_date if start_date.tzinfo else start_date.replace(tzinfo=timezone.utc)
    end_utc = end_date if end_date.tzinfo else end_date.replace(tzinfo=timezone.utc)
    if end_utc <= start_utc: return 0

    if timeframe == '1M':
        if not DATEUTIL_AVAILABLE:
            logger.error("python-dateutil is required for '1M' timeframe calculations."); return None
        try:
            delta = relativedelta(end_utc, start_utc)
            total_months = delta.years * 12 + delta.months
            if (delta.days > 0 or delta.hours > 0 or delta.minutes > 0 or delta.seconds > 0 or delta.microseconds > 0):
                 periods = total_months + 1
            elif total_months == 0 and end_utc > start_utc:
                 periods = 1
            else:
                 periods = total_months
            return periods if periods >= 0 else 0
        except Exception as e:
            logger.error(f"Error calculating month periods: {e}"); return None

    period_seconds = _get_seconds_per_period(timeframe)
    if period_seconds is None: return None
    total_seconds_diff = (end_utc - start_utc).total_seconds()
    return math.ceil(total_seconds_diff / period_seconds) if total_seconds_diff > 0 else 0

def estimate_days_in_periods(periods: int, timeframe: str) -> Optional[float]:
    """Estimates the approximate number of days spanned by a number of periods."""
    if not isinstance(periods, int) or periods < 0:
        logger.error(f"Invalid number of periods ({periods})"); return None
    if periods == 0: return 0.0

    if timeframe == '1M':
        avg_days_month = 365.2425 / 12.0
        return periods * avg_days_month

    period_seconds = _get_seconds_per_period(timeframe)
    if period_seconds is None: return None
    return (periods * period_seconds) / 86400.0

def estimate_future_date(start_date: datetime, periods: int, timeframe: str) -> Optional[datetime]:
    """Estimates the date 'periods' away from the start_date for the given timeframe."""
    if not isinstance(start_date, datetime):
        logger.error("Invalid start_date provided."); return None
    if not isinstance(periods, int) or periods < 0:
        logger.error(f"Invalid number of periods ({periods})."); return None
    if periods == 0: return start_date

    tz_info = start_date.tzinfo if start_date.tzinfo else timezone.utc
    start_date_aware = start_date.replace(tzinfo=tz_info)

    try:
        if timeframe == '1M':
            if not DATEUTIL_AVAILABLE:
                logger.error("python-dateutil required for '1M' future date estimation."); return None
            return start_date_aware + relativedelta(months=periods)

        period_seconds = _get_seconds_per_period(timeframe)
        if period_seconds is None: return None
        return start_date_aware + timedelta(seconds=periods * period_seconds)
    except OverflowError:
        logger.error(f"Date calculation resulted in overflow: {periods} {timeframe} from {start_date}.")
        return None
    except Exception as e:
        logger.error(f"Error calculating future date: {e}"); return None


# --- *** ADDED estimate_duration function *** ---
def estimate_duration(num_configs_or_indicators: int, max_lag: int, path_type: str) -> timedelta:
    """Provides a VERY rough estimate of the analysis duration."""
    # Heuristics (These are guesses and need tuning based on observed performance)
    base_seconds = 60 # Base time for setup, loading, data prep, final reports etc.

    if path_type == 'tweak': # Bayesian Optimization Path
        # num_configs_or_indicators here represents the number of indicators being optimized
        num_indicators = num_configs_or_indicators
        calls_per_lag = app_config.DEFAULTS.get("optimizer_n_calls", 50)
        # Estimate time per objective function call (highly dependent on indicator complexity and data size)
        secs_per_call = 0.05 # ROUGH GUESS per call (indicator calc + correlation)
        # Total optimization time estimate
        opt_time = num_indicators * max_lag * calls_per_lag * secs_per_call
        # Estimate number of unique configs generated by optimization (wild guess)
        # Assume optimizer explores roughly 20 unique configs per indicator on average
        est_final_configs = num_indicators * 20
        # Estimate time for final correlation calculation phase if needed (less significant if already cached)
        final_corr_time_per_config_lag = 0.005 # ROUGH GUESS
        corr_time = est_final_configs * max_lag * final_corr_time_per_config_lag
        # Total = Base + Optimization + Final Correlation/Reporting buffer
        total_seconds = base_seconds + opt_time # Correlation mostly done during opt
    else: # Classical Path
        # num_configs_or_indicators here represents the total number of generated configurations
        num_configs = num_configs_or_indicators
        # Estimate time per indicator calculation per config
        secs_per_config_indicator = 0.02 # ROUGH GUESS
        # Estimate time per correlation calculation per config per lag
        secs_per_config_correlation = 0.005 # ROUGH GUESS
        # Total indicator calculation time
        indicator_time = num_configs * secs_per_config_indicator
        # Total correlation calculation time
        correlation_time = num_configs * max_lag * secs_per_config_correlation
        # Total = Base + Indicator Calc + Correlation Calc
        total_seconds = base_seconds + indicator_time + correlation_time

    # Add a general buffer (e.g., 25%) for overhead, reporting, unexpected delays
    total_seconds *= 1.25
    logger.info(f"Rough duration estimate ({path_type}, {num_configs_or_indicators} items, lag {max_lag}): {total_seconds:.0f}s")
    return timedelta(seconds=total_seconds)
# --- *** END estimate_duration function *** ---


# --- Interim Report Function (Moved from main.py, adjusted) ---
def run_interim_reports(
    db_path: Path, symbol_id: int, timeframe_id: int,
    configs_for_report: List[Dict[str, Any]], max_lag: int,
    file_prefix: str, stage_name: str = "Interim",
    correlation_data: Optional[Dict[int, List[Optional[float]]]] = None # Accept pre-fetched data
) -> None:
    """Fetches correlation data (if not provided) and generates reports."""
    logger.info(f"--- Generating {stage_name} Reports ---")
    conn = None; interim_correlations = correlation_data; report_data_ok = False; actual_max_lag_interim = 0

    # Fetch data only if not provided
    if interim_correlations is None:
        logger.info(f"Fetching correlation data for {stage_name} reports...")
        try:
            conn = sqlite_manager.create_connection(str(db_path))
            if not conn: raise ConnectionError("Failed connect DB for interim reports.")
            config_ids_to_fetch = [cfg['config_id'] for cfg in configs_for_report if 'config_id' in cfg]
            if not config_ids_to_fetch:
                logger.warning(f"{stage_name} Reports: No config IDs to fetch."); return
            interim_correlations = sqlite_manager.fetch_correlations(conn, symbol_id, timeframe_id, config_ids_to_fetch)
        except Exception as fetch_err:
            logger.error(f"Error fetching data for {stage_name} reports: {fetch_err}", exc_info=True)
        finally:
            if conn: conn.close()

    # Validate fetched or provided data
    if interim_correlations:
         for data_list in interim_correlations.values():
              if data_list and isinstance(data_list, list):
                  valid_indices = [i for i, v in enumerate(data_list) if pd.notna(v)]
                  if valid_indices:
                      actual_max_lag_interim = max(actual_max_lag_interim, max(valid_indices) + 1)
         if actual_max_lag_interim > 0:
             report_data_ok = True
             logger.info(f"{stage_name} report using data up to actual max lag {actual_max_lag_interim}.")
         else:
             logger.warning(f"{stage_name} report: No valid correlation data found (max lag 0).")
    else:
        logger.warning(f"{stage_name} report: No correlation data available (fetched or provided).")

    if not report_data_ok:
        logger.warning(f"Skipping {stage_name} report generation due to lack of valid data.")
        return

    interim_prefix = f"{file_prefix}_{stage_name.upper()}"
    report_lag = min(max_lag, actual_max_lag_interim) # Use the smaller of target max lag and actual data lag

    # Generate reports (safely call each one, avoid console prints here)
    try:
        visualization_generator.generate_peak_correlation_report(interim_correlations, configs_for_report, report_lag, app_config.REPORTS_DIR, interim_prefix)
    except Exception as e: logger.error(f"Error generating {stage_name} peak report: {e}", exc_info=True)
    try:
        leaderboard_manager.generate_consistency_report(interim_correlations, configs_for_report, report_lag, app_config.REPORTS_DIR, interim_prefix)
    except Exception as e: logger.error(f"Error generating {stage_name} consistency report: {e}", exc_info=True)
    try:
        # Force update of leaderboard text and tally for this stage report
        logger.info(f"Updating leaderboard.txt and tally report for {stage_name} stage.")
        leaderboard_manager.export_leaderboard_to_text()
        leaderboard_manager.generate_leading_indicator_report()
    except Exception as e: logger.error(f"Error generating/exporting {stage_name} leaderboard/tally: {e}", exc_info=True)

    logger.info(f"--- Finished {stage_name} Reports ---")
# --- End of Interim Report Function ---