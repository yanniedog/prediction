# correlation_calculator.py
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import sqlite3 # For specific error handling if needed
import time
import concurrent.futures
import os # To get CPU count
from datetime import timedelta # For ETA calculation

import config
import sqlite_manager
import utils # For ETA formatting

logger = logging.getLogger(__name__)

# --- Constants ---
ETA_UPDATE_INTERVAL_SECONDS_CORR = config.DEFAULTS.get("eta_update_interval_seconds", 15) # Use global config

# --- Core Correlation Calculation Function ---
# Calculates Corr(Indicator[t], Close[t+lag]) by shifting Close backward.
# (No changes needed in this specific function)
def calculate_correlation_indicator_vs_future_price(
    data: pd.DataFrame, indicator_col: str, lag: int
) -> Optional[float]:
    """
    Calculate Pearson correlation: Indicator[t] vs Close[t+lag].

    Args:
        data (pd.DataFrame): DataFrame with 'close' and indicator column, sorted chronologically.
                             Should have NaNs handled prior to calling if possible.
        indicator_col (str): Name of the indicator column.
        lag (int): Future periods for the price (must be > 0).

    Returns:
        Optional[float]: Pearson correlation, np.nan if impossible, None on error.
    """
    if not all(col in data.columns for col in [indicator_col, 'close']):
        return None
    if lag <= 0:
        return None

    indicator_series = data[indicator_col]
    close_series = data['close']

    # Early exit checks
    if indicator_series.isnull().all() or close_series.isnull().all() or indicator_series.nunique(dropna=True) <= 1:
        return np.nan

    try:
        # Shift Close Price BACKWARD by lag
        shifted_close_future = close_series.shift(-lag)
        # Combine, dropna, and check length before correlating
        combined = pd.concat([indicator_series, shifted_close_future], axis=1).dropna()
        if len(combined) < 2: # Need at least 2 pairs
            return np.nan
        # Calculate correlation using pandas on the cleaned data
        correlation = combined.iloc[:, 0].corr(combined.iloc[:, 1])
        # Return float or nan
        return float(correlation) if pd.notna(correlation) else np.nan
    except Exception as e:
        # logger.error(f"Error calculating correlation for {indicator_col}, lag {lag}: {e}", exc_info=False) # Avoid excessive logging in worker
        return None # Return None on unexpected error


# --- Worker Function for Parallel Processing ---
def _calculate_correlations_for_single_indicator(
    indicator_col_name: str,
    indicator_series: pd.Series, # Pass the series directly
    shifted_closes_future: Dict[int, pd.Series], # Pass the pre-shifted closes
    max_lag: int,
    symbol_id: int,
    timeframe_id: int,
    config_id: int
) -> List[Tuple[int, int, int, int, Optional[float]]]:
    """
    Worker to calculate correlations for all lags for one indicator series.
    """
    results_for_indicator = []
    nan_count = 0
    error_count = 0

    # Basic check before loop
    if indicator_series.isnull().all() or indicator_series.nunique(dropna=True) <= 1:
        is_all_nan = indicator_series.isnull().all()
        # Changed to DEBUG as this is common/expected for some indicators
        logger.debug(f"Worker: Skipping {indicator_col_name} (ConfigID: {config_id}) - {'All NaN' if is_all_nan else 'Constant Value'}.")
        # Return Nones (for DB) for all lags
        return [(symbol_id, timeframe_id, config_id, lag, None) for lag in range(1, max_lag + 1)]

    for lag in range(1, max_lag + 1):
        correlation_value: Optional[float] = None
        db_value: Optional[float] = None
        try:
            shifted_close = shifted_closes_future.get(lag)
            if shifted_close is None:
                correlation_value = None # Or np.nan? Using None for DB consistency
                error_count += 1
            else:
                # Use pandas .corr() - handles alignment and NaN pair removal
                # Combine first to ensure enough pairs after dropping NaNs
                combined = pd.concat([indicator_series, shifted_close], axis=1).dropna()
                if len(combined) < 2: # Check if enough data remains
                     correlation = np.nan
                else:
                     # Correlate the cleaned columns
                     correlation = combined.iloc[:, 0].corr(combined.iloc[:, 1])

                if pd.notna(correlation):
                    correlation_value = float(correlation)
                else:
                    correlation_value = np.nan # Use NaN for calculation failures (insufficient pairs/variance)
                    nan_count += 1
        except Exception as e:
            # logger.error(f"Worker Error ({indicator_col_name}, lag {lag}): {e}") # Can be noisy
            correlation_value = None # Use None for unexpected errors
            error_count += 1

        # Map np.nan -> None for DB storage consistency
        db_value = None if pd.isna(correlation_value) else correlation_value
        results_for_indicator.append((symbol_id, timeframe_id, config_id, lag, db_value))

    if nan_count > 0 or error_count > 0:
         # Changed to DEBUG
         logger.debug(f"Worker ({indicator_col_name}, ConfigID: {config_id}): Completed with {nan_count} NaN results, {error_count} errors.")

    return results_for_indicator


# --- Process Correlations (Parallelized Version) ---
def process_correlations(
    data: pd.DataFrame,
    db_path: str,
    symbol_id: int,
    timeframe_id: int,
    indicator_configs_processed: List[Dict[str, Any]],
    max_lag: int
) -> bool:
    """
    Calculates correlations using parallel processing across indicators, with progress reporting.
    """
    start_time_total = time.time() # Absolute start time for this function
    num_configs = len(indicator_configs_processed)
    logger.info(f"Starting PARALLELIZED correlation calculation for {num_configs} configurations up to lag {max_lag}...")

    # --- Input Data Validation ---
    if 'close' not in data.columns or data['close'].isnull().all():
        logger.error("'close' column missing or all NaN. Cannot calculate correlations.")
        return False
    if max_lag <= 0:
        logger.error(f"Max lag must be positive, got {max_lag}.")
        return False
    # Ensure sufficient data length *after* NaN drop (as done in main.py)
    min_required_len = max_lag + config.DEFAULTS.get("min_data_points_for_lag", 1) # Use default, min 1 point after lag
    if len(data) < min_required_len:
         logger.error(f"Input data has insufficient rows ({len(data)}) for max_lag={max_lag} after NaN drop. Need {min_required_len}.");
         return False
    # Find valid indicator columns *after* main data checks
    indicator_columns_present = [col for col in data.columns if utils.parse_indicator_column_name(col) is not None]
    if not indicator_columns_present:
        logger.error("No valid indicator columns found in input data for correlation.")
        return False
    num_indicator_cols = len(indicator_columns_present) # Number of columns to process

    # --- Database Connection ---
    conn = sqlite_manager.create_connection(db_path)
    if not conn: return False

    try:
        # --- Pre-shift close prices ---
        logger.info(f"Pre-calculating {max_lag} shifted 'close' price series...")
        start_shift_time = time.time()
        close_series = data['close'].astype(float)
        # Reindex shifted series to match the main data index for robust alignment
        shifted_closes_future = {lag: close_series.shift(-lag).reindex(data.index) for lag in range(1, max_lag + 1)}
        logger.info(f"Pre-calculation of shifted closes complete. Time: {time.time() - start_shift_time:.2f}s.")

        # --- Prepare tasks for parallel execution ---
        tasks = []
        config_details_map = {cfg['config_id']: cfg for cfg in indicator_configs_processed if 'config_id' in cfg}
        valid_tasks_count = 0
        skipped_task_configs = 0

        for indicator_col_name in indicator_columns_present:
            parsed_info = utils.parse_indicator_column_name(indicator_col_name)
            if parsed_info is None:
                logger.warning(f"Could not parse '{indicator_col_name}'. Skipping."); skipped_task_configs += 1; continue
            base_name, config_id, output_suffix = parsed_info
            # Ensure config_id is int
            if not isinstance(config_id, int):
                logger.warning(f"Parsed non-integer config_id {config_id} from '{indicator_col_name}'. Skipping."); skipped_task_configs += 1; continue
            # Only add tasks for configs we actually processed and have details for
            if config_id not in config_details_map:
                logger.debug(f"ConfigID {config_id} ('{indicator_col_name}') not in processed list map. Skipping task."); skipped_task_configs += 1; continue

            # Pass the actual indicator series to the task
            indicator_series = data[indicator_col_name].astype(float) # Ensure float

            tasks.append((
                indicator_col_name, indicator_series, shifted_closes_future,
                max_lag, symbol_id, timeframe_id, config_id
            ))
            valid_tasks_count += 1

        logger.info(f"Prepared {valid_tasks_count} correlation tasks for {num_indicator_cols} indicator columns. Skipped {skipped_task_configs} columns.")
        if not tasks: logger.error("No valid tasks generated."); return False

        # --- Execute in Parallel with Progress Reporting ---
        all_correlation_results: List[Tuple[int, int, int, int, Optional[float]]] = []
        num_cores = os.cpu_count() or 4 # Default to 4 if cannot detect
        max_workers = max(1, num_cores - 1) if num_cores > 1 else 1 # Use N-1 cores, minimum 1
        logger.info(f"Starting parallel execution with up to {max_workers} workers...")
        start_parallel_time = time.time()
        last_progress_update_time = start_parallel_time
        processed_tasks = 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks and store future-to-task mapping
            future_to_task = {executor.submit(_calculate_correlations_for_single_indicator, *task_args): task_args for task_args in tasks}

            for future in concurrent.futures.as_completed(future_to_task):
                task_args = future_to_task[future]
                indicator_name_done = task_args[0]
                try:
                    result_list = future.result()
                    if result_list: all_correlation_results.extend(result_list)
                    else: logger.warning(f"Worker for '{indicator_name_done}' returned no results.")
                except Exception as exc:
                    logger.error(f"Worker for '{indicator_name_done}' generated exception: {exc}", exc_info=True)

                processed_tasks += 1
                current_time = time.time()

                # --- Update Progress & ETA ---
                if current_time - last_progress_update_time > ETA_UPDATE_INTERVAL_SECONDS_CORR or processed_tasks == valid_tasks_count:
                    elapsed_td = timedelta(seconds=current_time - start_parallel_time)
                    elapsed_str = utils.format_duration(elapsed_td)
                    percent = (processed_tasks / valid_tasks_count * 100) if valid_tasks_count > 0 else 0
                    eta_str = "Calculating..."
                    if percent > 1 and processed_tasks < valid_tasks_count : # Estimate after some progress
                        rate = processed_tasks / elapsed_td.total_seconds() if elapsed_td.total_seconds() > 1 else 0
                        if rate > 0:
                            remaining_tasks = valid_tasks_count - processed_tasks
                            eta_td = timedelta(seconds=remaining_tasks / rate)
                            eta_str = utils.format_duration(eta_td)
                        # else: eta_str remains "Calculating..."
                    elif processed_tasks == valid_tasks_count:
                        eta_str = "Done"

                    print(f"\rCorrelation Progress: {processed_tasks}/{valid_tasks_count} tasks ({percent:.1f}%) | Elapsed: {elapsed_str} | ETA: {eta_str}   ", end="")
                    last_progress_update_time = current_time
                    # Log less frequently to file
                    if processed_tasks % 50 == 0 or processed_tasks == valid_tasks_count:
                        logger.info(f"Parallel Correlation Progress: {processed_tasks}/{valid_tasks_count} tasks ({percent:.1f}%) | Elapsed: {elapsed_str} | ETA: {eta_str}")

        print() # Newline after progress updates finish
        parallel_duration = time.time() - start_parallel_time
        logger.info(f"Parallel execution finished. Time: {parallel_duration:.2f}s.")
        logger.info(f"Total correlation records collected: {len(all_correlation_results)}")

        # --- Batch Insert Results ---
        if not all_correlation_results:
            logger.warning("No correlation results generated by parallel workers.")
            # Ensure database connection is closed even if no data to insert
            if conn: conn.close()
            return True # Success, but no data inserted

        logger.info(f"Starting batch insert of {len(all_correlation_results)} records...")
        start_insert_time = time.time()
        # Pass the existing connection to the batch insert function
        batch_success = sqlite_manager.batch_insert_correlations(conn, all_correlation_results)
        insert_duration = time.time() - start_insert_time
        logger.info(f"Batch insertion complete. Success: {batch_success}. Time: {insert_duration:.2f}s.")

        if not batch_success:
            logger.error("Batch insertion of correlations failed.")
            # Connection is closed in finally block
            return False

        total_duration = time.time() - start_time_total
        logger.info(f"Parallel correlation processing finished. Total time: {total_duration:.2f}s.")
        return True

    except Exception as e:
        logger.error(f"An error occurred during correlation processing: {e}", exc_info=True)
        return False
    finally:
        if conn:
            try:
                conn.close()
                logger.debug("Correlation processing DB connection closed.")
            except Exception as close_err:
                logger.error(f"Error closing correlation DB connection: {close_err}")