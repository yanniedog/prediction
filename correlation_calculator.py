# correlation_calculator.py
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

import config
import sqlite_manager
import utils

logger = logging.getLogger(__name__)

# --- Debug Settings ---
DEBUG_INDICATOR_COL = 'sma_11'
DEBUG_LAG = 5

# --- NEW Corrected Function for User Goal ---
def calculate_correlation_indicator_vs_future_price(data: pd.DataFrame, indicator_col: str, lag: int) -> Optional[float]:
    """
    Calculate the Pearson correlation between the CURRENT indicator and FUTURE 'close' price.
    Assumes data is sorted chronologically ascending. Shifts CLOSE PRICE BACKWARD by lag.
    Calculates Corr(Indicator[t], Close[t+lag]).
    """
    if indicator_col not in data.columns or 'close' not in data.columns:
        logger.error(f"Missing '{indicator_col}' or 'close' column for correlation.")
        return None

    is_debug_target = (indicator_col.lower() == DEBUG_INDICATOR_COL.lower() and lag == DEBUG_LAG)

    indicator_series = data[indicator_col]
    close_series = data['close']

    if is_debug_target:
        logger.debug(f"--- CURRENT INDICATOR vs FUTURE PRICE DEBUG for {indicator_col} Lag={lag} ---")
        logger.debug(f"Input data len: {len(data)}")
        logger.debug(f"Indicator ('{indicator_col}') Series Head (first 5 non-NaN):\n{indicator_series.dropna().head(5)}")
        logger.debug(f"Indicator ('{indicator_col}') Series Tail (last 5):\n{indicator_series.tail(5)}")
        logger.debug(f"Close Series Head (first 5 non-NaN):\n{close_series.dropna().head(5)}")
        logger.debug(f"Close Series Tail (last 5):\n{close_series.tail(5)}")
        if indicator_series.isnull().all(): logger.debug("Indicator column is ALL NaN before shift.")
        if indicator_series.nunique(dropna=True) <= 1: logger.debug("Indicator column has <=1 unique non-NaN value before shift.")


    # Check for NaN/variance early
    if indicator_series.isnull().all(): return np.nan
    if indicator_series.nunique(dropna=True) <= 1: return np.nan

    try:
        # Correlate indicator[t] with close[t+lag]. Shift close price BACKWARD by lag.
        shifted_close_future = close_series.shift(-lag) # Negative shift on CLOSE

        if is_debug_target:
             logger.debug(f"Indicator Series Head (first 5):\n{indicator_series.head(5)}") # Show start
             logger.debug(f"Shifted Close Series (Future, lag={lag}, shift={-lag}) Head (first 5):\n{shifted_close_future.head(5)}") # Show start
             logger.debug(f"Indicator Series Tail (last {lag+5}):\n{indicator_series.tail(lag+5)}") # Show end where NaNs appear
             logger.debug(f"Shifted Close Series (Future, lag={lag}, shift={-lag}) Tail (last {lag+5}):\n{shifted_close_future.tail(lag+5)}") # Show end where NaNs appear

        # Align and drop NaNs. Pandas corr() handles this.
        combined_df = pd.concat([indicator_series, shifted_close_future], axis=1)
        combined_df.columns = ['indicator_current', 'close_future'] # Rename for clarity
        combined_df.dropna(inplace=True) # Drop rows where either is NaN (indicator_current or close_future)

        valid_count = len(combined_df)

        if is_debug_target: logger.debug(f"Valid Data Count after dropna: {valid_count}")

        if valid_count < 3:
            if is_debug_target: logger.warning("Not enough valid data points after dropna.")
            return np.nan

        valid_indicator_current = combined_df['indicator_current']
        valid_close_future = combined_df['close_future']

        # Check variance again on valid data
        if valid_indicator_current.std() < 1e-9 or valid_close_future.std() < 1e-9:
             if is_debug_target: logger.warning("Zero variance in valid data.")
             return np.nan

        # --- Log data being correlated ---
        if is_debug_target:
            logger.debug(f"VALID Indicator (Current) Data Head (first 5):\n{valid_indicator_current.head(5)}")
            logger.debug(f"VALID Close (Future) Data Head (first 5):\n{valid_close_future.head(5)}")
            logger.debug(f"VALID Indicator (Current) Data Tail (last 5):\n{valid_indicator_current.tail(5)}")
            logger.debug(f"VALID Close (Future) Data Tail (last 5):\n{valid_close_future.tail(5)}")
            logger.debug(f"Correlating series 1 (len {len(valid_indicator_current)}): Name='indicator_current' (From {indicator_col})")
            logger.debug(f"Correlating series 2 (len {len(valid_close_future)}): Name='close_future' (From Close, shifted {-lag})")
        # --- End Logging ---

        correlation = valid_indicator_current.corr(valid_close_future) # Corr(Indicator[t], Close[t+lag])

        if is_debug_target:
             logger.debug(f"CURRENT INDICATOR vs FUTURE PRICE Calculated Correlation: {correlation}")
             logger.debug(f"--- END CURRENT INDICATOR vs FUTURE PRICE DEBUG for {indicator_col} Lag={lag} ---")


        if np.isnan(correlation): logger.warning(f"Correlation NaN for {indicator_col}, lag {lag} (Indicator vs Future Price)."); return np.nan
        return float(correlation)

    except Exception as e: logger.error(f"Error calculating indicator vs future price correlation for {indicator_col}, lag {lag}: {e}", exc_info=True); return None


# --- process_correlations: Update to call the NEWEST function ---
def process_correlations(data: pd.DataFrame, db_path: str, symbol_id: int, timeframe_id: int, indicator_configs_processed: List[Dict[str, Any]], max_lag: int) -> bool:
    """ Calculates correlations for all computed indicators and lags, storing them in the database. USES CURRENT INDICATOR vs FUTURE PRICE CORRELATION """
    logger.info(f"Starting CURRENT INDICATOR vs FUTURE PRICE correlation calculation for {len(indicator_configs_processed)} base configurations up to lag {max_lag}...") # Indicate calculation type
    conn = sqlite_manager.create_connection(db_path);
    if not conn: return False

    indicator_columns_found = [col for col in data.columns if utils.parse_indicator_column_name(col) is not None]
    logger.info(f"Columns identified for correlation processing: {indicator_columns_found}") # Log the list
    if not indicator_columns_found: logger.error("No valid indicator columns found for correlation."); conn.close(); return False
    logger.info(f"Found {len(indicator_columns_found)} specific indicator output columns to process for correlation.")

    total_calcs_expected = len(indicator_columns_found) * max_lag; completed_calcs = 0; errors_found = 0; skipped_cols = 0
    try:
        cursor = conn.cursor()
        for indicator_col_name in indicator_columns_found:
            parsed_info = utils.parse_indicator_column_name(indicator_col_name)
            if parsed_info is None: logger.warning(f"Could not parse '{indicator_col_name}'. Skipping."); skipped_cols += max_lag; continue
            base_name, config_id, output_idx = parsed_info
            logger.debug(f"Processing correlations for: Col='{indicator_col_name}', Base='{base_name}', ConfigID={config_id}, OutIdx={output_idx}")

            # Sanity Check config_id
            if config_id not in [cfg['config_id'] for cfg in indicator_configs_processed]:
                 logger.warning(f"Col '{indicator_col_name}' ConfigID {config_id} not in initial list. Skipping."); skipped_cols += max_lag; continue

            for lag in range(1, max_lag + 1):
                # *** CALL THE NEWEST FUNCTION ***
                correlation_value = calculate_correlation_indicator_vs_future_price(data, indicator_col_name, lag)
                sqlite_manager.insert_correlation(conn, symbol_id, timeframe_id, config_id, lag, correlation_value)
                completed_calcs += 1
                if correlation_value is None or np.isnan(correlation_value): errors_found += 1
                if completed_calcs % 10000 == 0: logger.info(f"Correlation progress: {completed_calcs}/{total_calcs_expected} attempted.")

            # Commit after each indicator's lags are processed
            conn.commit()
            logger.debug(f"Committed correlations for {indicator_col_name}")


        logger.info(f"Finished CURRENT INDICATOR vs FUTURE PRICE correlation calculation. Attempted: {completed_calcs}, Skipped: {skipped_cols}, NaN/Errors: {errors_found}.")
        return True
    except Exception as e:
        try: conn.rollback()
        except Exception as rb_e: logger.error(f"Error during rollback: {rb_e}")
        logger.error(f"Major error during correlation loop: {e}", exc_info=True)
        return False
    finally:
        if conn: conn.close(); logger.debug("Closed DB connection after correlation processing.")