# leaderboard_manager.py
import sqlite3
import logging
from datetime import datetime, timezone
import json
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from collections import Counter # For tallying
# --> Added for consistency report <--
from pathlib import Path # <-- FIXED: Added missing import
from scipy.stats import linregress

import config
import utils

logger = logging.getLogger(__name__)

# --- Database Connection Helper ---
def _create_leaderboard_connection() -> Optional[sqlite3.Connection]:
    """Creates a connection to the leaderboard SQLite database."""
    conn = None
    try:
        config.LEADERBOARD_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Connecting to leaderboard database: {config.LEADERBOARD_DB_PATH}")
        # Increased timeout slightly
        conn = sqlite3.connect(config.LEADERBOARD_DB_PATH, timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys = OFF;") # Leaderboard is standalone
        conn.row_factory = sqlite3.Row # Return rows as dict-like objects
        logger.debug("Successfully connected to leaderboard database.")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to leaderboard database '{config.LEADERBOARD_DB_PATH}': {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error connecting to leaderboard DB: {e}", exc_info=True)
        return None


# --- Database Initialization ---
def initialize_leaderboard_db() -> bool:
    """Initializes the leaderboard database table if it doesn't exist."""
    conn = _create_leaderboard_connection()
    if conn is None: return False
    try:
        cursor = conn.cursor()
        cursor.execute("BEGIN TRANSACTION;")
        # Use millisecond precision for timestamp storage (TEXT ISO8601)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS leaderboard (
            lag INTEGER NOT NULL,
            correlation_type TEXT NOT NULL CHECK(correlation_type IN ('positive', 'negative')),
            correlation_value REAL NOT NULL,
            indicator_name TEXT NOT NULL,
            config_json TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            dataset_daterange TEXT NOT NULL,
            calculation_timestamp TEXT NOT NULL, -- Store as ISO 8601 text
            config_id_source_db INTEGER,
            source_db_name TEXT,
            PRIMARY KEY (lag, correlation_type)
        );""")
        # Add index if it doesn't exist
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_leaderboard_lag ON leaderboard(lag);")
        # Add index for faster predictor lookup per lag
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_leaderboard_lag_val ON leaderboard(lag, correlation_value);")

        conn.commit()
        logger.info(f"Leaderboard DB schema initialized/verified: {config.LEADERBOARD_DB_PATH}")
        return True
    except sqlite3.Error as e:
        logger.error(f"Error initializing leaderboard DB schema: {e}", exc_info=True)
        try: conn.rollback()
        except Exception as rb_err: logger.error(f"Rollback failed during schema init: {rb_err}")
        return False
    finally:
        if conn: conn.close()

# --- Load Leaderboard Data ---
def load_leaderboard() -> Dict[Tuple[int, str], Dict[str, Any]]:
    """Loads current leaderboard data from the database into a dictionary."""
    leaderboard_data: Dict[Tuple[int, str], Dict[str, Any]] = {}
    conn = _create_leaderboard_connection()
    if conn is None: logger.warning("Could not connect to load leaderboard."); return {}
    try:
        cursor = conn.cursor()
        # Check if table exists first
        cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='leaderboard' LIMIT 1;")
        if cursor.fetchone() is None:
            logger.warning("Leaderboard table does not exist. Returning empty.")
            return {}

        cursor.execute("""
            SELECT lag, correlation_type, correlation_value, indicator_name, config_json,
                   symbol, timeframe, dataset_daterange, calculation_timestamp,
                   config_id_source_db, source_db_name
            FROM leaderboard
        """)
        rows = cursor.fetchall()
        for row in rows:
            key = (row['lag'], row['correlation_type'])
            # Convert Row object to a standard dictionary
            leaderboard_data[key] = dict(row)
        logger.info(f"Loaded {len(leaderboard_data)} entries from leaderboard.")
        return leaderboard_data
    except sqlite3.Error as e:
        logger.error(f"Error loading leaderboard data: {e}", exc_info=True)
        return {}
    finally:
        if conn: conn.close()


# --- Check and Update Single Lag (Real-time Update) ---
def check_and_update_single_lag(
    lag: int,
    correlation_value: float,
    indicator_name: str,
    params: Dict[str, Any],
    config_id: int,
    symbol: str,
    timeframe: str,
    data_daterange: str,
    source_db_name: str
) -> bool:
    """
    Checks if the provided correlation is a new best for the given lag
    and updates the leaderboard DB immediately if it is. Also triggers
    leaderboard.txt export on update. Returns True if an update occurred.
    """
    if not pd.notna(correlation_value) or not isinstance(correlation_value, (float, int)):
        logger.debug(f"Skipping leaderboard check: Non-numeric correlation {correlation_value} (Lag {lag}, ID {config_id})")
        return False # Cannot compare NaN or non-numeric

    conn = _create_leaderboard_connection()
    if conn is None:
        logger.error("Cannot connect to leaderboard DB for single lag update.")
        return False

    updated = False
    try:
        cursor = conn.cursor()
        # Use IMMEDIATE transaction for better concurrency handling
        cursor.execute("BEGIN IMMEDIATE;")
        current_best_pos = -np.inf
        current_best_neg = np.inf

        # Get current bests for this lag
        cursor.execute("SELECT correlation_type, correlation_value FROM leaderboard WHERE lag = ?", (lag,))
        rows = cursor.fetchall()
        for row in rows:
            if row['correlation_value'] is not None: # Ensure value is not NULL
                if row['correlation_type'] == 'positive':
                    current_best_pos = max(current_best_pos, row['correlation_value'])
                elif row['correlation_type'] == 'negative':
                    current_best_neg = min(current_best_neg, row['correlation_value'])

        # Determine correlation type and check if it's a new best
        is_new_best = False
        new_corr_type = None
        current_val_float = float(correlation_value) # Ensure float for comparison

        # Use a small tolerance for floating point comparison? No, exact better here.
        if current_val_float > 0 and current_val_float > current_best_pos:
            is_new_best = True
            new_corr_type = 'positive'
        elif current_val_float < 0 and current_val_float < current_best_neg:
            is_new_best = True
            new_corr_type = 'negative'

        if is_new_best:
            old_best_val = current_best_pos if new_corr_type == 'positive' else current_best_neg
            # Changed to INFO level
            logger.info(f"LEADERBOARD UPDATE (Lag: {lag}, Type: {new_corr_type}): New Best Corr={current_val_float:.6f} (beats {old_best_val:.6f}) by Ind '{indicator_name}' (ID: {config_id})")
            try:
                 config_json = json.dumps(params, sort_keys=True, separators=(',', ':'))
            except TypeError as json_err:
                 logger.error(f"Cannot serialize params to JSON for leaderboard update (Lag {lag}, ID {config_id}): {params}. Error: {json_err}")
                 conn.rollback()
                 return False # Cannot store invalid JSON

            calculation_ts = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
            query = """INSERT OR REPLACE INTO leaderboard (lag, correlation_type, correlation_value, indicator_name, config_json,
                         symbol, timeframe, dataset_daterange, calculation_timestamp, config_id_source_db, source_db_name)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"""
            cursor.execute(query, (lag, new_corr_type, current_val_float, indicator_name, config_json,
                                  symbol, timeframe, data_daterange, calculation_ts, config_id, source_db_name))
            conn.commit() # Commit the single update
            updated = True
        # else:
        #     logger.debug(f"Leaderboard check (Lag: {lag}): Corr {current_val_float:.4f} not better than current bests (Pos: {current_best_pos:.4f}, Neg: {current_best_neg:.4f}).")

    except sqlite3.Error as e:
        logger.error(f"DB Error checking/updating leaderboard for lag {lag}: {e}", exc_info=True)
        try: conn.rollback()
        except Exception as rb_err: logger.error(f"Rollback failed: {rb_err}")
    except Exception as e:
         logger.error(f"Unexpected error during leaderboard check/update for lag {lag}: {e}", exc_info=True)
         try: conn.rollback()
         except Exception as rb_err: logger.error(f"Rollback failed: {rb_err}")
    finally:
        if conn: conn.close()

    # Trigger export AFTER closing DB connection if an update happened
    if updated:
        try:
            export_success = export_leaderboard_to_text()
            if not export_success:
                logger.error("Failed to export leaderboard.txt after real-time update.")
        except Exception as export_err:
            logger.error(f"Error triggering leaderboard export after update: {export_err}", exc_info=True)

    return updated


# --- Update Leaderboard Logic (Batch Mode - Calls Single Update) ---
def update_leaderboard(
    current_run_correlations: Dict[int, List[Optional[float]]],
    indicator_configs: List[Dict[str, Any]],
    max_lag: int,
    symbol: str,
    timeframe: str,
    data_daterange: str,
    source_db_name: str
) -> None:
    """
    Compares current run's results against the leaderboard and updates if better correlation found.
    Uses the `check_and_update_single_lag` function internally.
    """
    logger.info(f"Starting leaderboard update comparison (Batch Mode) for {symbol}_{timeframe} (Max Lag: {max_lag})...")

    config_details_map = {cfg['config_id']: cfg for cfg in indicator_configs if 'config_id' in cfg}
    updates_triggered_in_batch = 0
    total_checks = 0

    # Iterate through results and call the single-lag update function
    for config_id, correlations in current_run_correlations.items():
        if not isinstance(correlations, list):
            logger.warning(f"Invalid correlation data type for ID {config_id}. Skipping.")
            continue

        config_info = config_details_map.get(config_id)
        if not config_info:
            logger.warning(f"Config details missing for ID {config_id} during batch update loop. Skipping.")
            continue

        indicator_name = config_info.get('indicator_name', 'Unknown')
        params = config_info.get('params', {})

        # Check each lag for this config
        for lag_idx, value in enumerate(correlations):
            lag = lag_idx + 1
            if lag > max_lag: break # Don't process beyond the target max_lag
            total_checks += 1

            if pd.notna(value) and isinstance(value, (float, int)):
                updated = check_and_update_single_lag(
                    lag=lag,
                    correlation_value=float(value),
                    indicator_name=indicator_name,
                    params=params,
                    config_id=config_id,
                    symbol=symbol,
                    timeframe=timeframe,
                    data_daterange=data_daterange,
                    source_db_name=source_db_name
                )
                if updated:
                    updates_triggered_in_batch += 1
            # else: # NaN or invalid value, already logged by check_and_update_single_lag if necessary
            #    pass

    if updates_triggered_in_batch > 0:
        logger.info(f"Leaderboard batch comparison loop finished. Triggered {updates_triggered_in_batch} real-time updates/exports during {total_checks} checks.")
    else:
        logger.info(f"Leaderboard batch comparison loop finished. No updates triggered during {total_checks} checks (already optimal or real-time updates handled it).")


# --- Export Leaderboard to Text File ---
def export_leaderboard_to_text() -> bool:
    """Exports the current leaderboard to 'leaderboard.txt' in the reports directory."""
    logger.info("Exporting leaderboard to text file...")
    conn = _create_leaderboard_connection()
    if conn is None: logger.error("Cannot connect to leaderboard DB for export."); return False

    output_filepath = config.REPORTS_DIR / "leaderboard.txt"
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    current_timestamp_str = datetime.now(timezone.utc).isoformat(timespec='milliseconds')

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='leaderboard' LIMIT 1;")
        if cursor.fetchone() is None:
            logger.warning("Leaderboard table not found. Exporting empty file.")
            output_filepath.write_text(f"Leaderboard Table Not Found - Checked: {current_timestamp_str}\n", encoding='utf-8')
            return True

        # Fetch all current entries
        query = "SELECT * FROM leaderboard ORDER BY lag ASC, correlation_type ASC"
        cursor.execute(query)
        rows = cursor.fetchall()

        if not rows:
            logger.info("Leaderboard is empty. Exporting empty file.")
            output_filepath.write_text(f"Leaderboard Empty - Last Checked: {current_timestamp_str}\n", encoding='utf-8')
            return True

        # Determine the maximum lag present in the data
        max_lag_db = 0
        try: max_lag_db = max(row['lag'] for row in rows)
        except ValueError: max_lag_db = 0 # Handle empty sequence if somehow rows exist but lag is null

        # Aggregate data by lag, ensuring all lags up to max_lag_db are represented
        aggregated_data: Dict[int, Dict[str, Any]] = {lag: {'lag': lag} for lag in range(1, max_lag_db + 1)}
        for row in rows:
            lag = row['lag']
            if lag not in aggregated_data: aggregated_data[lag] = {'lag': lag} # Should not happen, but safe
            prefix = "pos_" if row['correlation_type'] == "positive" else "neg_"
            aggregated_data[lag].update({
                f'{prefix}value': row['correlation_value'],
                f'{prefix}indicator': row['indicator_name'],
                f'{prefix}params': row['config_json'],
                f'{prefix}symbol': row['symbol'],
                f'{prefix}timeframe': row['timeframe'],
                f'{prefix}dataset': row['dataset_daterange'],
                f'{prefix}timestamp': row['calculation_timestamp'],
                f'{prefix}config_id': row['config_id_source_db'],
                f'{prefix}source_db': row['source_db_name']
            })

        df = pd.DataFrame(list(aggregated_data.values()))
        # Ensure consistent column order
        cols = ['lag']
        pos_cols = ['pos_value', 'pos_indicator', 'pos_params', 'pos_symbol', 'pos_timeframe', 'pos_dataset', 'pos_config_id', 'pos_source_db', 'pos_timestamp']
        neg_cols = ['neg_value', 'neg_indicator', 'neg_params', 'neg_symbol', 'neg_timeframe', 'neg_dataset', 'neg_config_id', 'neg_source_db', 'neg_timestamp']
        all_expected_cols = cols + pos_cols + neg_cols
        # Add missing columns with None before formatting
        for col in all_expected_cols:
            if col not in df.columns: df[col] = None
        df = df[all_expected_cols] # Reorder

        # Formatting
        na_rep = 'N/A'; max_json_len = 50
        df['pos_value'] = df['pos_value'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else na_rep)
        df['neg_value'] = df['neg_value'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else na_rep)
        df['pos_config_id'] = df['pos_config_id'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else na_rep)
        df['neg_config_id'] = df['neg_config_id'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else na_rep)
        df['pos_params'] = df['pos_params'].fillna(na_rep).apply(lambda x: x if x == na_rep or len(x) <= max_json_len else x[:max_json_len-3] + '...')
        df['neg_params'] = df['neg_params'].fillna(na_rep).apply(lambda x: x if x == na_rep or len(x) <= max_json_len else x[:max_json_len-3] + '...')
        for col in df.columns:
             if 'timestamp' in col:
                  # Format timestamp with milliseconds
                  df[col] = pd.to_datetime(df[col], errors='coerce', utc=True).dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3].fillna(na_rep)
             elif col not in ['lag', 'pos_value', 'neg_value', 'pos_config_id', 'neg_config_id']:
                 df[col] = df[col].fillna(na_rep)

        output_string = f"Correlation Leaderboard - Last Updated: {current_timestamp_str}\n"
        separator_len = 250 # Increased width for timestamp
        output_string += "=" * separator_len + "\n"
        with pd.option_context('display.width', separator_len + 10):
            output_string += df.to_string( index=False, justify='left', max_colwidth=None, na_rep=na_rep )
        output_string += "\n" + "=" * separator_len

        output_filepath.write_text(output_string, encoding='utf-8')
        logger.info(f"Leaderboard successfully exported to: {output_filepath}")
        # Avoid printing to console during background updates
        # print(f"\nLeaderboard exported to: {output_filepath}")
        return True

    except Exception as e:
        logger.error(f"Error exporting leaderboard: {e}", exc_info=True)
        try: output_filepath.write_text(f"ERROR exporting leaderboard: {e}\n", encoding='utf-8')
        except: pass
        return False
    finally:
        if conn: conn.close()


# --- Find Best Predictor for a Given Lag ---
def find_best_predictor_for_lag(target_lag: int) -> Optional[Dict[str, Any]]:
    """Queries the leaderboard for the best performing configuration for a specific target lag (highest absolute correlation)."""
    logger.info(f"Searching leaderboard for best predictor for Lag = {target_lag}...")
    conn = _create_leaderboard_connection()
    if conn is None: logger.error("Cannot connect to leaderboard DB for predictor search."); return None

    best_entry: Optional[Dict[str, Any]] = None

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='leaderboard' LIMIT 1;")
        if cursor.fetchone() is None: logger.warning("Leaderboard table not found."); return None

        # Use ABS() and ORDER BY DESC directly in the query, ensure value is not NULL
        query = "SELECT * FROM leaderboard WHERE lag = ? AND correlation_value IS NOT NULL ORDER BY ABS(correlation_value) DESC LIMIT 1"
        cursor.execute(query, (target_lag,))
        row = cursor.fetchone()

        if not row: logger.warning(f"No valid entries found for Lag = {target_lag}."); return None

        current_corr_value = float(row['correlation_value']) # Already checked for NOT NULL

        try:
            params_dict = json.loads(row['config_json']) # Parse JSON params
            best_entry = dict(row) # Convert row to dict
            best_entry['params'] = params_dict # Replace JSON string with parsed dict
            max_abs_corr = abs(current_corr_value)
            logger.info(f"Found best predictor for Lag {target_lag}: {best_entry['indicator_name']} (ID: {best_entry['config_id_source_db']}), Abs Corr: {max_abs_corr:.6f}")
            return best_entry
        except json.JSONDecodeError:
            logger.error(f"Failed parse config_json for predictor candidate (Lag {target_lag}, Ind {row['indicator_name']}): {row['config_json']}")
            return None # Skip if JSON is bad

    except Exception as e:
        logger.error(f"Error searching leaderboard for predictor: {e}", exc_info=True)
        return None
    finally:
        if conn: conn.close()


# --- Generate Leading Indicator Report (Tally) ---
def generate_leading_indicator_report() -> bool:
    """Generates a report tallying which indicators lead most often on the main leaderboard."""
    logger.info("Generating leading indicators tally report...")
    conn = _create_leaderboard_connection()
    if conn is None:
        logger.error("Cannot connect to leaderboard DB for leading indicator report.")
        return False

    output_filepath = config.REPORTS_DIR / "leading_indicators_tally.txt" # Renamed file
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    current_timestamp_str = datetime.now(timezone.utc).isoformat(timespec='milliseconds')

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='leaderboard' LIMIT 1;")
        if cursor.fetchone() is None:
            logger.warning("Leaderboard table not found. Cannot generate leading indicator report.")
            output_filepath.write_text(f"Leaderboard Table Not Found - Checked: {current_timestamp_str}\n", encoding='utf-8')
            return True

        # Fetch necessary data: indicator name, config details, and type (pos/neg)
        query = """
        SELECT
            indicator_name, config_json, correlation_type, config_id_source_db
        FROM leaderboard;
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        if not rows:
            logger.info("Leaderboard is empty. Cannot generate leading indicator report.")
            output_filepath.write_text(f"Leaderboard Empty - Last Checked: {current_timestamp_str}\n", encoding='utf-8')
            return True

        pos_counts = Counter(); neg_counts = Counter()
        config_details_map = {} # Key: (indicator_name, config_id)

        for row in rows:
            ind_name, cfg_json, corr_type, cfg_id = row['indicator_name'], row['config_json'], row['correlation_type'], row['config_id_source_db']
            if cfg_id is None:
                logger.warning(f"Skipping leaderboard entry with null config_id for indicator '{ind_name}'.")
                continue
            key = (ind_name, cfg_id)
            if key not in config_details_map:
                 # Only store JSON string for display efficiency
                 config_details_map[key] = {'params_json': cfg_json}

            if corr_type == 'positive': pos_counts[key] += 1
            elif corr_type == 'negative': neg_counts[key] += 1

        report_data = []
        all_keys = set(pos_counts.keys()) | set(neg_counts.keys())

        for key in all_keys:
            ind_name, cfg_id = key
            pos_lead_count = pos_counts.get(key, 0)
            neg_lead_count = neg_counts.get(key, 0)
            total_lead_count = pos_lead_count + neg_lead_count
            params_json = config_details_map.get(key, {}).get('params_json', '{}')

            report_data.append({
                'Indicator': ind_name,
                'Config ID': cfg_id,
                'Params': params_json,
                'Positive Leads': pos_lead_count,
                'Negative Leads': neg_lead_count,
                'Total Leads': total_lead_count
            })

        if not report_data:
             logger.warning("No data aggregated for leading indicators report.")
             output_filepath.write_text(f"No valid leading indicator data found - Checked: {current_timestamp_str}\n", encoding='utf-8')
             return True

        # Create and sort DataFrame
        df_report = pd.DataFrame(report_data)
        df_report.sort_values(by=['Total Leads', 'Indicator', 'Config ID'], ascending=[False, True, True], inplace=True)

        # Format DataFrame for output
        max_p_len = 60
        df_report['Params'] = df_report['Params'].apply(lambda x: x if not isinstance(x, str) or len(x) <= max_p_len else x[:max_p_len-3] + '...')
        df_report['Config ID'] = df_report['Config ID'].astype(str) # Ensure string for display

        # Generate Output String
        output_string = f"Leading Indicators Tally (Based on Current Leaderboard)\n"
        output_string += f"Generated: {current_timestamp_str}\n"
        output_string += f"Counts how many lag periods each config holds the best positive or negative correlation.\n"
        output_string += "=" * 110 + "\n"
        with pd.option_context('display.width', 1000, 'display.max_colwidth', 70):
            output_string += df_report.to_string(index=False, justify='left')
        output_string += "\n" + "=" * 110

        output_filepath.write_text(output_string, encoding='utf-8')
        logger.info(f"Leading indicators tally report saved to: {output_filepath}")
        print(f"\nLeading indicators tally report saved to: {output_filepath}")
        return True

    except Exception as e:
        logger.error(f"Error generating leading indicators tally report: {e}", exc_info=True)
        try: output_filepath.write_text(f"ERROR generating leading indicators tally report: {e}\n", encoding='utf-8')
        except: pass
        return False
    finally:
        if conn: conn.close()


# --- NEW FUNCTION: Generate Consistency Report ---
def generate_consistency_report(
    correlations_by_config_id: Dict[int, List[Optional[float]]],
    indicator_configs_processed: List[Dict[str, Any]],
    max_lag: int,
    output_dir: Path,
    file_prefix: str,
    abs_corr_threshold: float = 0.15
) -> bool:
    """
    Generates a report analyzing the consistency of correlation across lags.
    """
    logger.info(f"Generating correlation consistency report (Threshold: {abs_corr_threshold})...")
    if not correlations_by_config_id or max_lag <= 0:
        logger.warning("No correlation data or invalid max_lag for consistency report.")
        return False

    output_filepath = output_dir / f"{file_prefix}_consistency_report.txt"
    current_timestamp_str = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
    configs_dict = {cfg['config_id']: cfg for cfg in indicator_configs_processed if 'config_id' in cfg}
    report_data = []
    lags_array = np.arange(1, max_lag + 1) # For regression slope calculation

    for cfg_id, corrs_full in correlations_by_config_id.items():
        config_info = configs_dict.get(cfg_id)
        if not config_info:
            logger.warning(f"Consistency Report: Config info missing for ID {cfg_id}. Skipping.")
            continue

        indicator_name = config_info.get('indicator_name', 'Unknown')
        params = config_info.get('params', {})
        try:
            params_str = json.dumps(params, separators=(',',':'))
        except Exception:
            params_str = str(params) # Fallback

        if corrs_full is None or not isinstance(corrs_full, list) or len(corrs_full) < max_lag:
            logger.warning(f"Consistency Report: Short/invalid corr data for {indicator_name} (ID {cfg_id}). Skipping.")
            continue

        corrs = corrs_full[:max_lag]
        corr_array = np.array(corrs, dtype=float)
        valid_mask = ~np.isnan(corr_array)
        valid_corrs = corr_array[valid_mask]
        valid_lags = lags_array[valid_mask]

        if len(valid_corrs) < 2: # Need at least 2 points for std dev, slope etc.
            logger.info(f"Consistency Report: Insufficient valid points for {indicator_name} (ID {cfg_id}). Skipping.")
            # Optionally add entry with NaNs if desired
            mean_abs_corr = np.nan
            std_dev_corr = np.nan
            lags_over_thresh = 0
            corr_slope = np.nan
            sign_flips = np.nan
        else:
            abs_corrs = np.abs(valid_corrs)
            mean_abs_corr = np.mean(abs_corrs)
            std_dev_corr = np.std(valid_corrs) # Std dev of original correlations
            lags_over_thresh = np.sum(abs_corrs > abs_corr_threshold)

            # Calculate correlation slope (Corr vs Lag)
            try:
                 # Use only valid points for linear regression
                 if len(valid_lags) >= 2:
                      slope_res = linregress(valid_lags, valid_corrs)
                      corr_slope = slope_res.slope
                 else:
                      corr_slope = np.nan
            except Exception as slope_err:
                 logger.warning(f"Could not calculate correlation slope for ID {cfg_id}: {slope_err}")
                 corr_slope = np.nan

            # Calculate sign flips
            signs = np.sign(valid_corrs)
            # Ignore zeros in sign flips, treat them as previous sign? Or filter them out? Let's filter.
            non_zero_signs = signs[signs != 0]
            if len(non_zero_signs) >= 2:
                 sign_changes = np.diff(non_zero_signs) != 0
                 sign_flips = np.sum(sign_changes)
            else:
                 sign_flips = 0 # No flips possible if < 2 non-zero points

        report_data.append({
            'Indicator': indicator_name,
            'Config ID': cfg_id,
            'Params': params_str,
            'Mean Abs Corr': mean_abs_corr,
            'Std Dev Corr': std_dev_corr,
            f'Lags > {abs_corr_threshold:.2f}': lags_over_thresh,
            'Corr Slope': corr_slope,
            'Sign Flips': sign_flips,
            'Valid Points': len(valid_corrs)
        })

    if not report_data:
        logger.warning("No data generated for consistency report.")
        output_filepath.write_text(f"No valid consistency data generated - Checked: {current_timestamp_str}\n", encoding='utf-8')
        return True

    # Create and sort DataFrame
    df_report = pd.DataFrame(report_data)
    # Sort primarily by Mean Abs Corr, then by Lags > Threshold, then Std Dev (lower is better)
    df_report.sort_values(
        by=['Mean Abs Corr', f'Lags > {abs_corr_threshold:.2f}', 'Std Dev Corr'],
        ascending=[False, False, True],
        na_position='last',
        inplace=True
    )

    # Format DataFrame for output
    max_p_len = 50
    df_report['Params'] = df_report['Params'].apply(lambda x: x if not isinstance(x, str) or len(x) <= max_p_len else x[:max_p_len-3] + '...')
    df_report['Config ID'] = df_report['Config ID'].astype(str)
    for col in ['Mean Abs Corr', 'Std Dev Corr', 'Corr Slope']:
         df_report[col] = df_report[col].map('{:.4f}'.format).replace('nan', 'N/A')
    for col in [f'Lags > {abs_corr_threshold:.2f}', 'Sign Flips', 'Valid Points']:
        # Handle potential NaNs in integer columns before formatting
        df_report[col] = df_report[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else 'N/A')


    # Generate Output String
    output_string = f"Correlation Consistency Report (Across Lags 1-{max_lag})\n"
    output_string += f"Generated: {current_timestamp_str}\n"
    output_string += f"Sorted by Mean Abs Corr (desc), then Lags > {abs_corr_threshold:.2f} (desc), then Std Dev Corr (asc).\n"
    output_string += "=" * 130 + "\n"
    with pd.option_context('display.width', 1000, 'display.max_colwidth', 60):
        output_string += df_report.to_string(index=False, justify='left', na_rep='N/A')
    output_string += "\n" + "=" * 130

    try:
        output_filepath.write_text(output_string, encoding='utf-8')
        logger.info(f"Consistency report saved to: {output_filepath}")
        print(f"\nConsistency report saved to: {output_filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving consistency report: {e}", exc_info=True)
        print("\nError saving consistency report file.")
        return False