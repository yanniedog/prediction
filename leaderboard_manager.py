# leaderboard_manager.py
import sqlite3
import logging
from datetime import datetime, timezone
import json
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from collections import Counter # For tallying

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
        conn = sqlite3.connect(config.LEADERBOARD_DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys = OFF;") # Leaderboard is standalone
        logger.debug("Successfully connected to leaderboard database.")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to leaderboard database '{config.LEADERBOARD_DB_PATH}': {e}", exc_info=True)
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
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_leaderboard_lag ON leaderboard(lag);")
        conn.commit()
        logger.info(f"Leaderboard DB schema initialized: {config.LEADERBOARD_DB_PATH}")
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
        cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='leaderboard';")
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
            (lag, corr_type, value, ind_name, cfg_json, sym, tf, dr, ts, cfg_id, src_db) = row
            key = (lag, corr_type)
            leaderboard_data[key] = {
                'correlation_value': value, 'indicator_name': ind_name, 'config_json': cfg_json,
                'symbol': sym, 'timeframe': tf, 'dataset_daterange': dr,
                'calculation_timestamp': ts, 'config_id_source_db': cfg_id, 'source_db_name': src_db
            }
        logger.info(f"Loaded {len(leaderboard_data)} entries from leaderboard.")
        return leaderboard_data
    except sqlite3.Error as e:
        logger.error(f"Error loading leaderboard data: {e}", exc_info=True)
        return {}
    finally:
        if conn: conn.close()


# --- NEW FUNCTION: Check and Update Single Lag (Real-time Update) ---
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
    if not pd.notna(correlation_value):
        # logger.debug(f"Skipping leaderboard check: NaN correlation (Lag {lag}, ID {config_id})")
        return False # Cannot compare NaN

    conn = _create_leaderboard_connection()
    if conn is None:
        logger.error("Cannot connect to leaderboard DB for single lag update.")
        return False

    updated = False
    try:
        cursor = conn.cursor()
        # Use IMMEDIATE transaction for better concurrency handling if needed, although WAL helps
        cursor.execute("BEGIN IMMEDIATE;")
        current_best_pos = -np.inf
        current_best_neg = np.inf

        # Get current bests for this lag
        cursor.execute("SELECT correlation_type, correlation_value FROM leaderboard WHERE lag = ?", (lag,))
        rows = cursor.fetchall()
        for corr_type_db, value_db in rows:
            if corr_type_db == 'positive':
                current_best_pos = value_db
            elif corr_type_db == 'negative':
                current_best_neg = value_db

        # Determine correlation type and check if it's a new best
        # (Handles correlation_value == 0 case implicitly by not being > pos or < neg)
        is_new_best = False
        new_corr_type = None
        if correlation_value > 0 and correlation_value > current_best_pos:
            is_new_best = True
            new_corr_type = 'positive'
        elif correlation_value < 0 and correlation_value < current_best_neg:
            is_new_best = True
            new_corr_type = 'negative'

        if is_new_best:
            old_best_val = current_best_pos if new_corr_type == 'positive' else current_best_neg
            logger.info(f"LEADERBOARD UPDATE (Lag: {lag}, Type: {new_corr_type}): New Best Corr={correlation_value:.6f} (beats {old_best_val:.6f}) by Ind '{indicator_name}' (ID: {config_id})")
            config_json = json.dumps(params, sort_keys=True, separators=(',', ':'))
            # Use millisecond precision for timestamp
            calculation_ts = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
            query = """INSERT OR REPLACE INTO leaderboard (lag, correlation_type, correlation_value, indicator_name, config_json,
                         symbol, timeframe, dataset_daterange, calculation_timestamp, config_id_source_db, source_db_name)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"""
            cursor.execute(query, (lag, new_corr_type, correlation_value, indicator_name, config_json,
                                  symbol, timeframe, data_daterange, calculation_ts, config_id, source_db_name))
            conn.commit() # Commit the single update
            updated = True
        # else:
        #     logger.debug(f"Leaderboard check (Lag: {lag}): Corr {correlation_value:.4f} not better than current bests (Pos: {current_best_pos:.4f}, Neg: {current_best_neg:.4f}).")

    except sqlite3.Error as e:
        logger.error(f"Error checking/updating leaderboard for lag {lag}: {e}", exc_info=True)
        try:
            conn.rollback()
        except Exception as rb_err:
            logger.error(f"Rollback failed: {rb_err}")
    finally:
        if conn:
            conn.close()

    # Export leaderboard text file AFTER DB connection is closed if an update happened
    if updated:
        try:
            export_success = export_leaderboard_to_text()
            if not export_success:
                logger.error("Failed to export leaderboard.txt after real-time update.")
        except Exception as export_err:
            logger.error(f"Error triggering leaderboard export after update: {export_err}", exc_info=True)

    return updated

# --- Update Leaderboard Logic (Original - Still used for Default Path) ---
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
    NOTE: This is less critical for Tweak path due to real-time updates, but essential for Default path
    or as a final consistency check. It also triggers leaderboard.txt export.
    """
    logger.info(f"Starting leaderboard update comparison (Batch Mode) for {symbol}_{timeframe} (Max Lag: {max_lag})...")

    config_details_map = {cfg['config_id']: cfg for cfg in indicator_configs if 'config_id' in cfg}
    updates_made_in_batch = 0

    # This loop now primarily calls the single-lag update function for simplicity and consistency
    for lag in range(1, max_lag + 1):
        for config_id, correlations in current_run_correlations.items():
            if correlations and len(correlations) >= lag:
                value = correlations[lag-1]
                if pd.notna(value) and isinstance(value, (float, int)):
                    config_info = config_details_map.get(config_id)
                    if config_info:
                        indicator_name = config_info.get('indicator_name', 'Unknown')
                        params = config_info.get('params', {})
                        # Call the single-lag check/update function
                        # This handles DB connection, comparison, commit, and export trigger internally
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
                            updates_made_in_batch += 1
                    else:
                        logger.warning(f"Config details missing for ID {config_id} during batch update loop (lag {lag}).")

    if updates_made_in_batch > 0:
        logger.info(f"Leaderboard batch comparison loop finished. Triggered {updates_made_in_batch} real-time updates/exports.")
    else:
        logger.info("Leaderboard batch comparison loop finished. No updates triggered (already optimal or real-time updates handled it).")
    # Explicit final export call after batch mode is redundant now, as check_and_update handles it.


# --- Export Leaderboard to Text File ---
def export_leaderboard_to_text() -> bool:
    """Exports the current leaderboard to 'leaderboard.txt' in the reports directory."""
    logger.info("Exporting leaderboard to text file...")
    conn = _create_leaderboard_connection()
    if conn is None: logger.error("Cannot connect to leaderboard DB for export."); return False

    output_filepath = config.REPORTS_DIR / "leaderboard.txt"
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    # Use millisecond precision in header
    current_timestamp_str = datetime.now(timezone.utc).isoformat(timespec='milliseconds')

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='leaderboard';")
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
        if rows:
            max_lag_db = max(row[0] for row in rows)

        # Aggregate data by lag, ensuring all lags up to max_lag_db are represented
        aggregated_data: Dict[int, Dict[str, Any]] = {lag: {'lag': lag} for lag in range(1, max_lag_db + 1)}
        for row in rows:
            lag, corr_type, value, ind_name, cfg_json, sym, tf, dr, ts, cfg_id, src_db = row
            if lag not in aggregated_data: aggregated_data[lag] = {'lag': lag} # Should not happen with pre-population, but safe
            prefix = "pos_" if corr_type == "positive" else "neg_"
            aggregated_data[lag].update({
                f'{prefix}value': value, f'{prefix}indicator': ind_name, f'{prefix}params': cfg_json,
                f'{prefix}symbol': sym, f'{prefix}timeframe': tf, f'{prefix}dataset': dr,
                f'{prefix}timestamp': ts, f'{prefix}config_id': cfg_id, f'{prefix}source_db': src_db
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
        # --- FIX: Handle potential None values before formatting ---
        df['pos_value'] = df['pos_value'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else na_rep)
        df['neg_value'] = df['neg_value'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else na_rep)
        df['pos_config_id'] = df['pos_config_id'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else na_rep)
        df['neg_config_id'] = df['neg_config_id'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else na_rep)
        # ----------------------------------------------------------
        df['pos_params'] = df['pos_params'].fillna(na_rep).apply(lambda x: x if x == na_rep or len(x) <= max_json_len else x[:max_json_len-3] + '...')
        df['neg_params'] = df['neg_params'].fillna(na_rep).apply(lambda x: x if x == na_rep or len(x) <= max_json_len else x[:max_json_len-3] + '...')
        for col in df.columns:
             if 'timestamp' in col:
                  # Ensure parsing handles potential timezone info correctly and output milliseconds
                  df[col] = pd.to_datetime(df[col], errors='coerce', utc=True).dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3].fillna(na_rep)
             elif col not in ['lag', 'pos_value', 'neg_value', 'pos_config_id', 'neg_config_id']:
                 df[col] = df[col].fillna(na_rep)

        output_string = f"Correlation Leaderboard - Last Updated: {current_timestamp_str}\n"
        separator_len = 250 # Increased width for timestamp
        output_string += "=" * separator_len + "\n"
        with pd.option_context('display.width', separator_len + 10):
            # Ensure all columns are included even if completely NaN
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

    best_entry: Optional[Dict[str, Any]] = None; max_abs_corr = -1.0

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='leaderboard';")
        if cursor.fetchone() is None: logger.warning("Leaderboard table not found."); return None

        # Use ABS() and ORDER BY DESC directly in the query
        query = "SELECT * FROM leaderboard WHERE lag = ? ORDER BY ABS(correlation_value) DESC LIMIT 1"
        cursor.execute(query, (target_lag,))
        row = cursor.fetchone()

        if not row: logger.warning(f"No entries found for Lag = {target_lag}."); return None

        lag, corr_type, value, ind_name, cfg_json, sym, tf, dr, ts, cfg_id, src_db = row
        current_corr_value = None
        if isinstance(value, (int, float)) and pd.notna(value): current_corr_value = float(value)
        else: logger.error(f"Non-numeric/NaN corr for Best Lag {lag}, Ind {ind_name}: {value}."); return None

        try:
            params_dict = json.loads(cfg_json) # Parse JSON params
            best_entry = { # Store parsed params dict
                'lag': lag, 'correlation_type': corr_type, 'correlation_value': current_corr_value,
                'indicator_name': ind_name, 'params': params_dict, 'config_json': cfg_json,
                'symbol': sym, 'timeframe': tf, 'dataset_daterange': dr, 'calculation_timestamp': ts,
                'config_id_source_db': cfg_id, 'source_db_name': src_db
            }
            max_abs_corr = abs(current_corr_value)
            logger.info(f"Found best predictor for Lag {target_lag}: {best_entry['indicator_name']} (ID: {best_entry['config_id_source_db']}), Abs Corr: {max_abs_corr:.6f}")
            return best_entry
        except json.JSONDecodeError:
            logger.error(f"Failed parse config_json for predictor candidate (Lag {lag}, Ind {ind_name}): {cfg_json}")
            return None # Skip if JSON is bad

    except Exception as e:
        logger.error(f"Error searching leaderboard for predictor: {e}", exc_info=True)
        return None
    finally:
        if conn: conn.close()


# --- NEW FUNCTION: Generate Leading Indicator Report ---
def generate_leading_indicator_report() -> bool:
    """Generates a report tallying which indicators lead most often."""
    logger.info("Generating leading indicators report...")
    conn = _create_leaderboard_connection()
    if conn is None:
        logger.error("Cannot connect to leaderboard DB for leading indicator report.")
        return False

    output_filepath = config.REPORTS_DIR / "leading_indicators.txt"
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    current_timestamp_str = datetime.now(timezone.utc).isoformat(timespec='milliseconds')

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='leaderboard';")
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

        # Use Counter for efficient tallying
        pos_counts = Counter()
        neg_counts = Counter()

        # Store unique config details for final report
        config_details_map = {} # Key: (indicator_name, config_id)

        for ind_name, cfg_json, corr_type, cfg_id in rows:
            if cfg_id is None: # Should not happen with current schema, but check
                logger.warning(f"Skipping leaderboard entry with null config_id for indicator '{ind_name}'.")
                continue
            key = (ind_name, cfg_id)
            if key not in config_details_map:
                 config_details_map[key] = {'params_json': cfg_json} # Store params for later display

            if corr_type == 'positive':
                pos_counts[key] += 1
            elif corr_type == 'negative':
                neg_counts[key] += 1

        # Combine counts into a list for DataFrame conversion
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
        with pd.option_context('display.width', 1000, 'display.max_colwidth', 70): # Adjust width/colwidth
            output_string += df_report.to_string(index=False, justify='left')
        output_string += "\n" + "=" * 110

        output_filepath.write_text(output_string, encoding='utf-8')
        logger.info(f"Leading indicators report saved to: {output_filepath}")
        print(f"\nLeading indicators report saved to: {output_filepath}")
        return True

    except Exception as e:
        logger.error(f"Error generating leading indicators report: {e}", exc_info=True)
        try:
            output_filepath.write_text(f"ERROR generating leading indicators report: {e}\n", encoding='utf-8')
        except: pass
        return False
    finally:
        if conn: conn.close()