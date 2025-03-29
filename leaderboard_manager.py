# leaderboard_manager.py
import sqlite3
import logging
from datetime import datetime, timezone
import json
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

import config # For LEADERBOARD_DB_PATH and REPORTS_DIR
import utils # For compare_param_dicts

logger = logging.getLogger(__name__)

# --- Database Functions (_create_leaderboard_connection, initialize_leaderboard_db, load_leaderboard) ---
# (Keep existing functions - no changes needed here)
def _create_leaderboard_connection() -> Optional[sqlite3.Connection]:
    """Creates a connection to the leaderboard SQLite database."""
    conn = None
    try:
        logger.debug(f"Connecting to leaderboard database: {config.LEADERBOARD_DB_PATH}")
        conn = sqlite3.connect(config.LEADERBOARD_DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys = OFF;") # Leaderboard is standalone
        logger.debug(f"Successfully connected to leaderboard database.")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to leaderboard database '{config.LEADERBOARD_DB_PATH}': {e}", exc_info=True)
        return None

def initialize_leaderboard_db() -> bool:
    """Initializes the leaderboard database table if it doesn't exist."""
    conn = _create_leaderboard_connection()
    if conn is None: return False
    try:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS leaderboard (
            lag INTEGER NOT NULL,
            correlation_type TEXT NOT NULL CHECK(correlation_type IN ('positive', 'negative')),
            correlation_value REAL NOT NULL,
            indicator_name TEXT NOT NULL,
            config_json TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            dataset_daterange TEXT NOT NULL, -- Format: YYYYMMDD-YYYYMMDD
            calculation_timestamp TEXT NOT NULL, -- ISO 8601 Format
            config_id_source_db INTEGER, -- Optional: ID from the analysis DB
            source_db_name TEXT,      -- Optional: Filename of the analysis DB
            PRIMARY KEY (lag, correlation_type)
        );""")
        conn.commit()
        logger.info(f"Leaderboard database schema initialized successfully: {config.LEADERBOARD_DB_PATH}")
        return True
    except sqlite3.Error as e:
        logger.error(f"Error initializing leaderboard database schema: {e}", exc_info=True)
        try: conn.rollback()
        except Exception: pass
        return False
    finally:
        if conn: conn.close()

def load_leaderboard() -> Dict[Tuple[int, str], Dict[str, Any]]:
    """Loads the current leaderboard data into a dictionary."""
    leaderboard_data = {}
    conn = _create_leaderboard_connection()
    if conn is None:
        logger.warning("Could not connect to load leaderboard. Returning empty.")
        return {}
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='leaderboard';")
        if cursor.fetchone() is None:
            logger.warning("Leaderboard table does not exist. Returning empty leaderboard.")
            return {}

        cursor.execute("SELECT lag, correlation_type, correlation_value, indicator_name, config_json, symbol, timeframe, dataset_daterange, calculation_timestamp, config_id_source_db, source_db_name FROM leaderboard")
        rows = cursor.fetchall()
        for row in rows:
            lag, corr_type, value, ind_name, cfg_json, sym, tf, dr, ts, cfg_id, src_db = row
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

# --- Leaderboard Update Logic (update_leaderboard) ---
# (Keep existing function - no changes needed here)
def update_leaderboard(
    current_run_correlations: Dict[int, List[Optional[float]]],
    indicator_configs: List[Dict[str, Any]],
    max_lag: int,
    symbol: str,
    timeframe: str,
    data_daterange: str, # Format "YYYYMMDD-YYYYMMDD"
    source_db_name: str
) -> None:
    """
    Compares current run results against the persistent leaderboard and updates it.
    """
    logger.info(f"Starting leaderboard update process for {symbol}_{timeframe} (Max Lag: {max_lag})...")
    existing_leaderboard = load_leaderboard()
    if not isinstance(existing_leaderboard, dict):
        logger.error("Failed to load existing leaderboard correctly. Aborting update.")
        return

    config_details_map = {cfg['config_id']: cfg for cfg in indicator_configs}
    updates_to_make = []
    calculation_ts = datetime.now(timezone.utc).isoformat(timespec='seconds')

    for lag in range(1, max_lag + 1):
        best_pos_in_run = {'config_id': None, 'value': -float('inf')}
        best_neg_in_run = {'config_id': None, 'value': float('inf')}

        for config_id, correlations in current_run_correlations.items():
            if correlations and len(correlations) >= lag:
                value = correlations[lag-1]
                if pd.notna(value):
                    if value > best_pos_in_run['value']: best_pos_in_run = {'config_id': config_id, 'value': value}
                    if value < best_neg_in_run['value']: best_neg_in_run = {'config_id': config_id, 'value': value}

        current_best_pos_val = existing_leaderboard.get((lag, 'positive'), {}).get('correlation_value', -float('inf'))
        if best_pos_in_run['config_id'] is not None and best_pos_in_run['value'] > current_best_pos_val:
            logger.info(f"New Leaderboard Record (Lag: {lag}, Type: positive): Value {best_pos_in_run['value']:.4f} > {current_best_pos_val:.4f}")
            config_id = best_pos_in_run['config_id']
            details = config_details_map.get(config_id)
            if details:
                 updates_to_make.append((
                     lag, 'positive', best_pos_in_run['value'],
                     details['indicator_name'], json.dumps(details['params'], sort_keys=True, separators=(',',':')),
                     symbol, timeframe, data_daterange, calculation_ts,
                     config_id, source_db_name
                 ))
            else: logger.warning(f"Could not find details for config_id {config_id} for positive leaderboard update at lag {lag}.")

        current_best_neg_val = existing_leaderboard.get((lag, 'negative'), {}).get('correlation_value', float('inf'))
        if best_neg_in_run['config_id'] is not None and best_neg_in_run['value'] < current_best_neg_val:
            logger.info(f"New Leaderboard Record (Lag: {lag}, Type: negative): Value {best_neg_in_run['value']:.4f} < {current_best_neg_val:.4f}")
            config_id = best_neg_in_run['config_id']
            details = config_details_map.get(config_id)
            if details:
                 updates_to_make.append((
                     lag, 'negative', best_neg_in_run['value'],
                     details['indicator_name'], json.dumps(details['params'], sort_keys=True, separators=(',',':')),
                     symbol, timeframe, data_daterange, calculation_ts,
                     config_id, source_db_name
                 ))
            else: logger.warning(f"Could not find details for config_id {config_id} for negative leaderboard update at lag {lag}.")

    if updates_to_make:
        conn = _create_leaderboard_connection()
        if conn is None:
            logger.error("Failed to connect to leaderboard DB to save updates.")
            return
        try:
            cursor = conn.cursor()
            query = """
            INSERT OR REPLACE INTO leaderboard
            (lag, correlation_type, correlation_value, indicator_name, config_json, symbol, timeframe, dataset_daterange, calculation_timestamp, config_id_source_db, source_db_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """
            cursor.executemany(query, updates_to_make)
            conn.commit()
            logger.info(f"Successfully updated {len(updates_to_make)} leaderboard entries.")
        except sqlite3.Error as e:
            logger.error(f"Error updating leaderboard database: {e}", exc_info=True)
            try: conn.rollback()
            except Exception: pass
        finally:
            if conn: conn.close()
    else:
        logger.info("No leaderboard updates found in this run.")


# --- UPDATED FUNCTION ---
def export_leaderboard_to_text() -> bool: # Removed file_prefix parameter
    """Exports the current leaderboard to a fixed 'leaderboard.txt' file in the reports directory."""
    logger.info("Exporting leaderboard to text file...")
    conn = _create_leaderboard_connection()
    if conn is None:
        logger.error("Cannot connect to leaderboard database for export.")
        return False

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='leaderboard';")
        if cursor.fetchone() is None:
            logger.warning("Leaderboard table not found. Cannot export.")
            return True

        query = "SELECT * FROM leaderboard ORDER BY lag ASC, correlation_type ASC"
        cursor.execute(query)
        rows = cursor.fetchall()

        if not rows:
            logger.info("Leaderboard is empty. No text file generated.")
            return True

        aggregated_data = {}
        for row in rows:
            lag, corr_type, value, ind_name, cfg_json, sym, tf, dr, ts, cfg_id, src_db = row
            if lag not in aggregated_data:
                aggregated_data[lag] = {'lag': lag}

            prefix = "pos_" if corr_type == "positive" else "neg_"
            aggregated_data[lag][f'{prefix}value'] = value
            aggregated_data[lag][f'{prefix}indicator'] = ind_name
            aggregated_data[lag][f'{prefix}params'] = cfg_json
            aggregated_data[lag][f'{prefix}symbol'] = sym
            aggregated_data[lag][f'{prefix}timeframe'] = tf
            aggregated_data[lag][f'{prefix}dataset'] = dr
            aggregated_data[lag][f'{prefix}timestamp'] = ts

        report_list = list(aggregated_data.values())
        df = pd.DataFrame(report_list)
        cols = ['lag']
        pos_cols = ['pos_value', 'pos_indicator', 'pos_params', 'pos_symbol', 'pos_timeframe', 'pos_dataset', 'pos_timestamp']
        neg_cols = ['neg_value', 'neg_indicator', 'neg_params', 'neg_symbol', 'neg_timeframe', 'neg_dataset', 'neg_timestamp']
        all_expected_cols = cols + pos_cols + neg_cols
        for col in all_expected_cols:
            if col not in df.columns: df[col] = None
        df = df[all_expected_cols]

        na_rep = 'N/A'
        df['pos_value'] = df['pos_value'].map('{:.6f}'.format).fillna(na_rep)
        df['neg_value'] = df['neg_value'].map('{:.6f}'.format).fillna(na_rep)

        max_json_len = 60
        df['pos_params'] = df['pos_params'].fillna(na_rep).apply(lambda x: x if x == na_rep or len(x) <= max_json_len else x[:max_json_len-3] + '...')
        df['neg_params'] = df['neg_params'].fillna(na_rep).apply(lambda x: x if x == na_rep or len(x) <= max_json_len else x[:max_json_len-3] + '...')
        for col in df.columns:
             if col not in ['lag', 'pos_value', 'neg_value']: df[col] = df[col].fillna(na_rep)

        output_string = f"Correlation Leaderboard - Last Updated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n" # Changed title
        output_string += "=" * 200 + "\n"
        output_string += df.to_string( index=False, justify='left', max_colwidth=None ) # Allow wider columns
        output_string += "\n" + "=" * 200

        # <<< Use fixed filename >>>
        output_filename = "leaderboard.txt"
        # <<< End change >>>
        output_filepath = config.REPORTS_DIR / output_filename
        output_filepath.parent.mkdir(parents=True, exist_ok=True)

        output_filepath.write_text(output_string, encoding='utf-8')
        logger.info(f"Leaderboard successfully exported to: {output_filepath}")
        print(f"\nLeaderboard exported to: {output_filepath}")
        return True

    except sqlite3.Error as e:
        logger.error(f"Database error exporting leaderboard: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error exporting leaderboard: {e}", exc_info=True)
        return False
    finally:
        if conn: conn.close()