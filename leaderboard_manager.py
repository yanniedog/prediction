# leaderboard_manager.py
import sqlite3
import logging
from datetime import datetime, timezone
import json
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from collections import Counter # For tallying
from pathlib import Path
from scipy.stats import linregress # Used in consistency report

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
        # Use autocommit mode off, increased timeout, add busy timeout
        conn = sqlite3.connect(str(config.LEADERBOARD_DB_PATH), timeout=15.0, isolation_level=None) # isolation_level=None enables autocommit unless BEGIN is used
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys = OFF;") # Leaderboard is standalone, FKs not critical here
        conn.execute("PRAGMA busy_timeout = 10000;") # Wait 10s if DB is locked
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
        # Use DEFERRED transaction for schema init (less locking)
        conn.execute("BEGIN DEFERRED;")
        # Use millisecond precision for timestamp storage (TEXT ISO8601 'YYYY-MM-DDTHH:MM:SS.sssZ')
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
            calculation_timestamp TEXT NOT NULL, -- Store as ISO 8601 text with Z
            config_id_source_db INTEGER, -- Store the ID from the source symbol DB
            source_db_name TEXT, -- Store the name of the source symbol DB
            PRIMARY KEY (lag, correlation_type)
        );""")
        # Add indices if they don't exist for faster lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_leaderboard_lag ON leaderboard(lag);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_leaderboard_lag_val ON leaderboard(lag, correlation_value);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_leaderboard_indicator_config ON leaderboard(indicator_name, config_id_source_db);")

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

        # Select all columns
        cursor.execute("""
            SELECT lag, correlation_type, correlation_value, indicator_name, config_json,
                   symbol, timeframe, dataset_daterange, calculation_timestamp,
                   config_id_source_db, source_db_name
            FROM leaderboard
        """)
        rows = cursor.fetchall()
        loaded_count = 0
        for row in rows:
            # Ensure essential keys are present before adding
            if row['lag'] is not None and row['correlation_type'] is not None:
                key = (row['lag'], row['correlation_type'])
                # Convert Row object to a standard dictionary for external use
                leaderboard_data[key] = dict(row)
                loaded_count += 1
            else:
                logger.warning(f"Skipping leaderboard row with missing lag/type: {dict(row)}")
        logger.info(f"Loaded {loaded_count} valid entries from leaderboard.")
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
    config_id: int, # ID from the source symbol/timeframe DB
    symbol: str,
    timeframe: str,
    data_daterange: str,
    source_db_name: str # Filename of the source DB
) -> bool:
    """
    Checks if the provided correlation is a new best for the given lag
    and updates the leaderboard DB immediately if it is. Also triggers
    leaderboard.txt export on update. Returns True if an update occurred.
    """
    # Validate input correlation value
    if not pd.notna(correlation_value) or not isinstance(correlation_value, (float, int)):
        logger.debug(f"Skipping leaderboard check: Non-numeric correlation {correlation_value} (Lag {lag}, ID {config_id})")
        return False
    if config_id is None or not isinstance(config_id, int): # Ensure we have a valid source config ID
         logger.error(f"Skipping leaderboard check: Invalid/missing source config_id ({config_id}) (Lag {lag}, Ind {indicator_name})")
         return False

    conn = _create_leaderboard_connection()
    if conn is None:
        logger.error("Cannot connect to leaderboard DB for single lag update.")
        return False

    updated = False
    try:
        cursor = conn.cursor()
        # Use IMMEDIATE transaction for potentially better concurrency during updates
        cursor.execute("BEGIN IMMEDIATE;")
        current_best_pos = -np.inf
        current_best_neg = np.inf

        # Get current bests for this lag from the database
        cursor.execute("SELECT correlation_type, correlation_value FROM leaderboard WHERE lag = ?", (lag,))
        rows = cursor.fetchall()
        for row in rows:
            # Ensure value from DB is not NULL and is numeric before comparison
            if row['correlation_value'] is not None:
                try:
                    db_val = float(row['correlation_value'])
                    if row['correlation_type'] == 'positive':
                        current_best_pos = max(current_best_pos, db_val)
                    elif row['correlation_type'] == 'negative':
                        current_best_neg = min(current_best_neg, db_val)
                except (ValueError, TypeError):
                     logger.warning(f"Non-numeric value found in leaderboard DB for lag {lag}: {row['correlation_value']}. Skipping comparison for this row.")

        # Determine correlation type and check if the current value is a new best
        is_new_best = False
        new_corr_type = None
        current_val_float = float(correlation_value) # Ensure float for comparison

        # Using direct comparison (tolerance likely not needed with REAL type in SQLite)
        if current_val_float > 0 and current_val_float > current_best_pos:
            is_new_best = True
            new_corr_type = 'positive'
        elif current_val_float < 0 and current_val_float < current_best_neg:
            is_new_best = True
            new_corr_type = 'negative'

        # If it's a new best, update the database
        if is_new_best:
            old_best_val_str = f"{current_best_pos:.6f}" if new_corr_type == 'positive' else f"{current_best_neg:.6f}"
            logger.info(f"LEADERBOARD UPDATE (Lag {lag}, Type {new_corr_type}): New Best Corr={current_val_float:.6f} (beats {old_best_val_str}) by Ind '{indicator_name}' (SrcID: {config_id})")
            try:
                 # Serialize params using the consistent utils hasher method
                 config_json = json.dumps(utils.round_floats_for_hashing(params), sort_keys=True, separators=(',', ':'))
            except TypeError as json_err:
                 logger.error(f"Cannot serialize params to JSON for leaderboard update (Lag {lag}, ID {config_id}): {params}. Error: {json_err}")
                 conn.rollback()
                 return False # Cannot store invalid JSON

            # Get current UTC time in ISO format with Z
            calculation_ts = datetime.now(timezone.utc).isoformat(timespec='milliseconds') + 'Z'
            query = """INSERT OR REPLACE INTO leaderboard (lag, correlation_type, correlation_value, indicator_name, config_json,
                         symbol, timeframe, dataset_daterange, calculation_timestamp, config_id_source_db, source_db_name)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"""
            cursor.execute(query, (lag, new_corr_type, current_val_float, indicator_name, config_json,
                                  symbol, timeframe, data_daterange, calculation_ts, config_id, source_db_name))
            conn.commit() # Commit the single update
            updated = True
        # else: # Reduce log noise, only log if it IS a new best
        #     pass

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
            # This function now logs success/failure internally and doesn't print to console
            export_leaderboard_to_text()
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
    source_db_name: str,
    min_correlation_threshold: float = 0.0,
    max_updates_per_config: Optional[int] = None
) -> Dict[int, Dict[str, Any]]:
    """Update leaderboard with new correlation results.
    
    Args:
        current_run_correlations: Dictionary mapping config IDs to lists of correlation values for each lag
        indicator_configs: List of indicator configurations used in the current run
        max_lag: Maximum lag value used in the analysis
        symbol: Symbol being analyzed
        timeframe: Timeframe being analyzed
        data_daterange: Date range of the dataset
        source_db_name: Name of the source database
        min_correlation_threshold: Minimum correlation value to consider (default: 0.0)
        max_updates_per_config: Maximum number of updates to allow per config (default: None)
        
    Returns:
        Dictionary mapping config IDs to their best correlation results
    """
    try:
        # Validate inputs
        if not current_run_correlations or not indicator_configs:
            logger.warning("No correlations or configs provided for leaderboard update")
            return {}
        
        if not isinstance(current_run_correlations, dict):
            raise ValueError("current_run_correlations must be a dictionary")
        
        if not isinstance(indicator_configs, list):
            raise ValueError("indicator_configs must be a list")
        
        # Create connection
        conn = _create_leaderboard_connection()
        if not conn:
            raise ValueError("Failed to create database connection")
        
        cursor = conn.cursor()
        updates = {}
        
        try:
            # Process each configuration
            for config in indicator_configs:
                config_id = config.get('config_id')
                if not config_id:
                    logger.warning("Skipping config without ID")
                    continue
                
                if config_id not in current_run_correlations:
                    logger.warning(f"No correlations found for config {config_id}")
                    continue
                
                correlations = current_run_correlations[config_id]
                if not correlations:
                    continue
                
                # Get current best correlations for this config
                cursor.execute("""
                    SELECT lag, correlation_type, correlation_value
                    FROM leaderboard
                    WHERE config_id_source_db = ? AND symbol = ? AND timeframe = ?
                    ORDER BY correlation_value DESC
                """, (config_id, symbol, timeframe))
                
                current_best = {
                    (row['lag'], row['correlation_type']): row['correlation_value']
                    for row in cursor.fetchall()
                }
                
                # Process each lag's correlations
                config_updates = []
                for lag, corr_value in enumerate(correlations, 1):
                    if lag > max_lag:
                        break
                    
                    if corr_value is None or abs(corr_value) < min_correlation_threshold:
                        continue
                    
                    # Determine correlation type
                    corr_type = 'positive' if corr_value > 0 else 'negative'
                    abs_corr = abs(corr_value)
                    
                    # Check if this is better than current best
                    current_best_key = (lag, corr_type)
                    current_best_value = current_best.get(current_best_key, 0.0)
                    
                    if abs_corr > current_best_value:
                        # Prepare update data
                        update_data = {
                            'lag': lag,
                            'correlation_type': corr_type,
                            'correlation_value': abs_corr,
                            'indicator_name': config['indicator_name'],
                            'config_json': json.dumps(config['params']),
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'dataset_daterange': data_daterange,
                            'calculation_timestamp': datetime.now().isoformat(),
                            'config_id_source_db': config_id,
                            'source_db_name': source_db_name
                        }
                        
                        # Update or insert record
                        cursor.execute("""
                            INSERT INTO leaderboard (
                                lag, correlation_type, correlation_value,
                                indicator_name, config_json, symbol, timeframe,
                                dataset_daterange, calculation_timestamp,
                                config_id_source_db, source_db_name
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT (lag, correlation_type)
                            DO UPDATE SET
                                correlation_value = excluded.correlation_value,
                                indicator_name = excluded.indicator_name,
                                config_json = excluded.config_json,
                                symbol = excluded.symbol,
                                timeframe = excluded.timeframe,
                                dataset_daterange = excluded.dataset_daterange,
                                calculation_timestamp = excluded.calculation_timestamp,
                                config_id_source_db = excluded.config_id_source_db,
                                source_db_name = excluded.source_db_name
                            WHERE excluded.correlation_value > leaderboard.correlation_value
                        """, (
                            update_data['lag'], update_data['correlation_type'],
                            update_data['correlation_value'], update_data['indicator_name'],
                            update_data['config_json'], update_data['symbol'],
                            update_data['timeframe'], update_data['dataset_daterange'],
                            update_data['calculation_timestamp'], update_data['config_id_source_db'],
                            update_data['source_db_name']
                        ))
                        
                        config_updates.append(update_data)
                
                # Apply updates if within limit
                if config_updates:
                    if max_updates_per_config is None or len(config_updates) <= max_updates_per_config:
                        conn.commit()
                        updates[config_id] = {
                            'updates': config_updates,
                            'best_correlation': max(update['correlation_value'] for update in config_updates)
                        }
                    else:
                        logger.warning(f"Too many updates ({len(config_updates)}) for config {config_id}")
                        conn.rollback()
        
        except sqlite3.Error as e:
            logger.error(f"Database error during leaderboard update: {e}")
            conn.rollback()
            raise
        
        finally:
            conn.close()
        
        return updates
        
    except Exception as e:
        logger.error(f"Error updating leaderboard: {e}", exc_info=True)
        raise


# --- Export Leaderboard to Text File ---
def export_leaderboard_to_text() -> bool:
    """Exports the current leaderboard to 'leaderboard.txt' in the reports directory."""
    logger.info("Exporting leaderboard to text file...")
    conn = _create_leaderboard_connection()
    if conn is None: logger.error("Cannot connect to leaderboard DB for export."); return False

    # Use config variable for output path
    output_filepath = config.REPORTS_DIR / "leaderboard.txt"
    output_filepath.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    current_timestamp_str = datetime.now(timezone.utc).isoformat(timespec='milliseconds') + 'Z'

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='leaderboard' LIMIT 1;")
        if cursor.fetchone() is None:
            logger.warning("Leaderboard table not found. Exporting empty file.")
            output_filepath.write_text(f"Leaderboard Table Not Found - Checked: {current_timestamp_str}\n", encoding='utf-8')
            return True

        # Fetch all current entries, ordered for consistency
        query = "SELECT * FROM leaderboard ORDER BY lag ASC, correlation_type ASC"
        cursor.execute(query)
        rows = cursor.fetchall()

        if not rows:
            logger.info("Leaderboard is empty. Exporting empty file.")
            output_filepath.write_text(f"Leaderboard Empty - Last Checked: {current_timestamp_str}\n", encoding='utf-8')
            return True

        # Determine the maximum lag present in the data to structure the output DataFrame
        max_lag_db = 0
        try: max_lag_db = max(row['lag'] for row in rows if row['lag'] is not None)
        except ValueError: max_lag_db = 0 # Handle empty sequence if all lags were None

        # Aggregate data by lag, ensuring all lags up to max_lag_db are represented
        aggregated_data: Dict[int, Dict[str, Any]] = {lag: {'lag': lag} for lag in range(1, max_lag_db + 1)}
        for row in rows:
            lag = row['lag']
            # Skip rows with invalid lag somehow missed earlier
            if lag is None or lag <= 0: continue
            if lag not in aggregated_data: aggregated_data[lag] = {'lag': lag} # Should not happen, but safe

            prefix = "pos_" if row['correlation_type'] == "positive" else "neg_"
            # Update the dictionary for the specific lag
            aggregated_data[lag].update({
                f'{prefix}value': row['correlation_value'],
                f'{prefix}indicator': row['indicator_name'],
                f'{prefix}params': row['config_json'],
                f'{prefix}symbol': row['symbol'],
                f'{prefix}timeframe': row['timeframe'],
                f'{prefix}dataset': row['dataset_daterange'],
                f'{prefix}timestamp': row['calculation_timestamp'], # Store raw timestamp string
                f'{prefix}config_id': row['config_id_source_db'],
                f'{prefix}source_db': row['source_db_name']
            })

        df = pd.DataFrame(list(aggregated_data.values()))
        # Ensure consistent column order for the output text file
        cols = ['lag']
        pos_cols = ['pos_value', 'pos_indicator', 'pos_params', 'pos_symbol', 'pos_timeframe', 'pos_dataset', 'pos_config_id', 'pos_source_db', 'pos_timestamp']
        neg_cols = ['neg_value', 'neg_indicator', 'neg_params', 'neg_symbol', 'neg_timeframe', 'neg_dataset', 'neg_config_id', 'neg_source_db', 'neg_timestamp']
        all_expected_cols = cols + pos_cols + neg_cols
        # Add missing columns with None before formatting (handles cases where only pos or neg exists for a lag)
        for col in all_expected_cols:
            if col not in df.columns: df[col] = None
        df = df[all_expected_cols] # Reorder

        # Formatting for readability
        na_rep = 'N/A'; max_json_len = 50
        float_cols = ['pos_value', 'neg_value']
        id_cols = ['pos_config_id', 'neg_config_id']
        param_cols = ['pos_params', 'neg_params']
        ts_cols = ['pos_timestamp', 'neg_timestamp']

        # Apply formatting safely
        for col in float_cols: df[col] = df[col].apply(lambda x: f"{x:.6f}" if pd.notna(x) and isinstance(x, (float, int, np.number)) else na_rep)
        for col in id_cols: df[col] = df[col].apply(lambda x: f"{int(x)}" if pd.notna(x) and isinstance(x, (float, int, np.number)) else na_rep)
        for col in param_cols: df[col] = df[col].fillna(na_rep).apply(lambda x: x if x == na_rep or len(str(x)) <= max_json_len else str(x)[:max_json_len-3] + '...')
        for col in ts_cols:
            # Convert ISO string timestamps to a more readable format for the report
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce', utc=True).dt.strftime('%Y-%m-%d %H:%M:%S').fillna(na_rep)
            except Exception: # Fallback if conversion fails
                 df[col] = df[col].fillna(na_rep)

        # Fill NA for remaining object columns (like text fields)
        for col in df.select_dtypes(include=['object']).columns:
            if col not in float_cols + id_cols + param_cols + ts_cols:
                df[col] = df[col].fillna(na_rep)

        # Generate Output String
        output_string = f"Correlation Leaderboard - Last Updated: {current_timestamp_str}\n"
        separator_len = 250 # Ensure enough width
        output_string += "=" * separator_len + "\n"
        with pd.option_context('display.width', separator_len + 10, 'display.max_colwidth', 60): # Adjust max_colwidth
            output_string += df.to_string( index=False, justify='left', na_rep=na_rep )
        output_string += "\n" + "=" * separator_len

        # Write to file
        output_filepath.write_text(output_string, encoding='utf-8')
        logger.info(f"Leaderboard successfully exported to: {output_filepath}")
        # ** Removed print statement **
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

        if not row:
            logger.warning(f"No valid entries found for Lag = {target_lag}.")
            return None

        # Convert row to dict
        best_entry = dict(row)
        # Ensure correlation value is float
        current_corr_value = float(best_entry['correlation_value'])

        try:
            # Parse JSON params, handle potential errors gracefully
            params_dict = json.loads(best_entry['config_json'])
            best_entry['params'] = params_dict # Replace JSON string with parsed dict
            max_abs_corr = abs(current_corr_value)
            logger.info(f"Found best predictor for Lag {target_lag}: {best_entry['indicator_name']} (SrcID: {best_entry['config_id_source_db']}), Abs Corr: {max_abs_corr:.6f}")
            return best_entry
        except json.JSONDecodeError:
            logger.error(f"Failed parse config_json for predictor candidate (Lag {target_lag}, Ind {best_entry['indicator_name']}): {best_entry['config_json']}")
            return None # Skip if JSON is bad
        except Exception as parse_err:
            logger.error(f"Error processing predictor entry for Lag {target_lag}: {parse_err}")
            return None

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

    # Use config variable for output path
    output_filepath = config.REPORTS_DIR / "leading_indicators_tally.txt"
    output_filepath.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    current_timestamp_str = datetime.now(timezone.utc).isoformat(timespec='milliseconds') + 'Z'

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='leaderboard' LIMIT 1;")
        if cursor.fetchone() is None:
            logger.warning("Leaderboard table not found. Cannot generate leading indicator report.")
            output_filepath.write_text(f"Leaderboard Table Not Found - Checked: {current_timestamp_str}\n", encoding='utf-8')
            return True

        # Fetch necessary data: indicator name, config details, type (pos/neg), source config ID
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
        config_details_map = {} # Key: (indicator_name, config_id) -> stores params_json

        for row in rows:
            # Ensure necessary fields are present and valid
            ind_name = row['indicator_name']
            cfg_json = row['config_json']
            corr_type = row['correlation_type']
            cfg_id = row['config_id_source_db'] # Source DB Config ID

            if cfg_id is None or ind_name is None or cfg_json is None or corr_type is None:
                logger.warning(f"Skipping leaderboard entry with missing fields: {dict(row)}")
                continue
            # Ensure config_id is int
            try: cfg_id_int = int(cfg_id)
            except (ValueError, TypeError):
                logger.warning(f"Skipping entry with non-integer config_id {cfg_id}.")
                continue

            key = (ind_name, cfg_id_int)
            # Store config details only once per unique key
            if key not in config_details_map:
                 config_details_map[key] = {'params_json': cfg_json}

            # Increment counts based on correlation type
            if corr_type == 'positive': pos_counts[key] += 1
            elif corr_type == 'negative': neg_counts[key] += 1

        report_data = []
        # Get all unique keys from both positive and negative counts
        all_keys = set(pos_counts.keys()) | set(neg_counts.keys())

        for key in all_keys:
            ind_name, cfg_id = key
            pos_lead_count = pos_counts.get(key, 0)
            neg_lead_count = neg_counts.get(key, 0)
            total_lead_count = pos_lead_count + neg_lead_count
            # Get params string from the map
            params_json = config_details_map.get(key, {}).get('params_json', '{}')

            report_data.append({
                'Indicator': ind_name,
                'Config ID': cfg_id,
                'Params': params_json, # Keep as JSON string for report
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

        # Write to file
        output_filepath.write_text(output_string, encoding='utf-8')
        logger.info(f"Leading indicators tally report saved to: {output_filepath}")
        # ** Removed print statement **
        return True

    except Exception as e:
        logger.error(f"Error generating leading indicators tally report: {e}", exc_info=True)
        try: output_filepath.write_text(f"ERROR generating leading indicators tally report: {e}\n", encoding='utf-8')
        except: pass
        return False
    finally:
        if conn: conn.close()


# --- Generate Consistency Report ---
def generate_consistency_report(
    correlations_by_config_id: Dict[int, List[Optional[float]]],
    indicator_configs_processed: List[Dict[str, Any]],
    max_lag: int,
    output_dir: Path,
    file_prefix: str,
    abs_corr_threshold: float = 0.15 # Use config default later if needed
) -> bool:
    """
    Generates a report analyzing the consistency of correlation across lags.
    Operates on the provided correlation dictionary.
    """
    logger.info(f"Generating correlation consistency report (Threshold: {abs_corr_threshold})...")
    if not correlations_by_config_id or max_lag <= 0:
        logger.warning("No correlation data or invalid max_lag for consistency report.")
        return False

    output_filepath = output_dir / f"{file_prefix}_consistency_report.txt"
    current_timestamp_str = datetime.now(timezone.utc).isoformat(timespec='milliseconds') + 'Z'
    # Create a quick lookup map for configs
    configs_dict = {cfg['config_id']: cfg for cfg in indicator_configs_processed if 'config_id' in cfg}
    report_data = []
    lags_array = np.arange(1, max_lag + 1) # For regression slope calculation

    for cfg_id, corrs_full in correlations_by_config_id.items():
        # Ensure config ID is valid int
        if not isinstance(cfg_id, int):
            logger.warning(f"Consistency Report: Invalid config_id key type ({type(cfg_id)}). Skipping."); continue

        config_info = configs_dict.get(cfg_id)
        if not config_info:
            logger.warning(f"Consistency Report: Config info missing for ID {cfg_id}. Skipping.")
            continue

        indicator_name = config_info.get('indicator_name', 'Unknown')
        params = config_info.get('params', {})
        try:
            # Use consistent hashing/serialization method
            params_str = json.dumps(utils.round_floats_for_hashing(params), sort_keys=True, separators=(',',':'))
        except Exception:
            params_str = str(params) # Fallback

        # Validate correlation list structure and length
        if corrs_full is None or not isinstance(corrs_full, list) or len(corrs_full) < max_lag:
            # logger.debug(f"Consistency Report: Short/invalid corr data for {indicator_name} (ID {cfg_id}). Skipping.")
            continue

        corrs = corrs_full[:max_lag]
        # Convert to numeric array, coercing errors (like None) to NaN
        corr_array = pd.to_numeric(corrs, errors='coerce')
        valid_mask = ~np.isnan(corr_array)
        valid_corrs = corr_array[valid_mask]
        valid_lags = lags_array[valid_mask]

        if len(valid_corrs) < 2: # Need at least 2 points for std dev, slope etc.
            # logger.debug(f"Consistency Report: Insufficient valid points (<2) for {indicator_name} (ID {cfg_id}). Skipping.")
            continue

        # Calculate stats on valid points
        abs_corrs = np.abs(valid_corrs)
        mean_abs_corr = np.mean(abs_corrs)
        std_dev_corr = np.std(valid_corrs) # Std dev of original correlations
        lags_over_thresh = np.sum(abs_corrs > abs_corr_threshold)

        # Calculate correlation slope (Corr vs Lag) using only valid points
        corr_slope = np.nan
        try:
             if len(valid_lags) >= 2:
                  # Use linregress for slope, intercept, r_value, p_value, stderr
                  slope_res = linregress(valid_lags, valid_corrs)
                  corr_slope = slope_res.slope if pd.notna(slope_res.slope) else np.nan
             # else: corr_slope remains np.nan
        except Exception as slope_err:
             logger.warning(f"Could not calculate correlation slope for ID {cfg_id}: {slope_err}")
             corr_slope = np.nan

        # Calculate sign flips on valid, non-zero points
        sign_flips = 0
        # Get signs, filter out zeros
        signs = np.sign(valid_corrs)
        non_zero_signs = signs[signs != 0]
        # Calculate diff only if at least 2 non-zero signs exist
        if len(non_zero_signs) >= 2:
             sign_changes = np.diff(non_zero_signs) != 0
             sign_flips = np.sum(sign_changes)
        # else: sign_flips remains 0

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
        # Ensure file is written even if empty
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        output_filepath.write_text(f"No valid consistency data generated - Checked: {current_timestamp_str}\n", encoding='utf-8')
        return True

    # Create and sort DataFrame
    df_report = pd.DataFrame(report_data)
    # Sort primarily by Mean Abs Corr, then by Lags > Threshold, then Std Dev (lower is better)
    sort_col_lags = f'Lags > {abs_corr_threshold:.2f}'
    df_report.sort_values(
        by=['Mean Abs Corr', sort_col_lags, 'Std Dev Corr'],
        ascending=[False, False, True],
        na_position='last', # Place rows with NaN mean/std at the end
        inplace=True
    )

    # Format DataFrame for output
    max_p_len = 50
    df_report['Params'] = df_report['Params'].apply(lambda x: x if not isinstance(x, str) or len(x) <= max_p_len else x[:max_p_len-3] + '...')
    df_report['Config ID'] = df_report['Config ID'].astype(str)
    # Format numeric columns safely, handling potential NaNs/Infs from calculations
    for col in ['Mean Abs Corr', 'Std Dev Corr', 'Corr Slope']:
         df_report[col] = df_report[col].apply(lambda x: f"{float(x):.4f}" if pd.notna(x) and np.isfinite(x) else 'N/A')
    for col in [sort_col_lags, 'Sign Flips', 'Valid Points']:
        df_report[col] = df_report[col].apply(lambda x: f"{int(x)}" if pd.notna(x) and np.isfinite(x) else 'N/A')

    # Generate Output String
    output_string = f"Correlation Consistency Report (Across Lags 1-{max_lag})\n"
    output_string += f"Generated: {current_timestamp_str}\n"
    output_string += f"Sorted by Mean Abs Corr (desc), then Lags > {abs_corr_threshold:.2f} (desc), then Std Dev Corr (asc).\n"
    output_string += "=" * 130 + "\n"
    with pd.option_context('display.width', 1000, 'display.max_colwidth', 60):
        output_string += df_report.to_string(index=False, justify='left', na_rep='N/A')
    output_string += "\n" + "=" * 130

    try:
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        output_filepath.write_text(output_string, encoding='utf-8')
        logger.info(f"Consistency report saved to: {output_filepath}")
        # ** Removed print statement **
        return True
    except Exception as e:
        logger.error(f"Error saving consistency report: {e}", exc_info=True)
        return False

class LeaderboardManager:
    def __init__(self, db_path=None):
        """Initialize the leaderboard manager.
        
        Args:
            db_path: Optional path to the leaderboard database. If None, uses default from config.
        """
        if db_path is not None:
            # Temporarily override the config path for this instance
            self._original_db_path = config.LEADERBOARD_DB_PATH
            config.LEADERBOARD_DB_PATH = Path(db_path)
        else:
            self._original_db_path = None
            
        self._initialize_db()
    
    def _initialize_db(self) -> bool:
        """Initialize the database on class instantiation."""
        success = initialize_leaderboard_db()
        if self._original_db_path is not None:
            # Restore original path
            config.LEADERBOARD_DB_PATH = self._original_db_path
        return success
    
    def create_connection(self) -> Optional[sqlite3.Connection]:
        """Create a connection to the leaderboard database."""
        return _create_leaderboard_connection()
    
    def load_leaderboard(self) -> Dict[Tuple[int, str], Dict[str, Any]]:
        """Load current leaderboard data from the database."""
        return load_leaderboard()
    
    def check_and_update_single_lag(
        self,
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
        """Check and update a single lag in the leaderboard."""
        return check_and_update_single_lag(
            lag, correlation_value, indicator_name, params,
            config_id, symbol, timeframe, data_daterange, source_db_name
        )
    
    def update_leaderboard(
        self,
        current_run_correlations: Dict[int, List[Optional[float]]],
        indicator_configs: List[Dict[str, Any]],
        max_lag: int,
        symbol: str,
        timeframe: str,
        data_daterange: str,
        source_db_name: str,
        min_correlation_threshold: float = 0.0,
        max_updates_per_config: Optional[int] = None
    ) -> Dict[int, Dict[str, Any]]:
        """Update leaderboard with new correlation results.
        
        Args:
            current_run_correlations: Dictionary mapping config IDs to lists of correlation values for each lag
            indicator_configs: List of indicator configurations used in the current run
            max_lag: Maximum lag value used in the analysis
            symbol: Symbol being analyzed
            timeframe: Timeframe being analyzed
            data_daterange: Date range of the dataset
            source_db_name: Name of the source database
            min_correlation_threshold: Minimum correlation value to consider (default: 0.0)
            max_updates_per_config: Maximum number of updates to allow per config (default: None)
            
        Returns:
            Dictionary mapping config IDs to their best correlation results
        """
        return update_leaderboard(
            current_run_correlations, indicator_configs,
            max_lag, symbol, timeframe, data_daterange, source_db_name,
            min_correlation_threshold, max_updates_per_config
        )
    
    def export_leaderboard_to_text(self) -> bool:
        """Export the leaderboard to a text file."""
        return export_leaderboard_to_text()
    
    def find_best_predictor_for_lag(self, target_lag: int) -> Optional[Dict[str, Any]]:
        """Find the best predictor for a given lag."""
        return find_best_predictor_for_lag(target_lag)
    
    def generate_leading_indicator_report(self) -> bool:
        """Generate a report of leading indicators."""
        return generate_leading_indicator_report()
    
    def generate_consistency_report(
        self,
        correlations_by_config_id: Dict[int, List[Optional[float]]],
        indicator_configs_processed: List[Dict[str, Any]],
        max_lag: int,
        output_dir: Path,
        file_prefix: str,
        abs_corr_threshold: float = 0.15
    ) -> bool:
        """Generate a consistency report for indicators."""
        return generate_consistency_report(
            correlations_by_config_id, indicator_configs_processed,
            max_lag, output_dir, file_prefix, abs_corr_threshold
        )