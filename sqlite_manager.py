# sqlite_manager.py
import sqlite3
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import config
import os
from pathlib import Path
import hashlib # <<< ENSURED IMPORT IS PRESENT

logger = logging.getLogger(__name__)

def create_connection(db_path: str) -> Optional[sqlite3.Connection]:
    """Creates a database connection to the SQLite database specified by db_path."""
    conn = None
    try:
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True) # Ensure parent directory exists
        conn = sqlite3.connect(db_path, timeout=10) # Added timeout
        conn.execute("PRAGMA journal_mode=WAL;") # Use WAL mode for better concurrency (optional)
        conn.execute("PRAGMA foreign_keys = ON;")
        # conn.execute("PRAGMA defer_foreign_keys = ON;") # Try enabling this if ON isn't enough
        logger.debug(f"Successfully connected to database: {db_path}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database '{db_path}': {e}", exc_info=True)
        return None

def initialize_database(db_path: str) -> bool:
    """Initializes or recreates the database schema."""
    conn = create_connection(db_path)
    if conn is None: return False
    try:
        cursor = conn.cursor()
        conn.execute("BEGIN TRANSACTION;") # Use transaction for schema changes

        # --- Metadata Tables (Create if not exists) ---
        cursor.execute("""CREATE TABLE IF NOT EXISTS symbols (
                            id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT UNIQUE NOT NULL);""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS timeframes (
                            id INTEGER PRIMARY KEY AUTOINCREMENT, timeframe TEXT UNIQUE NOT NULL);""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS indicators (
                            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL);""")

        # --- Clean Recreate indicator_configs if it exists with wrong constraints ---
        # Check if the table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='indicator_configs';")
        table_exists = cursor.fetchone()

        # Attempt to drop cleanly if exists
        if table_exists:
             logger.warning("Found existing 'indicator_configs' table. Attempting clean recreate for schema consistency.")
             try:
                 # Drop dependent objects first if necessary (e.g., triggers, views) - none expected here
                 cursor.execute("DROP TABLE indicator_configs;") # Drop requires no IF EXISTS if checked before
                 logger.info("Dropped existing 'indicator_configs' table.")
             except sqlite3.Error as drop_err:
                 logger.error(f"Could not cleanly drop existing 'indicator_configs' table: {drop_err}. Manual check might be needed.")
                 conn.rollback()
                 return False


        # --- Create indicator_configs with correct schema ---
        cursor.execute("""
        CREATE TABLE indicator_configs (
            config_id INTEGER PRIMARY KEY AUTOINCREMENT, -- Explicit PK name
            indicator_id INTEGER NOT NULL,
            config_hash TEXT NOT NULL, -- Hash as TEXT
            config_json TEXT NOT NULL,
            FOREIGN KEY (indicator_id) REFERENCES indicators(id) ON DELETE CASCADE,
            UNIQUE (indicator_id, config_hash) -- Composite UNIQUE constraint
        );""")
        logger.info("Created 'indicator_configs' table with UNIQUE(indicator_id, config_hash).")
        # Index for foreign key lookup (optional, but good practice)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicator_configs_indicator_id ON indicator_configs(indicator_id);")


        # --- Data Table (Create if not exists) ---
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS historical_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT, symbol_id INTEGER NOT NULL, timeframe_id INTEGER NOT NULL,
            open_time INTEGER UNIQUE NOT NULL, open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL,
            close_time INTEGER NOT NULL, quote_asset_volume REAL, number_of_trades INTEGER,
            taker_buy_base_asset_volume REAL, taker_buy_quote_asset_volume REAL,
            FOREIGN KEY (symbol_id) REFERENCES symbols(id) ON DELETE CASCADE,
            FOREIGN KEY (timeframe_id) REFERENCES timeframes(id) ON DELETE CASCADE
        );""")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_historical_data_symbol_timeframe_opentime ON historical_data(symbol_id, timeframe_id, open_time);")

        # --- Correlation Table (Create if not exists) ---
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS correlations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, symbol_id INTEGER NOT NULL, timeframe_id INTEGER NOT NULL,
            -- Reference the explicitly named PK of indicator_configs
            indicator_config_id INTEGER NOT NULL REFERENCES indicator_configs(config_id) ON DELETE CASCADE,
            lag INTEGER NOT NULL, correlation_value REAL,
            FOREIGN KEY (symbol_id) REFERENCES symbols(id) ON DELETE CASCADE,
            FOREIGN KEY (timeframe_id) REFERENCES timeframes(id) ON DELETE CASCADE,
            UNIQUE(symbol_id, timeframe_id, indicator_config_id, lag)
        );""")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_correlations_main ON correlations (symbol_id, timeframe_id, indicator_config_id, lag);")

        conn.commit() # Commit transaction
        logger.info(f"Database schema operations completed successfully: {db_path}")
        return True
    except sqlite3.Error as e:
        logger.error(f"Error operating on database schema '{db_path}': {e}", exc_info=True)
        try:
            conn.rollback()
        except sqlite3.Error as rb_err:
             logger.error(f"Error during rollback: {rb_err}")
        return False
    finally:
        if conn:
            conn.close()

def _get_or_create_id(conn: sqlite3.Connection, table: str, column: str, value: str) -> int:
    """Gets the ID for a value in a simple lookup table, creating it if necessary."""
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT id FROM {table} WHERE {column} = ?", (value,))
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            cursor.execute(f"INSERT INTO {table} ({column}) VALUES (?)", (value,))
            conn.commit() # Commit immediately after successful insert for lookup tables
            new_id = cursor.lastrowid
            logger.info(f"Inserted '{value}' into '{table}', ID: {new_id}")
            return new_id
    except sqlite3.IntegrityError: # Catch specifically the unique constraint violation
        conn.rollback() # Rollback the failed insert
        logger.warning(f"IntegrityError inserting '{value}' into '{table}'. Re-querying assuming it exists.")
        cursor.execute(f"SELECT id FROM {table} WHERE {column} = ?", (value,))
        result = cursor.fetchone()
        if result:
            return result[0]
        else: # If it's still not there after an IntegrityError, something is wrong
            logger.critical(f"CRITICAL: Could not retrieve ID for '{value}' in '{table}' even after IntegrityError.")
            raise ValueError(f"Could not get or create ID for {value} in {table}")
    except sqlite3.Error as e: # Catch other potential DB errors
        conn.rollback()
        logger.error(f"Database error in _get_or_create_id for '{value}' in '{table}': {e}", exc_info=True)
        raise # Re-raise the exception

def get_or_create_indicator_config_id(conn: sqlite3.Connection, indicator_name: str, params: Dict[str, Any]) -> int:
    """Gets the ID for a specific indicator configuration, creating it if necessary."""
    try:
        indicator_id = _get_or_create_id(conn, 'indicators', 'name', indicator_name)
        config_str = json.dumps(params, sort_keys=True, separators=(',', ':'))
        # Using hashlib SHA256 hash instead of default hash() for better stability/distribution
        config_hash = hashlib.sha256(config_str.encode('utf-8')).hexdigest() # Now hashlib is defined

        cursor = conn.cursor()
        # Query using the composite key
        cursor.execute("SELECT config_id, config_json FROM indicator_configs WHERE indicator_id = ? AND config_hash = ?", (indicator_id, config_hash))
        result = cursor.fetchone()

        if result:
            config_id_found, existing_json = result
            # Verify JSON match for robustness against hash collisions
            if existing_json == config_str:
                logger.debug(f"Found existing config for {indicator_name} / {params}, ID: {config_id_found}")
                return config_id_found
            else:
                # Hash collision
                logger.error(f"HASH COLLISION DETECTED for indicator ID {indicator_id}, hash {config_hash}. DB JSON: {existing_json}, Expected: {config_str}")
                raise ValueError(f"Hash collision detected for {indicator_name} / {params}")
        else:
            # Config not found, attempt insert
            try:
                cursor.execute("INSERT INTO indicator_configs (indicator_id, config_hash, config_json) VALUES (?, ?, ?)",
                               (indicator_id, config_hash, config_str))
                # Commit immediately after successful insert
                conn.commit()
                new_config_id = cursor.lastrowid
                logger.info(f"Inserted new config for '{indicator_name}' (Params: {params}), ID: {new_config_id}")
                return new_config_id
            except sqlite3.IntegrityError:
                # Race condition or potential issue with UNIQUE constraint implementation detail
                conn.rollback() # Rollback failed insert
                logger.warning(f"IntegrityError inserting config for indicator ID {indicator_id}, hash {config_hash}. Re-querying.")
                # Re-query using the composite key
                cursor.execute("SELECT config_id, config_json FROM indicator_configs WHERE indicator_id = ? AND config_hash = ?", (indicator_id, config_hash))
                result = cursor.fetchone()
                if result:
                    config_id_found, existing_json = result
                    if existing_json == config_str: # Verify json match again
                        logger.debug(f"Found config ID {config_id_found} for {indicator_name} / {params} after IntegrityError.")
                        return config_id_found
                    else: # Collision + Race Condition - very unlikely
                        logger.error(f"CRITICAL: IntegrityError AND JSON mismatch after re-query for indicator ID {indicator_id}, hash {config_hash}.")
                        raise ValueError(f"Data inconsistency detected for {indicator_name} / {params} after insert attempt.")
                else: # Failed insert but not found on re-query
                    logger.critical(f"CRITICAL: Could not find config for indicator ID {indicator_id}, hash {config_hash} even after IntegrityError.")
                    raise ValueError(f"Could not resolve config ID for {indicator_name} / {params} after database error.")
    except sqlite3.Error as e:
         logger.error(f"Database error getting/creating config ID for '{indicator_name}' / {params}: {e}", exc_info=True)
         raise # Re-raise DB errors
    except Exception as e:
         logger.error(f"Unexpected error in get_or_create_indicator_config_id for '{indicator_name}' / {params}: {e}", exc_info=True)
         raise


# --- Batch Insert Function ---
def batch_insert_correlations(
    conn: sqlite3.Connection,
    data_to_insert: List[Tuple[int, int, int, int, Optional[float]]]
) -> bool:
    """
    Inserts or replaces multiple correlation values in a single transaction.

    Args:
        conn: Active SQLite connection.
        data_to_insert: A list of tuples, where each tuple is
                        (symbol_id, timeframe_id, indicator_config_id, lag, correlation_value)
    Returns:
        True if successful or partially successful (some rows inserted), False on major error.
    """
    if not data_to_insert:
        logger.info("No correlation data provided for batch insert.")
        return True

    query = """
    INSERT OR REPLACE INTO correlations (symbol_id, timeframe_id, indicator_config_id, lag, correlation_value)
    VALUES (?, ?, ?, ?, ?);
    """
    cursor = conn.cursor()
    inserted_count = 0
    error_count = 0

    try:
        # Use executemany for potentially faster inserts within a transaction
        cursor.execute("BEGIN TRANSACTION;")
        # Prepare data (handle potential NaNs in correlation value)
        prepared_data = []
        for row in data_to_insert:
            s_id, t_id, cfg_id, lag, corr_val = row
            # Ensure corr_val is float or None before isnan check
            if isinstance(corr_val, (int, float)):
                 db_value = None if np.isnan(corr_val) else float(corr_val)
            else:
                 db_value = None # Treat non-numeric as None/NULL
            prepared_data.append((s_id, t_id, cfg_id, lag, db_value))

        cursor.executemany(query, prepared_data)
        inserted_count = cursor.rowcount # Note: executemany rowcount might be -1 or unreliable on some drivers/versions
        conn.commit()
        # If rowcount unreliable, use len(prepared_data) as estimate
        actual_inserted = inserted_count if inserted_count != -1 else len(prepared_data)
        logger.info(f"Batch inserted/replaced correlations for {actual_inserted} records.")
        return True
    except sqlite3.Error as e:
        logger.error(f"Database error during batch correlation insert: {e}", exc_info=True)
        try:
            conn.rollback()
            logger.warning("Rolled back batch correlation insert due to error.")
        except sqlite3.Error as rb_err:
             logger.error(f"Error during rollback: {rb_err}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during batch correlation preparation/insert: {e}", exc_info=True)
        try: conn.rollback()
        except: pass
        return False

# --- fetch_correlations remains unchanged ---
def fetch_correlations(conn: sqlite3.Connection, symbol_id: int, timeframe_id: int, config_ids: List[int]) -> Dict[int, List[Optional[float]]]:
    """Fetches correlations for given config IDs, returns dict {config_id: [corr_lag1, ...]}."""
    if not config_ids: return {}
    logger.debug(f"Fetching correlations for {len(config_ids)} config IDs...")
    placeholders = ','.join('?' for _ in config_ids)
    # SELECT indicator_config_id (the FK column)
    query = f"""
    SELECT indicator_config_id, lag, correlation_value
    FROM correlations
    WHERE symbol_id = ? AND timeframe_id = ? AND indicator_config_id IN ({placeholders})
    ORDER BY indicator_config_id, lag;
    """
    try:
        cursor = conn.cursor()
        params = [symbol_id, timeframe_id] + config_ids
        cursor.execute(query, params)
        rows = cursor.fetchall()

        if not rows:
            logger.warning(f"No correlation data found for requested config IDs (SymbolID: {symbol_id}, TFID: {timeframe_id}).")
            # Return dict with empty lists for all requested IDs
            return {cfg_id: [] for cfg_id in config_ids}

        max_lag_found = max(row[1] for row in rows) if rows else 0
        if max_lag_found <= 0: # Handle case where only invalid lags might exist (shouldn't happen)
             logger.warning("Max lag found in DB is zero or less. Cannot structure results.")
             return {cfg_id: [] for cfg_id in config_ids}
        logger.debug(f"Max lag found in fetched correlations: {max_lag_found}")

        # Initialize dict with None lists only for IDs that HAVE data initially
        results_dict: Dict[int, List[Optional[float]]] = {}
        current_row_cfg_id = -1
        current_list: List[Optional[float]] = []

        for cfg_id_from_db, lag, value in rows: # cfg_id_from_db is the value from indicator_config_id column
            if cfg_id_from_db != current_row_cfg_id:
                 if current_row_cfg_id != -1: # Store previous list if valid
                      # Pad previous list to max_lag_found if shorter
                      if len(current_list) < max_lag_found:
                           current_list.extend([None] * (max_lag_found - len(current_list)))
                      results_dict[current_row_cfg_id] = current_list

                 # Start new list for the new config_id
                 current_row_cfg_id = cfg_id_from_db
                 # Pre-allocate list with Nones up to max_lag_found
                 current_list = [None] * max_lag_found

            # Populate the list at the correct index (lag-1)
            # Ensure lag is within the expected range (1 to max_lag_found)
            if 1 <= lag <= max_lag_found:
                 # Convert DB value (which could be None) to float or keep None
                 current_list[lag - 1] = float(value) if value is not None else None


        # Store the last processed list after loop finishes
        if current_row_cfg_id != -1:
             if len(current_list) < max_lag_found:
                 current_list.extend([None] * (max_lag_found - len(current_list)))
             results_dict[current_row_cfg_id] = current_list

        # Ensure all *originally requested* config IDs have an entry in the final dict
        # If an ID had no data, its entry will be a list of Nones of length max_lag_found
        for cfg_id_req in config_ids:
             if cfg_id_req not in results_dict:
                 results_dict[cfg_id_req] = [None] * max_lag_found

        logger.info(f"Fetched and structured correlation data for {len(results_dict)} configurations.")
        return results_dict

    except sqlite3.Error as e:
        logger.error(f"Error fetching correlations: {e}", exc_info=True)
        return {} # Return empty dict on DB error
    except Exception as e:
         logger.error(f"Unexpected error processing fetched correlation data: {e}", exc_info=True)
         return {} # Return empty dict on other errors