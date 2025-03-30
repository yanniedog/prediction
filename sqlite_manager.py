# sqlite_manager.py
import sqlite3
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import config
import os
from pathlib import Path
import hashlib
import math # Added for ceiling division in batching

logger = logging.getLogger(__name__)

# Define a safe limit for SQL variables (SQLite default is often 999 or 32766, be conservative)
SQLITE_MAX_VARIABLE_NUMBER = 900

def create_connection(db_path: str) -> Optional[sqlite3.Connection]:
    """Creates a database connection to the SQLite database."""
    conn = None
    try:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # Increased timeout, added isolation level for better control with BEGIN IMMEDIATE
        conn = sqlite3.connect(db_path, timeout=15, isolation_level=None) # Autocommit mode OFF by default
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 10000;") # Wait 10s if DB is locked
        logger.debug(f"Connected to database: {db_path}")
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
        # Use DEFERRED transaction for initialization (less locking)
        conn.execute("BEGIN DEFERRED;")

        # Metadata Tables
        cursor.execute("CREATE TABLE IF NOT EXISTS symbols (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT UNIQUE NOT NULL);")
        cursor.execute("CREATE TABLE IF NOT EXISTS timeframes (id INTEGER PRIMARY KEY AUTOINCREMENT, timeframe TEXT UNIQUE NOT NULL);")
        cursor.execute("CREATE TABLE IF NOT EXISTS indicators (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL);")

        # Check indicator_configs schema (no changes from previous version needed here)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='indicator_configs';")
        configs_exists = cursor.fetchone()
        recreated_tables = False
        if configs_exists:
            try:
                cursor.execute("SELECT sql FROM sqlite_master WHERE name='indicator_configs';")
                schema_sql = cursor.fetchone()[0]
                if 'id INTEGER PRIMARY KEY' in schema_sql.lower() and 'config_id' not in schema_sql.lower():
                    logger.warning("Old schema detected. Recreating 'indicator_configs' and 'correlations' tables.")
                    cursor.execute("PRAGMA foreign_keys=OFF;")
                    cursor.execute("DROP TABLE IF EXISTS correlations;")
                    cursor.execute("DROP TABLE IF EXISTS indicator_configs;")
                    recreated_tables = True
            except Exception as schema_check_err:
                 logger.warning(f"Could not check existing schema: {schema_check_err}. Assuming potential need for recreate.")
                 try:
                    cursor.execute("PRAGMA foreign_keys=OFF;")
                    cursor.execute("DROP TABLE IF EXISTS correlations;")
                    cursor.execute("DROP TABLE IF EXISTS indicator_configs;")
                    recreated_tables = True
                 except sqlite3.Error as drop_err:
                    logger.error(f"Could not drop existing tables: {drop_err}. Manual check needed."); conn.rollback(); cursor.execute("PRAGMA foreign_keys=ON;"); return False

        # Create indicator_configs if needed
        if recreated_tables or not configs_exists:
            cursor.execute("""
            CREATE TABLE indicator_configs (
                config_id INTEGER PRIMARY KEY AUTOINCREMENT, indicator_id INTEGER NOT NULL,
                config_hash TEXT NOT NULL, config_json TEXT NOT NULL,
                FOREIGN KEY (indicator_id) REFERENCES indicators(id) ON DELETE CASCADE,
                UNIQUE (indicator_id, config_hash) );""")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicator_configs_indicator_id ON indicator_configs(indicator_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicator_configs_hash ON indicator_configs(config_hash);") # Index hash too

        # Data Table (schema unchanged)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS historical_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT, symbol_id INTEGER NOT NULL, timeframe_id INTEGER NOT NULL,
            open_time INTEGER NOT NULL, open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL,
            close_time INTEGER NOT NULL, quote_asset_volume REAL, number_of_trades INTEGER,
            taker_buy_base_asset_volume REAL, taker_buy_quote_asset_volume REAL,
            FOREIGN KEY (symbol_id) REFERENCES symbols(id) ON DELETE CASCADE,
            FOREIGN KEY (timeframe_id) REFERENCES timeframes(id) ON DELETE CASCADE,
            UNIQUE(symbol_id, timeframe_id, open_time) );""")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_historical_data_symbol_timeframe_opentime ON historical_data(symbol_id, timeframe_id, open_time);")

        # Correlation Table - Create if needed (schema unchanged)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS correlations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, symbol_id INTEGER NOT NULL, timeframe_id INTEGER NOT NULL,
            indicator_config_id INTEGER NOT NULL REFERENCES indicator_configs(config_id) ON DELETE CASCADE,
            lag INTEGER NOT NULL, correlation_value REAL,
            FOREIGN KEY (symbol_id) REFERENCES symbols(id) ON DELETE CASCADE,
            FOREIGN KEY (timeframe_id) REFERENCES timeframes(id) ON DELETE CASCADE,
            UNIQUE(symbol_id, timeframe_id, indicator_config_id, lag) );""")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_correlations_main ON correlations (symbol_id, timeframe_id, indicator_config_id, lag);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_correlations_config_lag ON correlations (indicator_config_id, lag);") # Add helpful index

        conn.execute("PRAGMA foreign_keys=ON;") # Ensure FKs are ON before commit
        conn.commit()
        logger.info(f"Database schema operations completed: {db_path}")
        return True
    except sqlite3.Error as e:
        logger.error(f"Error operating on DB schema '{db_path}': {e}", exc_info=True)
        try: conn.rollback()
        except Exception as rb_err: logger.error(f"Rollback failed during schema init: {rb_err}")
        return False
    finally:
        if conn: conn.close()

def _get_or_create_id(conn: sqlite3.Connection, table: str, column: str, value: Any) -> int:
    """Gets ID for value in lookup table, creating if necessary (case-insensitive lookup). Assumes caller handles transactions."""
    cursor = conn.cursor()
    str_value = str(value) if value is not None else None
    if str_value is None: raise ValueError(f"Cannot get/create ID for None in {table}.{column}")
    lookup_value = str_value.lower()

    try:
        cursor.execute(f"SELECT id FROM {table} WHERE LOWER({column}) = ?", (lookup_value,))
        result = cursor.fetchone()
        if result: return result[0]
        else:
            cursor.execute(f"INSERT INTO {table} ({column}) VALUES (?)", (str_value,))
            new_id = cursor.lastrowid
            if new_id is None: # Re-query if insert didn't return ID
                 logger.warning(f"Insert into '{table}' for '{str_value}' no lastrowid. Re-querying.")
                 cursor.execute(f"SELECT id FROM {table} WHERE LOWER({column}) = ?", (lookup_value,))
                 result = cursor.fetchone()
                 if result: return result[0]
                 else: raise ValueError(f"Cannot retrieve ID for '{str_value}' in '{table}' after insert.")
            logger.debug(f"Inserted '{str_value}' into '{table}', ID: {new_id}") # Changed to DEBUG for less noise
            return new_id
    except sqlite3.IntegrityError: # Handle concurrent insert
        logger.warning(f"IntegrityError inserting '{str_value}' into '{table}'. Re-querying.")
        cursor.execute(f"SELECT id FROM {table} WHERE LOWER({column}) = ?", (lookup_value,))
        result = cursor.fetchone()
        if result: return result[0]
        else: logger.critical(f"CRITICAL: Cannot retrieve ID for '{str_value}' in '{table}' after IntegrityError."); raise
    except sqlite3.Error as e: logger.error(f"DB error get/create ID for '{str_value}' in '{table}': {e}", exc_info=True); raise

def get_or_create_indicator_config_id(conn: sqlite3.Connection, indicator_name: str, params: Dict[str, Any]) -> int:
    """Gets the ID for an indicator config, creating if necessary. Manages its own transaction."""
    cursor = conn.cursor()
    try:
        # Use IMMEDIATE transaction for potentially faster write lock acquisition
        cursor.execute("BEGIN IMMEDIATE;")
        indicator_id = _get_or_create_id(conn, 'indicators', 'name', indicator_name)
        config_str = json.dumps(params, sort_keys=True, separators=(',', ':'))
        config_hash = hashlib.sha256(config_str.encode('utf-8')).hexdigest()

        cursor.execute("SELECT config_id, config_json FROM indicator_configs WHERE indicator_id = ? AND config_hash = ?", (indicator_id, config_hash))
        result = cursor.fetchone()

        if result:
            config_id_found, existing_json = result
            if existing_json == config_str:
                logger.debug(f"Found existing config ID {config_id_found} for {indicator_name} / {params}")
                conn.commit(); return config_id_found
            else: logger.error(f"HASH COLLISION DETECTED: ID {indicator_id}, hash {config_hash}. DB JSON: {existing_json}, Expected: {config_str}"); conn.rollback(); raise ValueError("Hash collision")
        else: # Insert new config
            try:
                cursor.execute("INSERT INTO indicator_configs (indicator_id, config_hash, config_json) VALUES (?, ?, ?)", (indicator_id, config_hash, config_str))
                new_config_id = cursor.lastrowid
                if new_config_id is None: raise sqlite3.Error("INSERT successful but lastrowid is None.")
                conn.commit()
                logger.debug(f"Inserted new config for '{indicator_name}' (ID: {new_config_id}), Params: {params}") # Changed to DEBUG
                return new_config_id
            except sqlite3.IntegrityError: # Handle race condition
                conn.rollback(); logger.warning(f"IntegrityError inserting config for indicator ID {indicator_id}, hash {config_hash}. Re-querying.")
                cursor.execute("SELECT config_id, config_json FROM indicator_configs WHERE indicator_id = ? AND config_hash = ?", (indicator_id, config_hash))
                result = cursor.fetchone()
                if result:
                    config_id_found, existing_json = result
                    if existing_json == config_str: logger.debug(f"Found config ID {config_id_found} after IntegrityError."); return config_id_found
                    else: logger.error(f"CRITICAL: IntegrityError AND JSON mismatch after re-query: ID {indicator_id}, hash {config_hash}."); raise ValueError("Data inconsistency")
                else: logger.critical(f"CRITICAL: Cannot find config ID {indicator_id}, hash {config_hash} after IntegrityError."); raise ValueError("Cannot resolve config ID")
    except Exception as e:
         logger.error(f"Error get/create config ID for '{indicator_name}' / {params}: {e}", exc_info=True)
         try: conn.rollback()
         except: pass
         raise

# --- Batch Insert Function ---
def batch_insert_correlations(conn: sqlite3.Connection, data_to_insert: List[Tuple[int, int, int, int, Optional[float]]]) -> bool:
    """Inserts or replaces multiple correlation values in a single transaction."""
    if not data_to_insert: logger.info("No correlation data for batch insert."); return True
    query = "INSERT OR REPLACE INTO correlations (symbol_id, timeframe_id, indicator_config_id, lag, correlation_value) VALUES (?, ?, ?, ?, ?);"
    cursor = conn.cursor()
    try:
        prepared_data = []
        for s_id, t_id, cfg_id, lag, corr_val in data_to_insert:
            db_value = float(corr_val) if pd.notna(corr_val) and isinstance(corr_val, (int, float)) else None
            prepared_data.append((s_id, t_id, cfg_id, lag, db_value))

        cursor.execute("BEGIN IMMEDIATE;") # Use IMMEDIATE for faster write lock
        cursor.executemany(query, prepared_data)
        conn.commit()
        logger.info(f"Batch inserted/replaced correlations for {len(prepared_data)} records.")
        return True
    except sqlite3.Error as e:
        logger.error(f"DB error during batch corr insert: {e}", exc_info=True)
        try: conn.rollback(); logger.warning("Rolled back batch corr insert.")
        except: pass
        return False
    except Exception as e:
        logger.error(f"Unexpected error during batch corr prep/insert: {e}", exc_info=True)
        try: conn.rollback()
        except: pass
        return False

# --- Fetch Correlations Function (BATCHED) ---
def fetch_correlations(conn: sqlite3.Connection, symbol_id: int, timeframe_id: int, config_ids: List[int]) -> Dict[int, List[Optional[float]]]:
    """Fetches correlations in batches, returns dict {config_id: [corr_lag1, ..., corr_lagN]}."""
    if not config_ids: return {}
    logger.debug(f"Fetching correlations for {len(config_ids)} IDs (Sym: {symbol_id}, TF: {timeframe_id})...")

    # Determine batch size (ensure it's less than the max variable limit, accounting for symbol_id and timeframe_id)
    batch_size = SQLITE_MAX_VARIABLE_NUMBER - 2 # Subtract 2 for the fixed params
    if batch_size <= 0:
        logger.error("SQLITE_MAX_VARIABLE_NUMBER too low to fetch correlations.")
        return {}

    all_rows = []
    num_batches = math.ceil(len(config_ids) / batch_size)
    logger.info(f"Fetching {len(config_ids)} correlations in {num_batches} batches (size {batch_size}).")

    cursor = conn.cursor()
    try:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_ids = config_ids[start_idx:end_idx]
            if not batch_ids: continue

            placeholders = ','.join('?' for _ in batch_ids)
            query = f"""
                SELECT indicator_config_id, lag, correlation_value
                FROM correlations
                WHERE symbol_id = ? AND timeframe_id = ? AND indicator_config_id IN ({placeholders})
                ORDER BY indicator_config_id, lag;
            """
            params = [symbol_id, timeframe_id] + batch_ids
            logger.debug(f"Executing fetch batch {i+1}/{num_batches} ({len(batch_ids)} IDs)")
            cursor.execute(query, params)
            batch_rows = cursor.fetchall()
            all_rows.extend(batch_rows)
            logger.debug(f" Batch {i+1} fetched {len(batch_rows)} rows.")

        if not all_rows:
            logger.warning(f"No correlation data found for requested IDs (Sym: {symbol_id}, TF: {timeframe_id}) after fetching batches.")
            return {cfg_id: [] for cfg_id in config_ids} # Return empty lists for all requested

        # Determine max lag across all fetched rows
        max_lag_found = max((row[1] for row in all_rows if isinstance(row[1], int)), default=0)
        if max_lag_found <= 0:
            logger.warning("Max lag found <= 0 after fetching batches.")
            return {cfg_id: [] for cfg_id in config_ids} # Still return empty lists

        logger.debug(f"Max lag found across all batches: {max_lag_found}")

        # Initialize results dictionary for ALL originally requested IDs
        results_dict: Dict[int, List[Optional[float]]] = {cfg_id: [None] * max_lag_found for cfg_id in config_ids}

        # Populate the results dictionary
        processed_rows = 0
        for cfg_id_db, lag, value in all_rows:
            if isinstance(lag, int) and 1 <= lag <= max_lag_found:
                 if cfg_id_db in results_dict:
                      # Only assign if the index is valid (should always be if max_lag_found is correct)
                      if lag-1 < len(results_dict[cfg_id_db]):
                          results_dict[cfg_id_db][lag - 1] = float(value) if value is not None else None
                          processed_rows += 1
                      else:
                           logger.error(f"Internal Error: Lag {lag} out of bounds for CfgID {cfg_id_db} (maxlag={max_lag_found}).")
                 # else: # This case should ideally not happen if config_ids requested match what's fetched
                 #    logger.warning(f"Fetched corr for config_id {cfg_id_db} not in original request list? Ignoring.")
            else:
                 # Log invalid lags more informatively
                 logger.warning(f"Invalid lag {lag} (type: {type(lag)}, value: {value}) encountered for config_id {cfg_id_db}. Ignoring.")

        logger.info(f"Fetched and processed {processed_rows} correlation points for {len(config_ids)} configs up to lag {max_lag_found}.")
        return results_dict
    except sqlite3.Error as e:
        logger.error(f"SQLite error during batched correlation fetch: {e}", exc_info=True)
        return {} # Return empty on error
    except Exception as e:
        logger.error(f"Error fetching/processing correlations: {e}", exc_info=True)
        return {}


# --- Functions for Custom Mode ---

def get_max_lag_for_pair(conn: sqlite3.Connection, symbol_id: int, timeframe_id: int) -> Optional[int]:
    """Gets the maximum lag value stored in the correlations table for a symbol/timeframe pair."""
    query = "SELECT MAX(lag) FROM correlations WHERE symbol_id = ? AND timeframe_id = ?;"
    cursor = conn.cursor()
    try:
        cursor.execute(query, (symbol_id, timeframe_id))
        result = cursor.fetchone()
        if result and result[0] is not None:
            max_lag = int(result[0])
            logger.info(f"Determined max lag from existing DB data: {max_lag} for SymbolID {symbol_id}, TFID {timeframe_id}")
            return max_lag
        else:
            logger.warning(f"No correlation data found to determine max lag for SymbolID {symbol_id}, TFID {timeframe_id}.")
            return None
    except Exception as e:
        logger.error(f"Error getting max lag for pair: {e}", exc_info=True)
        return None

def get_distinct_config_ids_for_pair(conn: sqlite3.Connection, symbol_id: int, timeframe_id: int) -> List[int]:
    """Gets a list of distinct indicator_config_ids that have correlation data for a symbol/timeframe pair."""
    query = "SELECT DISTINCT indicator_config_id FROM correlations WHERE symbol_id = ? AND timeframe_id = ?;"
    cursor = conn.cursor()
    config_ids = []
    try:
        cursor.execute(query, (symbol_id, timeframe_id))
        rows = cursor.fetchall()
        config_ids = [row[0] for row in rows if row[0] is not None]
        logger.info(f"Found {len(config_ids)} distinct config IDs with correlation data for SymbolID {symbol_id}, TFID {timeframe_id}.")
        return config_ids
    except Exception as e:
        logger.error(f"Error getting distinct config IDs for pair: {e}", exc_info=True)
        return []

def get_indicator_configs_by_ids(conn: sqlite3.Connection, config_ids: List[int]) -> List[Dict[str, Any]]:
    """Fetches indicator configuration details (name, params) for a list of config_ids in batches."""
    if not config_ids:
        return []

    batch_size = SQLITE_MAX_VARIABLE_NUMBER # Use full batch size here
    num_batches = math.ceil(len(config_ids) / batch_size)
    logger.info(f"Fetching details for {len(config_ids)} configs in {num_batches} batches (size {batch_size}).")

    all_rows = []
    cursor = conn.cursor()
    try:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_ids = config_ids[start_idx:end_idx]
            if not batch_ids: continue

            placeholders = ','.join('?' for _ in batch_ids)
            query = f"""
                SELECT
                    ic.config_id,
                    i.name AS indicator_name,
                    ic.config_json
                FROM indicator_configs ic
                JOIN indicators i ON ic.indicator_id = i.id
                WHERE ic.config_id IN ({placeholders});
            """
            logger.debug(f"Executing fetch batch {i+1}/{num_batches} for config details ({len(batch_ids)} IDs)")
            cursor.execute(query, batch_ids)
            batch_rows = cursor.fetchall()
            all_rows.extend(batch_rows)
            logger.debug(f" Batch {i+1} fetched {len(batch_rows)} config detail rows.")

        configs_processed = []
        processed_count = 0
        for cfg_id, name, json_str in all_rows:
            try:
                params = json.loads(json_str)
                configs_processed.append({
                    'config_id': cfg_id,
                    'indicator_name': name,
                    'params': params
                })
                processed_count += 1
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON parameters for config_id {cfg_id} ('{name}'): {json_str}")
            except Exception as e:
                 logger.error(f"Error processing config details for config_id {cfg_id} ('{name}'): {e}")

        logger.info(f"Retrieved and processed details for {processed_count}/{len(config_ids)} requested config IDs.")
        return configs_processed

    except sqlite3.Error as e:
        logger.error(f"SQLite error during batched config detail fetch: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Error fetching indicator config details by IDs: {e}", exc_info=True)
        return []