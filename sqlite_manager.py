# sqlite_manager.py
import sqlite3
import json
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import pandas as pd
import config
import os
from pathlib import Path
import hashlib
import math # Added for ceiling division in batching
import utils
import shutil
import time

logger = logging.getLogger(__name__)

# Define a safe limit for SQL variables (SQLite default is often 999 or 32766, be conservative)
# Check your specific SQLite version if needed, but 900 is generally safe.
SQLITE_MAX_VARIABLE_NUMBER = config.DEFAULTS.get("sqlite_max_variable_number", 900)

def create_connection(db_path: Union[str, Path], timeout: float = 30.0) -> Optional[sqlite3.Connection]:
    """Create a database connection with retry logic."""
    db_path = Path(db_path)
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            # Create parent directories if they don't exist
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect with timeout and other optimizations
            conn = sqlite3.connect(str(db_path), timeout=timeout)
            
            # Set pragmas for better performance and reliability
            conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
            conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
            conn.execute("PRAGMA foreign_keys=ON")  # Enforce foreign key constraints
            conn.execute("PRAGMA busy_timeout=5000")  # 5 second busy timeout
            
            return conn
            
        except sqlite3.Error as e:
            if attempt < max_retries - 1:
                logger.warning(f"Database connection attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
            else:
                logger.error(f"Failed to connect to database after {max_retries} attempts: {e}")
                raise
                
    return None

def initialize_database(db_path: Union[str, Path], symbol: str, timeframe: str) -> bool:
    """Initialize database with required tables and constraints."""
    conn = None
    try:
        db_path = Path(db_path)
        
        # Create parent directories if they don't exist
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if database exists and is corrupted
        if db_path.exists():
            try:
                test_conn = create_connection(db_path)
                if test_conn:
                    # Check if tables exist and have required columns
                    cursor = test_conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='symbols'")
                    if not cursor.fetchone():
                        logger.warning(f"Database {db_path} exists but is missing required tables. Recreating...")
                        test_conn.close()
                        db_path.unlink()
                    else:
                        # Verify symbol and timeframe exist
                        cursor.execute("SELECT id FROM symbols WHERE symbol = ?", (symbol,))
                        symbol_id = cursor.fetchone()
                        cursor.execute("SELECT id FROM timeframes WHERE timeframe = ?", (timeframe,))
                        timeframe_id = cursor.fetchone()
                        if not symbol_id or not timeframe_id:
                            logger.warning(f"Database {db_path} exists but missing required symbol/timeframe. Recreating...")
                            test_conn.close()
                            db_path.unlink()
                        else:
                            test_conn.close()
                            return True
            except sqlite3.DatabaseError:
                logger.warning(f"Database {db_path} appears to be corrupted. Removing...")
                db_path.unlink()
                
        # Create new connection
        conn = create_connection(db_path)
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        # Create tables with proper constraints
        cursor.executescript("""
            -- Symbols table
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Timeframes table
            CREATE TABLE IF NOT EXISTS timeframes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timeframe TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Indicators table
            CREATE TABLE IF NOT EXISTS indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                type TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Indicator configurations table
            CREATE TABLE IF NOT EXISTS indicator_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                indicator_id INTEGER NOT NULL,
                params JSON NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (indicator_id) REFERENCES indicators(id)
                    ON DELETE CASCADE
            );
            
            -- Historical data table
            CREATE TABLE IF NOT EXISTS historical_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol_id INTEGER NOT NULL,
                timeframe_id INTEGER NOT NULL,
                open_time INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                close_time INTEGER NOT NULL,
                quote_asset_volume REAL,
                number_of_trades INTEGER,
                taker_buy_base_asset_volume REAL,
                taker_buy_quote_asset_volume REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (symbol_id) REFERENCES symbols(id)
                    ON DELETE CASCADE,
                FOREIGN KEY (timeframe_id) REFERENCES timeframes(id)
                    ON DELETE CASCADE,
                UNIQUE(symbol_id, timeframe_id, open_time)
            );
            
            -- Correlations table
            CREATE TABLE IF NOT EXISTS correlations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol_id INTEGER NOT NULL,
                timeframe_id INTEGER NOT NULL,
                indicator_id INTEGER NOT NULL,
                lag INTEGER NOT NULL,
                correlation REAL NOT NULL,
                p_value REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (symbol_id) REFERENCES symbols(id)
                    ON DELETE CASCADE,
                FOREIGN KEY (timeframe_id) REFERENCES timeframes(id)
                    ON DELETE CASCADE,
                FOREIGN KEY (indicator_id) REFERENCES indicators(id)
                    ON DELETE CASCADE,
                UNIQUE(symbol_id, timeframe_id, indicator_id, lag)
            );
            
            -- Leaderboard table
            CREATE TABLE IF NOT EXISTS leaderboard (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol_id INTEGER NOT NULL,
                timeframe_id INTEGER NOT NULL,
                indicator_id INTEGER NOT NULL,
                lag INTEGER NOT NULL,
                correlation_type TEXT NOT NULL CHECK(correlation_type IN ('positive', 'negative')),
                correlation_value REAL NOT NULL,
                config_json TEXT NOT NULL,
                dataset_daterange TEXT NOT NULL,
                calculation_timestamp TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (symbol_id) REFERENCES symbols(id)
                    ON DELETE CASCADE,
                FOREIGN KEY (timeframe_id) REFERENCES timeframes(id)
                    ON DELETE CASCADE,
                FOREIGN KEY (indicator_id) REFERENCES indicators(id)
                    ON DELETE CASCADE,
                UNIQUE(symbol_id, timeframe_id, indicator_id, lag, correlation_type)
            );
            
            -- Create indices for better query performance
            CREATE INDEX IF NOT EXISTS idx_historical_data_symbol_timeframe ON historical_data(symbol_id, timeframe_id);
            CREATE INDEX IF NOT EXISTS idx_historical_data_open_time ON historical_data(open_time);
            CREATE INDEX IF NOT EXISTS idx_correlations_main ON correlations(symbol_id, timeframe_id, indicator_id, lag);
            CREATE INDEX IF NOT EXISTS idx_leaderboard_main ON leaderboard(symbol_id, timeframe_id, indicator_id, lag);
            CREATE INDEX IF NOT EXISTS idx_leaderboard_correlation ON leaderboard(correlation_type, correlation_value);
        """)
        
        # Insert initial symbol and timeframe and get their IDs
        cursor.execute("INSERT OR IGNORE INTO symbols (symbol) VALUES (?)", (symbol,))
        cursor.execute("SELECT id FROM symbols WHERE symbol = ?", (symbol,))
        symbol_id = cursor.fetchone()[0]
        
        cursor.execute("INSERT OR IGNORE INTO timeframes (timeframe) VALUES (?)", (timeframe,))
        cursor.execute("SELECT id FROM timeframes WHERE timeframe = ?", (timeframe,))
        timeframe_id = cursor.fetchone()[0]
        
        conn.commit()
        logger.info(f"Database schema initialized/verified: {db_path} (symbol_id={symbol_id}, timeframe_id={timeframe_id})")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Error initializing database: {e}")
        if conn:
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
            conn.close()
        return False

def recover_database(db_path: str, symbol: str, timeframe: str) -> bool:
    """Attempt to recover a corrupted database."""
    try:
        # First try to backup the corrupted file
        backup_path = f"{db_path}.bak"
        if os.path.exists(db_path):
            shutil.copy2(db_path, backup_path)
        # Try to repair using SQLite's built-in recovery
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check;")
            result = cursor.fetchone()
            if result[0] != "ok":
                raise sqlite3.DatabaseError("Database integrity check failed")
            conn.close()
            return True
        except sqlite3.DatabaseError:
            # If corrupted, try to recover
            if os.path.exists(db_path):
                os.remove(db_path)  # Remove corrupted file
            # Recreate schema
            if not initialize_database(db_path, symbol, timeframe):
                raise sqlite3.DatabaseError("Failed to recreate database schema")
            return True
    except Exception as e:
        logger.error(f"Error recovering database: {e}", exc_info=True)
        # Restore from backup if available
        backup_path = f"{db_path}.bak"
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, db_path)
        return False

def _get_or_create_id(conn: sqlite3.Connection, table: str, column: str, value: Any) -> int:
    """Gets ID for value in lookup table, creating if necessary (case-insensitive lookup). Assumes caller handles transactions."""
    cursor = conn.cursor()
    # Ensure value is string for consistent lookup/insertion, handle None
    str_value = str(value) if value is not None else None
    if str_value is None: raise ValueError(f"Cannot get/create ID for None value in {table}.{column}")
    # Use lower case for case-insensitive lookup
    lookup_value = str_value.lower()

    try:
        # Check if the value (case-insensitive) already exists
        cursor.execute(f"SELECT id FROM {table} WHERE LOWER({column}) = ?", (lookup_value,))
        result = cursor.fetchone()
        if result:
            return result[0] # Return existing ID
        else:
            # Insert the new value (using original case)
            cursor.execute(f"INSERT INTO {table} ({column}) VALUES (?)", (str_value,))
            new_id = cursor.lastrowid
            # Verify insertion and ID retrieval
            if new_id is None:
                 # This can happen in rare cases, re-query to be sure
                 logger.warning(f"Insert into '{table}' for '{str_value}' did not return lastrowid. Re-querying.")
                 cursor.execute(f"SELECT id FROM {table} WHERE LOWER({column}) = ?", (lookup_value,))
                 result = cursor.fetchone()
                 if result: return result[0]
                 else: raise sqlite3.OperationalError(f"Cannot retrieve ID for '{str_value}' in '{table}' after insert.")
            # logger.debug(f"Inserted '{str_value}' into '{table}', ID: {new_id}") # Reduce log noise
            return new_id
    except sqlite3.IntegrityError: # Handle potential race condition if another process inserts concurrently
        logger.warning(f"IntegrityError inserting '{str_value}' into '{table}'. Re-querying.")
        cursor.execute(f"SELECT id FROM {table} WHERE LOWER({column}) = ?", (lookup_value,))
        result = cursor.fetchone()
        if result: return result[0]
        else: # This should ideally not happen if IntegrityError occurred due to uniqueness
            logger.critical(f"CRITICAL: Cannot retrieve ID for '{str_value}' in '{table}' after IntegrityError.")
            raise
    except sqlite3.Error as e:
        logger.error(f"DB error getting/creating ID for '{str_value}' in '{table}': {e}", exc_info=True)
        raise # Re-raise the original SQLite error

def get_or_create_indicator_config_id(conn: sqlite3.Connection, indicator_name: str, params: Dict[str, Any]) -> int:
    """Gets the ID for an indicator config, creating if necessary. Manages its own transaction."""
    cursor = conn.cursor()
    try:
        # Use IMMEDIATE transaction for potentially faster write lock acquisition if creating
        cursor.execute("BEGIN IMMEDIATE;")
        # Get the ID for the indicator name itself
        indicator_id = _get_or_create_id(conn, 'indicators', 'name', indicator_name)
        # Create a stable hash of the parameters (using utils function for consistency)
        config_hash = utils.get_config_hash(params)
        # Serialize parameters to JSON string for storage
        config_str = json.dumps(params, sort_keys=True, separators=(',', ':'))

        # Check if this exact config (indicator_id + hash) already exists
        cursor.execute("SELECT id, config_json FROM indicator_configs WHERE indicator_id = ? AND config_hash = ?", (indicator_id, config_hash))
        result = cursor.fetchone()

        if result:
            config_id_found, existing_json = result
            # Optional: Verify stored JSON matches current serialization (defense against hash collisions)
            if existing_json == config_str:
                # logger.debug(f"Found existing config ID {config_id_found} for {indicator_name} / hash {config_hash[:8]}") # Reduce noise
                conn.commit(); return config_id_found
            else:
                 # This indicates a hash collision or data corruption - critical error
                 logger.error(f"HASH COLLISION/DATA MISMATCH: ID {indicator_id}, hash {config_hash}. DB JSON: {existing_json}, Expected JSON: {config_str}")
                 conn.rollback(); raise ValueError("Hash collision or data mismatch detected for indicator config.")
        else:
            # Insert the new configuration
            try:
                cursor.execute("INSERT INTO indicator_configs (indicator_id, config_hash, config_json) VALUES (?, ?, ?)", (indicator_id, config_hash, config_str))
                new_config_id = cursor.lastrowid
                if new_config_id is None:
                    # Re-query if insert didn't return ID (should be rare)
                    cursor.execute("SELECT id FROM indicator_configs WHERE indicator_id = ? AND config_hash = ?", (indicator_id, config_hash))
                    result_requery = cursor.fetchone()
                    if result_requery: new_config_id = result_requery[0]
                    else: raise sqlite3.Error("INSERT successful but could not retrieve new config_id.")
                conn.commit()
                # logger.debug(f"Inserted new config for '{indicator_name}' (ID: {new_config_id}), Hash: {config_hash[:8]}") # Reduce noise
                return new_config_id
            except sqlite3.IntegrityError: # Handle race condition if another process inserted just now
                conn.rollback() # Rollback the failed insert attempt
                logger.warning(f"IntegrityError inserting config for indicator ID {indicator_id}, hash {config_hash}. Re-querying.")
                # Re-query to get the ID inserted by the other process
                cursor.execute("SELECT id, config_json FROM indicator_configs WHERE indicator_id = ? AND config_hash = ?", (indicator_id, config_hash))
                result = cursor.fetchone()
                if result:
                    config_id_found, existing_json = result
                    # Verify JSON again just in case
                    if existing_json == config_str:
                         # logger.debug(f"Found config ID {config_id_found} after IntegrityError.") # Reduce noise
                         return config_id_found
                    else:
                         logger.critical(f"CRITICAL: IntegrityError AND JSON mismatch after re-query: ID {indicator_id}, hash {config_hash}.")
                         raise ValueError("Data inconsistency after IntegrityError.")
                else:
                    # This state should be highly unlikely if IntegrityError was due to uniqueness constraint
                    logger.critical(f"CRITICAL: Cannot find config ID {indicator_id}, hash {config_hash} after IntegrityError and rollback.")
                    raise ValueError("Cannot resolve config ID after IntegrityError.")
    except Exception as e:
         logger.error(f"Error get/create config ID for '{indicator_name}' / {params}: {e}", exc_info=True)
         try: conn.rollback() # Ensure rollback on any error
         except Exception as rb_err: logger.error(f"Rollback failed after error in get_or_create_indicator_config_id: {rb_err}")
         raise # Re-raise the original exception

# --- Batch Insert Function ---
def batch_insert_correlations(conn: sqlite3.Connection, data_to_insert: List[Tuple[int, int, int, int, Optional[float]]]) -> bool:
    """Inserts or replaces multiple correlation values in a single transaction."""
    if not data_to_insert:
        logger.info("No correlation data provided for batch insert.")
        return True

    query = "INSERT OR REPLACE INTO correlations (symbol_id, timeframe_id, indicator_config_id, lag, correlation_value) VALUES (?, ?, ?, ?, ?);"
    cursor = conn.cursor()
    try:
        # Prepare data: ensure correlation_value is float or None
        prepared_data = []
        for s_id, t_id, cfg_id, lag, corr_val in data_to_insert:
            # Convert to float if numeric, otherwise None. Handles NaN, ints, etc.
            db_value = float(corr_val) if pd.notna(corr_val) and isinstance(corr_val, (int, float, np.number)) else None
            prepared_data.append((s_id, t_id, cfg_id, lag, db_value))

        # Execute in a transaction for performance and atomicity
        cursor.execute("BEGIN IMMEDIATE;") # Use IMMEDIATE for faster write lock acquisition
        cursor.executemany(query, prepared_data)
        conn.commit()
        logger.info(f"Batch inserted/replaced correlations for {len(prepared_data)} records.")
        return True
    except sqlite3.Error as e:
        logger.error(f"DB error during batch correlation insert: {e}", exc_info=True)
        try: conn.rollback(); logger.warning("Rolled back batch correlation insert due to DB error.")
        except Exception as rb_err: logger.error(f"Rollback attempt failed after batch insert error: {rb_err}")
        return False
    except Exception as e: # Catch other errors like data preparation issues
        logger.error(f"Unexpected error during batch correlation preparation/insert: {e}", exc_info=True)
        try: conn.rollback()
        except Exception as rb_err: logger.error(f"Rollback attempt failed after unexpected batch insert error: {rb_err}")
        return False

# --- Fetch Correlations Function (BATCHED) ---
def fetch_correlations(conn: sqlite3.Connection, symbol_id: int, timeframe_id: int, config_ids: List[int]) -> Dict[int, List[Optional[float]]]:
    """Fetches correlations in batches, returns dict {config_id: [corr_lag1, ..., corr_lagN]}."""
    if not config_ids: return {}
    logger.info(f"Fetching correlations for {len(config_ids)} IDs (Sym: {symbol_id}, TF: {timeframe_id})...")

    # Determine batch size based on SQLite variable limit
    # Subtract 2 for symbol_id and timeframe_id placeholders
    batch_size = max(1, SQLITE_MAX_VARIABLE_NUMBER - 2)
    num_batches = math.ceil(len(config_ids) / batch_size)
    logger.info(f"Fetching {len(config_ids)} correlations in {num_batches} batches (size {batch_size}).")

    all_rows = []
    cursor = conn.cursor()
    try:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_ids = config_ids[start_idx:end_idx]
            if not batch_ids: continue # Should not happen with ceil, but safety check

            placeholders = ','.join('?' * len(batch_ids)) # Correct way to generate placeholders
            query = f"""
                SELECT indicator_config_id, lag, correlation_value
                FROM correlations
                WHERE symbol_id = ? AND timeframe_id = ? AND indicator_config_id IN ({placeholders})
                ORDER BY indicator_config_id, lag;
            """
            # Parameters must match placeholders: symbol_id, timeframe_id, then all batch_ids
            params = [symbol_id, timeframe_id] + batch_ids
            logger.debug(f"Executing fetch batch {i+1}/{num_batches} ({len(batch_ids)} IDs)")
            cursor.execute(query, params)
            batch_rows = cursor.fetchall()
            all_rows.extend(batch_rows)
            logger.debug(f" Batch {i+1} fetched {len(batch_rows)} rows.")

        if not all_rows:
            logger.warning(f"No correlation data found for requested IDs (Sym: {symbol_id}, TF: {timeframe_id}) after fetching batches.")
            # Return empty lists for all requested IDs if no data found
            return {cfg_id: [] for cfg_id in config_ids}

        # Determine max lag across ALL fetched rows to structure the result dict correctly
        # Ensure lag is treated as int, default to 0 if no valid lags found
        max_lag_found = max((row[1] for row in all_rows if isinstance(row[1], int)), default=0)
        if max_lag_found <= 0:
            logger.warning("Max lag found <= 0 after fetching batches.")
            return {cfg_id: [] for cfg_id in config_ids} # Still return empty lists

        logger.debug(f"Max lag found across all batches: {max_lag_found}")

        # Initialize results dictionary for ALL originally requested IDs with None placeholders
        results_dict: Dict[int, List[Optional[float]]] = {cfg_id: [None] * max_lag_found for cfg_id in config_ids}

        # Populate the results dictionary from fetched rows
        processed_rows = 0
        for cfg_id_db, lag, value in all_rows:
            # Ensure lag is a valid integer within the determined range
            if isinstance(lag, int) and 1 <= lag <= max_lag_found:
                 if cfg_id_db in results_dict:
                      # Assign value (convert to float or None) to the correct index (lag-1)
                      results_dict[cfg_id_db][lag - 1] = float(value) if value is not None else None
                      processed_rows += 1
                 # else: Silently ignore rows for config IDs not originally requested (shouldn't happen with IN clause)
            else:
                 logger.warning(f"Invalid lag value {lag} (type: {type(lag)}) encountered for config_id {cfg_id_db}. Ignoring row.")

        logger.info(f"Fetched and processed {processed_rows} correlation points for {len(config_ids)} configs up to lag {max_lag_found}.")
        return results_dict
    except sqlite3.Error as e:
        logger.error(f"SQLite error during batched correlation fetch: {e}", exc_info=True)
        return {} # Return empty on DB error
    except Exception as e: # Catch other potential errors
        logger.error(f"Error fetching/processing correlations: {e}", exc_info=True)
        return {}


# --- Functions for Custom Mode / Data Retrieval ---

def get_max_lag_for_pair(conn: sqlite3.Connection, symbol_id: int, timeframe_id: int) -> Optional[int]:
    """Gets the maximum lag value stored in the correlations table for a symbol/timeframe pair."""
    query = "SELECT MAX(lag) FROM correlations WHERE symbol_id = ? AND timeframe_id = ?;"
    cursor = conn.cursor()
    try:
        cursor.execute(query, (symbol_id, timeframe_id))
        result = cursor.fetchone()
        # Check if result exists and the value is not None
        if result and result[0] is not None:
            max_lag = int(result[0])
            logger.info(f"Determined max lag from existing DB data: {max_lag} for SymbolID {symbol_id}, TFID {timeframe_id}")
            return max_lag
        else:
            logger.warning(f"No correlation data found to determine max lag for SymbolID {symbol_id}, TFID {timeframe_id}.")
            return None
    except (sqlite3.Error, ValueError, TypeError) as e: # Catch DB errors and potential type conversion errors
        logger.error(f"Error getting max lag for pair (SymID {symbol_id}, TFID {timeframe_id}): {e}", exc_info=True)
        return None

def get_distinct_config_ids_for_pair(conn: sqlite3.Connection, symbol_id: int, timeframe_id: int) -> List[int]:
    """Gets a list of distinct indicator_config_ids that have correlation data for a symbol/timeframe pair."""
    query = "SELECT DISTINCT indicator_config_id FROM correlations WHERE symbol_id = ? AND timeframe_id = ?;"
    cursor = conn.cursor()
    config_ids = []
    try:
        cursor.execute(query, (symbol_id, timeframe_id))
        rows = cursor.fetchall()
        # Extract valid integer IDs
        config_ids = [row[0] for row in rows if row[0] is not None and isinstance(row[0], int)]
        logger.info(f"Found {len(config_ids)} distinct config IDs with correlation data for SymbolID {symbol_id}, TFID {timeframe_id}.")
        return config_ids
    except sqlite3.Error as e:
        logger.error(f"Error getting distinct config IDs for pair (SymID {symbol_id}, TFID {timeframe_id}): {e}", exc_info=True)
        return []

def get_indicator_configs_by_ids(conn: sqlite3.Connection, config_ids: List[int]) -> List[Dict[str, Any]]:
    """Fetches indicator configuration details (name, params) for a list of config_ids in batches."""
    if not config_ids:
        return []

    # Use full batch size here as only ? placeholders are needed for config_ids
    batch_size = SQLITE_MAX_VARIABLE_NUMBER
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

            placeholders = ','.join('?' * len(batch_ids)) # Generate placeholders for IN clause
            query = f"""
                SELECT
                    ic.id,
                    i.name AS indicator_name,
                    ic.config_json
                FROM indicator_configs ic
                JOIN indicators i ON ic.indicator_id = i.id
                WHERE ic.id IN ({placeholders});
            """
            logger.debug(f"Executing fetch batch {i+1}/{num_batches} for config details ({len(batch_ids)} IDs)")
            cursor.execute(query, batch_ids) # Pass only the batch IDs as parameters
            batch_rows = cursor.fetchall()
            all_rows.extend(batch_rows)
            logger.debug(f" Batch {i+1} fetched {len(batch_rows)} config detail rows.")

        configs_processed = []
        processed_count = 0
        for cfg_id, name, json_str in all_rows:
            try:
                # Parse the JSON string back into a dictionary
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
    except Exception as e: # Catch other errors like JSON parsing
        logger.error(f"Error fetching indicator config details by IDs: {e}", exc_info=True)
        return []

class SQLiteManager:
    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.conn = None
        self._initialize_db()
        
    def _initialize_db(self):
        """Initialize database connection and create tables if they don't exist."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self._create_tables()
        except sqlite3.Error as e:
            logging.error(f"Database initialization failed: {e}")
            raise
            
    def _create_tables(self):
        """Create required tables if they don't exist."""
        cursor = self.conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS timeframes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timeframe TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS historical_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol_id INTEGER NOT NULL,
                timeframe_id INTEGER NOT NULL,
                open_time INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                close_time INTEGER NOT NULL,
                quote_asset_volume REAL,
                number_of_trades INTEGER,
                taker_buy_base_asset_volume REAL,
                taker_buy_quote_asset_volume REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (symbol_id) REFERENCES symbols(id) ON DELETE CASCADE,
                FOREIGN KEY (timeframe_id) REFERENCES timeframes(id) ON DELETE CASCADE,
                UNIQUE(symbol_id, timeframe_id, open_time)
            );
        """)
        self.conn.commit()
        
    def _execute(self, query: str, params: tuple = None) -> sqlite3.Cursor:
        """Execute a SQL query with parameters.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            sqlite3.Cursor: Cursor with query results
        """
        try:
            cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            self.conn.commit()
            return cursor
        except sqlite3.Error as e:
            logging.error(f"Query execution failed: {e}")
            self.conn.rollback()
            raise

    def insert(self, table, row_dict):
        cols = ', '.join(row_dict.keys())
        placeholders = ', '.join(['?'] * len(row_dict))
        sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
        values = list(row_dict.values())
        cur = self.conn.cursor()
        cur.execute(sql, values)
        self.conn.commit()

    def select(self, table, columns):
        cols = ', '.join(columns)
        sql = f"SELECT {cols} FROM {table}"
        cur = self.conn.cursor()
        cur.execute(sql)
        return cur.fetchall()