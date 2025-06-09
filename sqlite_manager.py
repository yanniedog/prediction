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
from datetime import datetime

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

        # Create new database
        conn = create_connection(db_path)
        if not conn:
            raise sqlite3.DatabaseError("Failed to create database connection")

        cursor = conn.cursor()

        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY,
                symbol TEXT UNIQUE NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS timeframes (
                id INTEGER PRIMARY KEY,
                timeframe TEXT UNIQUE NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS indicators (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                type TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS indicator_configs (
                id INTEGER PRIMARY KEY,
                indicator_id INTEGER NOT NULL,
                config_json TEXT NOT NULL,
                FOREIGN KEY (indicator_id) REFERENCES indicators(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_data (
                id INTEGER PRIMARY KEY,
                symbol_id INTEGER NOT NULL,
                timeframe_id INTEGER NOT NULL,
                open_time TIMESTAMP NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (symbol_id) REFERENCES symbols(id),
                FOREIGN KEY (timeframe_id) REFERENCES timeframes(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS correlations (
                id INTEGER PRIMARY KEY,
                symbol_id INTEGER NOT NULL,
                timeframe_id INTEGER NOT NULL,
                indicator_id INTEGER NOT NULL,
                config_id INTEGER NOT NULL,
                lag INTEGER NOT NULL,
                correlation_type TEXT NOT NULL,
                correlation_value REAL NOT NULL,
                calculation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (symbol_id) REFERENCES symbols(id),
                FOREIGN KEY (timeframe_id) REFERENCES timeframes(id),
                FOREIGN KEY (indicator_id) REFERENCES indicators(id),
                FOREIGN KEY (config_id) REFERENCES indicator_configs(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS leaderboard (
                id INTEGER PRIMARY KEY,
                lag INTEGER NOT NULL,
                correlation_type TEXT NOT NULL,
                correlation_value REAL NOT NULL,
                indicator_name TEXT NOT NULL,
                config_json TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                dataset_daterange TEXT NOT NULL,
                calculation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                config_id_source_db INTEGER,
                source_db_name TEXT
            )
        """)

        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_historical_data_symbol_timeframe ON historical_data(symbol_id, timeframe_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_historical_data_open_time ON historical_data(open_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_correlations_symbols ON correlations(symbol_id, timeframe_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_correlations_indicators ON correlations(indicator_id, config_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_leaderboard_correlation ON leaderboard(correlation_type, correlation_value)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_leaderboard_lag ON leaderboard(lag)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_leaderboard_lag_val ON leaderboard(lag, correlation_value)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_leaderboard_indicator_config ON leaderboard(indicator_name, config_json)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_leaderboard_symbol_timeframe ON leaderboard(symbol, timeframe)")

        # Insert initial data
        cursor.execute("INSERT OR IGNORE INTO symbols (symbol) VALUES (?)", (symbol,))
        cursor.execute("INSERT OR IGNORE INTO timeframes (timeframe) VALUES (?)", (timeframe,))

        conn.commit()
        return True

    except sqlite3.Error as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        return False
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

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
        # Map table names to their actual column names
        column_map = {
            'symbols': 'symbol',
            'timeframes': 'timeframe',
            'indicators': 'name'
        }
        actual_column = column_map.get(table, column)
        
        # Check if the value (case-insensitive) already exists
        cursor.execute(f"SELECT id FROM {table} WHERE LOWER({actual_column}) = ?", (lookup_value,))
        result = cursor.fetchone()
        if result:
            return result[0]
            
        # Insert new value
        cursor.execute(f"INSERT INTO {table} ({actual_column}) VALUES (?)", (str_value,))
        conn.commit()
        return cursor.lastrowid
        
    except sqlite3.Error as e:
        logger.error(f"DB error getting/creating ID for '{value}' in '{table}': {e}", exc_info=True)
        try: conn.rollback()
        except Exception as rb_err: logger.error(f"Rollback failed in _get_or_create_id: {rb_err}")
        raise

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
    """Class for managing SQLite database operations."""
    
    def __init__(self, db_path: Union[str, Path], timeout: float = 30.0):
        """Initialize SQLiteManager with database path.
        
        Args:
            db_path: Path to SQLite database file
            timeout: Connection timeout in seconds
        """
        self.db_path = Path(db_path)
        self.timeout = timeout
        self.connection = None
        self.initialize_database()
    
    def create_connection(self) -> Optional[sqlite3.Connection]:
        """Create a connection to the SQLite database."""
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=self.timeout)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database {self.db_path}: {e}")
            return None
    
    def initialize_database(self) -> bool:
        """Initialize database schema and create necessary tables."""
        try:
            conn = self.create_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS symbols (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT UNIQUE NOT NULL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS timeframes (
                    id INTEGER PRIMARY KEY,
                    timeframe TEXT UNIQUE NOT NULL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS indicators (
                    id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    type TEXT NOT NULL,
                    description TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS indicator_configs (
                    id INTEGER PRIMARY KEY,
                    indicator_id INTEGER NOT NULL,
                    config_json TEXT NOT NULL,
                    FOREIGN KEY (indicator_id) REFERENCES indicators(id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS historical_data (
                    id INTEGER PRIMARY KEY,
                    symbol_id INTEGER NOT NULL,
                    timeframe_id INTEGER NOT NULL,
                    open_time TIMESTAMP NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    FOREIGN KEY (symbol_id) REFERENCES symbols(id),
                    FOREIGN KEY (timeframe_id) REFERENCES timeframes(id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS correlations (
                    id INTEGER PRIMARY KEY,
                    symbol_id INTEGER NOT NULL,
                    timeframe_id INTEGER NOT NULL,
                    indicator_id INTEGER NOT NULL,
                    config_id INTEGER NOT NULL,
                    lag INTEGER NOT NULL,
                    correlation_type TEXT NOT NULL,
                    correlation_value REAL NOT NULL,
                    calculation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (symbol_id) REFERENCES symbols(id),
                    FOREIGN KEY (timeframe_id) REFERENCES timeframes(id),
                    FOREIGN KEY (indicator_id) REFERENCES indicators(id),
                    FOREIGN KEY (config_id) REFERENCES indicator_configs(id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS leaderboard (
                    id INTEGER PRIMARY KEY,
                    lag INTEGER NOT NULL,
                    correlation_type TEXT NOT NULL CHECK(correlation_type IN ('positive', 'negative')),
                    correlation_value REAL NOT NULL,
                    indicator_name TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    dataset_daterange TEXT NOT NULL,
                    calculation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    config_id_source_db INTEGER,
                    source_db_name TEXT,
                    UNIQUE(lag, correlation_type, indicator_name, config_json, symbol, timeframe)
                )
            """)
            
            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_historical_data_symbol_timeframe ON historical_data(symbol_id, timeframe_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_historical_data_open_time ON historical_data(open_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_correlations_symbols ON correlations(symbol_id, timeframe_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_leaderboard_correlation ON leaderboard(correlation_value)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_leaderboard_lag ON leaderboard(lag)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_leaderboard_lag_val ON leaderboard(lag, correlation_value)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_leaderboard_indicator_config ON leaderboard(indicator_name, config_id_source_db)")
            
            conn.commit()
            conn.close()
            logger.info(f"Database schema initialized/verified: {self.db_path}")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error initializing database {self.db_path}: {e}")
            if conn:
                conn.close()
            return False
    
    def create_table(self, table_name: str, columns: Dict[str, str]) -> bool:
        """Create a new table with specified columns.
        
        Args:
            table_name: Name of the table to create
            columns: Dictionary mapping column names to SQLite types
            
        Returns:
            bool: True if table was created successfully
        """
        try:
            conn = self.create_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            
            # Build CREATE TABLE statement
            column_defs = [f"{col} {type_}" for col, type_ in columns.items()]
            create_stmt = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(column_defs)})"
            
            cursor.execute(create_stmt)
            conn.commit()
            conn.close()
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error creating table {table_name}: {e}")
            if conn:
                conn.close()
            return False
    
    def insert_data(self, table_name: str, data: Dict[str, Any]) -> bool:
        """Insert a single row of data into a table."""
        # Convert non-primitive types to strings (e.g., pd.Timestamp)
        clean_data = {k: (str(v) if hasattr(v, 'isoformat') or type(v).__name__ == 'Timestamp' else v) for k, v in data.items()}
        try:
            conn = self.create_connection()
            if not conn:
                return False
            cursor = conn.cursor()
            # Build INSERT statement
            columns = list(clean_data.keys())
            placeholders = ['?' for _ in columns]
            insert_stmt = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
            cursor.execute(insert_stmt, list(clean_data.values()))
            conn.commit()
            conn.close()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error inserting data into {table_name}: {e}")
            if conn:
                conn.close()
            return False
    
    def insert_many(self, table_name: str, data_list: List[Dict[str, Any]]) -> bool:
        """Insert multiple rows of data into a table.
        
        Args:
            table_name: Name of the table to insert into
            data_list: List of dictionaries mapping column names to values
            
        Returns:
            bool: True if all inserts were successful
        """
        if not data_list:
            return True
            
        try:
            conn = self.create_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            
            # Build INSERT statement using first row's keys
            columns = list(data_list[0].keys())
            placeholders = ['?' for _ in columns]
            insert_stmt = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
            
            # Prepare values for each row
            values = [[row[col] for col in columns] for row in data_list]
            
            cursor.executemany(insert_stmt, values)
            conn.commit()
            conn.close()
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error inserting multiple rows into {table_name}: {e}")
            if conn:
                conn.close()
            return False
    
    def execute_query(self, query: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results as list of dictionaries.
        
        Args:
            query: SQL query string
            params: Tuple of parameters for the query
            
        Returns:
            List of dictionaries containing query results
        """
        try:
            conn = self.create_connection()
            if not conn:
                return []
            
            cursor = conn.cursor()
            cursor.execute(query, params)
            results = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return results
            
        except sqlite3.Error as e:
            logger.error(f"Error executing query: {e}")
            if conn:
                conn.close()
            return []
    
    def get_or_create_id(self, table_name: str, name_column: str, value: str) -> Optional[int]:
        """Get or create an ID for a value in a lookup table.
        
        Args:
            table_name: Name of the lookup table
            name_column: Name of the column containing the value
            value: Value to look up or insert
            
        Returns:
            Optional[int]: ID of the value, or None if operation failed
        """
        try:
            conn = self.create_connection()
            if not conn:
                return None
            
            cursor = conn.cursor()
            
            # Try to get existing ID
            cursor.execute(f"SELECT id FROM {table_name} WHERE {name_column} = ?", (value,))
            result = cursor.fetchone()
            
            if result:
                id_ = result['id']
            else:
                # Insert new value
                cursor.execute(f"INSERT INTO {table_name} ({name_column}) VALUES (?)", (value,))
                id_ = cursor.lastrowid
                conn.commit()
            
            conn.close()
            return id_
            
        except sqlite3.Error as e:
            logger.error(f"Error getting/creating ID for {value} in {table_name}: {e}")
            if conn:
                conn.close()
            return None
    
    def close(self):
        """Close the database connection if it exists."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def insert(self, table_name: str, data: dict) -> bool:
        """Insert a single row into a table."""
        # Convert non-primitive types to strings (e.g., pd.Timestamp)
        clean_data = {k: (str(v) if hasattr(v, 'isoformat') or type(v).__name__ == 'Timestamp' else v) for k, v in data.items()}
        return self.insert_data(table_name, clean_data)

    def select(self, table_name: str, columns: list) -> list:
        """Select rows from a table."""
        query = f"SELECT {', '.join(columns)} FROM {table_name}"
        conn = self.create_connection()
        if not conn:
            return []
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        return rows

    def update(self, table_name: str, data: dict, where: str = None) -> bool:
        """Update rows in a table."""
        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
        query = f"UPDATE {table_name} SET {set_clause}"
        if where:
            query += f" WHERE {where}"
        conn = self.create_connection()
        if not conn:
            return False
        cursor = conn.cursor()
        cursor.execute(query, list(data.values()))
        conn.commit()
        conn.close()
        return True

    def delete(self, table_name: str, where: str = None) -> bool:
        """Delete rows from a table."""
        query = f"DELETE FROM {table_name}"
        if where:
            query += f" WHERE {where}"
        conn = self.create_connection()
        if not conn:
            return False
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
        conn.close()
        return True