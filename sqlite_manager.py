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
import re

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
                config_hash TEXT NOT NULL,
                config_json TEXT NOT NULL,
                UNIQUE(indicator_id, config_hash),
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
                indicator_config_id INTEGER NOT NULL,
                lag INTEGER NOT NULL,
                correlation_value REAL NOT NULL,
                calculation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (symbol_id) REFERENCES symbols(id),
                FOREIGN KEY (timeframe_id) REFERENCES timeframes(id),
                FOREIGN KEY (indicator_config_id) REFERENCES indicator_configs(id)
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
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_correlations_indicators ON correlations(indicator_config_id)")
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
        # Don't commit here - let the caller handle the transaction
        return cursor.lastrowid
        
    except sqlite3.Error as e:
        logger.error(f"DB error getting/creating ID for '{value}' in '{table}': {e}", exc_info=True)
        # Don't rollback here - let the caller handle the transaction
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
        
        # Try to initialize database and set connection
        try:
            if self.initialize_database():
                self.connection = self.create_connection()
        except Exception as e:
            logger.error(f"Failed to initialize database {self.db_path}: {e}")
            self.connection = None
    
    def create_connection(self) -> Optional[sqlite3.Connection]:
        """Create a connection to the SQLite database."""
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=self.timeout)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database {self.db_path}: {e}")
            return None
    
    def connect(self) -> bool:
        """Connect to the database and store the connection."""
        try:
            self.connection = self.create_connection()
            return self.connection is not None
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return False
    
    def initialize_database(self) -> bool:
        """Initialize database schema and create necessary tables."""
        try:
            # Check if the database path is valid
            try:
                # Try to create parent directories
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                logger.error(f"Cannot create database directory {self.db_path.parent}: {e}")
                return False
            
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
                    config_hash TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    UNIQUE(indicator_id, config_hash),
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
                    indicator_config_id INTEGER NOT NULL,
                    lag INTEGER NOT NULL,
                    correlation_value REAL NOT NULL,
                    calculation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (symbol_id) REFERENCES symbols(id),
                    FOREIGN KEY (timeframe_id) REFERENCES timeframes(id),
                    FOREIGN KEY (indicator_config_id) REFERENCES indicator_configs(id)
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
        except Exception as e:
            logger.error(f"Unexpected error initializing database {self.db_path}: {e}")
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
            
        Raises:
            ValueError: If table name is invalid or columns are invalid
        """
        # Validate table name
        if not table_name or not isinstance(table_name, str):
            raise ValueError("Table name must be a non-empty string")
        
        # Check for invalid characters in table name
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
            raise ValueError(f"Invalid table name: {table_name}. Must start with letter or underscore and contain only alphanumeric characters and underscores.")
        
        # Validate columns
        if not columns or not isinstance(columns, dict):
            raise ValueError("Columns must be a non-empty dictionary")
        
        for col_name, col_type in columns.items():
            if not col_name or not isinstance(col_name, str):
                raise ValueError("Column names must be non-empty strings")
            if not col_type or not isinstance(col_type, str):
                raise ValueError("Column types must be non-empty strings")
            
            # Check for invalid characters in column names
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', col_name):
                raise ValueError(f"Invalid column name: {col_name}. Must start with letter or underscore and contain only alphanumeric characters and underscores.")
        
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
            # Use transaction connection if available, otherwise create new connection
            conn = getattr(self, '_transaction_conn', None) or self.create_connection()
            if not conn:
                return False
            cursor = conn.cursor()
            # Build INSERT statement
            columns = list(clean_data.keys())
            placeholders = ['?' for _ in columns]
            insert_stmt = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
            cursor.execute(insert_stmt, list(clean_data.values()))
            
            # Only commit if not in a transaction
            if not hasattr(self, '_transaction_conn'):
                conn.commit()
                conn.close()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error inserting data into {table_name}: {e}")
            if conn and not hasattr(self, '_transaction_conn'):
                conn.close()
            return False
    
    def insert_many(self, table_name: str, data_list: List[Dict[str, Any]]) -> bool:
        """Insert multiple rows of data into a table.
        
        Args:
            table_name: Name of the table to insert into
            data_list: List of dictionaries containing row data
            
        Returns:
            bool: True if insertion was successful
        """
        if not data_list:
            return True
        
        try:
            conn = self.create_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            
            # Get column names from first row
            columns = list(data_list[0].keys())
            
            # Prepare data for insertion - convert timestamps to strings
            processed_data = []
            for row in data_list:
                processed_row = {}
                for key, value in row.items():
                    if hasattr(value, 'strftime'):  # Handle pandas Timestamp and datetime objects
                        processed_row[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        processed_row[key] = value
                processed_data.append(processed_row)
            
            # Build INSERT statement
            placeholders = ', '.join(['?' for _ in columns])
            insert_stmt = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
            
            # Execute batch insert
            cursor.executemany(insert_stmt, [tuple(row[col] for col in columns) for row in processed_data])
            conn.commit()
            conn.close()
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error inserting multiple rows into {table_name}: {e}")
            if conn:
                conn.close()
            return False
    
    def execute_query(self, query: str, params: Tuple = ()) -> List[Tuple]:
        """Execute a SQL query and return results as list of tuples.
        
        Args:
            query: SQL query string
            params: Tuple of parameters for the query
            
        Returns:
            List of tuples containing query results
            
        Raises:
            sqlite3.Error: If there's an error executing the query
        """
        conn = None
        try:
            conn = self.create_connection()
            if not conn:
                raise sqlite3.Error("Failed to create database connection")
            
            cursor = conn.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            return results
            
        except sqlite3.Error as e:
            logger.error(f"Error executing query: {e}")
            raise  # Re-raise the exception
        finally:
            if conn:
                conn.close()
    
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
                id_ = result[0]  # Access as tuple
            else:
                # Insert new value
                cursor.execute(f"INSERT INTO {table_name} ({name_column}) VALUES (?)", (value,))
                id_ = cursor.lastrowid
            
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

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            bool: True if table exists, False otherwise
        """
        try:
            conn = self.create_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            result = cursor.fetchone()
            conn.close()
            return result is not None
            
        except sqlite3.Error as e:
            logger.error(f"Error checking if table {table_name} exists: {e}")
            return False

    def query(self, query: str, params: Tuple = ()) -> List[Tuple]:
        """Execute a SQL query and return results as list of tuples.
        
        Args:
            query: SQL query string
            params: Tuple of parameters for the query
            
        Returns:
            List of tuples containing query results
        """
        return self.execute_query(query, params)

    def update_data(self, table_name: str, data: Dict[str, Any], where_conditions: Dict[str, Any]) -> bool:
        """Update rows in a table based on where conditions.
        
        Args:
            table_name: Name of the table to update
            data: Dictionary of column names and new values
            where_conditions: Dictionary of column names and values for WHERE clause
            
        Returns:
            bool: True if update was successful
        """
        try:
            conn = self.create_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            
            # Build SET clause
            set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
            
            # Build WHERE clause
            where_clause = ' AND '.join([f"{k} = ?" for k in where_conditions.keys()])
            
            # Build complete query
            query = f"UPDATE {table_name} SET {set_clause}"
            if where_conditions:
                query += f" WHERE {where_clause}"
            
            # Prepare parameters
            params = list(data.values()) + list(where_conditions.values())
            
            cursor.execute(query, params)
            conn.commit()
            conn.close()
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error updating data in {table_name}: {e}")
            if conn:
                conn.close()
            return False

    def delete_data(self, table_name: str, where_conditions: Dict[str, Any]) -> bool:
        """Delete rows from a table based on where conditions.
        
        Args:
            table_name: Name of the table to delete from
            where_conditions: Dictionary of column names and values for WHERE clause
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            conn = self.create_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            
            # Build WHERE clause
            where_clause = ' AND '.join([f"{k} = ?" for k in where_conditions.keys()])
            
            # Build complete query
            query = f"DELETE FROM {table_name}"
            if where_conditions:
                query += f" WHERE {where_clause}"
            
            # Execute query
            cursor.execute(query, list(where_conditions.values()))
            conn.commit()
            conn.close()
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error deleting data from {table_name}: {e}")
            if conn:
                conn.close()
            return False

    def transaction(self):
        """Context manager for database transactions.
        
        Returns:
            Transaction context manager
        """
        return TransactionContext(self)

    def query_to_dataframe(self, query: str, params: Tuple = ()) -> pd.DataFrame:
        """Execute a SQL query and return results as a pandas DataFrame.
        
        Args:
            query: SQL query string
            params: Tuple of parameters for the query
            
        Returns:
            pandas DataFrame containing query results
        """
        try:
            conn = self.create_connection()
            if not conn:
                return pd.DataFrame()
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            return df
            
        except sqlite3.Error as e:
            logger.error(f"Error executing query to DataFrame: {e}")
            if conn:
                conn.close()
            return pd.DataFrame()

    def drop_table(self, table_name: str) -> bool:
        """Drop a table from the database.
        
        Args:
            table_name: Name of the table to drop
            
        Returns:
            bool: True if table was dropped successfully
        """
        conn = self.connection if self.connection else self.create_connection()
        try:
            if not conn:
                return False
            cursor = conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.commit()
            # Only close if we created a new connection
            if not self.connection:
                conn.close()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error dropping table {table_name}: {e}")
            if not self.connection and conn:
                conn.close()
            return False

    def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the database.
        
        Args:
            backup_path: Path where to save the backup
            
        Returns:
            bool: True if backup was successful
        """
        try:
            if not self.db_path.exists():
                logger.error(f"Source database {self.db_path} does not exist")
                return False
            
            # Create backup directory if it doesn't exist
            backup_path = Path(backup_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the database file
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error backing up database: {e}")
            return False

    def restore_database(self, backup_path: str) -> bool:
        """Restore database from a backup.
        
        Args:
            backup_path: Path to the backup file
            
        Returns:
            bool: True if restore was successful
        """
        try:
            backup_path = Path(backup_path)
            if not backup_path.exists():
                logger.error(f"Backup file {backup_path} does not exist")
                return False
            
            # Close existing connection
            if self.connection:
                self.connection.close()
                self.connection = None
            
            # Copy backup to current database location
            shutil.copy2(backup_path, self.db_path)
            logger.info(f"Database restored from {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring database: {e}")
            return False

    def delete_all(self, table_name: str) -> bool:
        """Delete all rows from a table.
        
        Args:
            table_name: Name of the table to delete from
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            conn = self.create_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {table_name}")
            conn.commit()
            conn.close()
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error deleting all data from {table_name}: {e}")
            if conn:
                conn.close()
            return False

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get the schema of a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of dictionaries containing column information
        """
        try:
            conn = self.create_connection()
            if not conn:
                return []
            
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            schema = cursor.fetchall()
            conn.close()
            
            # Convert to list of dictionaries
            result = []
            for row in schema:
                result.append({
                    'cid': row[0],
                    'name': row[1],
                    'type': row[2],
                    'notnull': row[3],
                    'dflt_value': row[4],
                    'pk': row[5]
                })
            
            return result
            
        except sqlite3.Error as e:
            logger.error(f"Error getting schema for table {table_name}: {e}")
            if conn:
                conn.close()
            return []

    def is_connected(self) -> bool:
        """Check if the database is connected."""
        try:
            if self.connection is None:
                return False
            # Test the connection with a simple query
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            return True
        except Exception:
            return False

    def get_tables(self) -> List[str]:
        """Get list of all tables in the database."""
        try:
            conn = self.create_connection()
            if not conn:
                return []
            
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            return tables
            
        except sqlite3.Error as e:
            logger.error(f"Error getting tables: {e}")
            return []

    def insert_row(self, table_name: str, data: Dict[str, Any]) -> bool:
        """Insert a single row into a table."""
        try:
            cursor = self.connection.cursor()
            columns = list(data.keys())
            placeholders = ','.join(['?' for _ in columns])
            values = list(data.values())
            
            query = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})"
            cursor.execute(query, values)
            self.connection.commit()
            return True
        except sqlite3.IntegrityError as e:
            logger.error(f"Integrity error inserting row into {table_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inserting row into {table_name}: {e}")
            return False

    def update_row(self, table_name: str, data: Dict[str, Any], where_conditions: Dict[str, Any]) -> bool:
        """Update a single row in a table."""
        try:
            cursor = self.connection.cursor()
            set_clause = ','.join([f"{k}=?" for k in data.keys()])
            where_clause = ' AND '.join([f"{k}=?" for k in where_conditions.keys()])
            
            query = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
            values = list(data.values()) + list(where_conditions.values())
            
            cursor.execute(query, values)
            self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating row in {table_name}: {e}")
            return False

    def update_many(self, table_name: str, data: Dict[str, Any], where_conditions: Dict[str, Any]) -> bool:
        """Update multiple rows in a table."""
        try:
            cursor = self.connection.cursor()
            set_clause = ','.join([f"{k}=?" for k in data.keys()])
            where_clause = ' AND '.join([f"{k}=?" for k in where_conditions.keys()])
            
            query = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
            values = list(data.values()) + list(where_conditions.values())
            
            cursor.execute(query, values)
            self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating rows in {table_name}: {e}")
            return False

    def delete_row(self, table_name: str, where_conditions: Dict[str, Any]) -> bool:
        """Delete a single row from a table."""
        try:
            cursor = self.connection.cursor()
            where_clause = ' AND '.join([f"{k}=?" for k in where_conditions.keys()])
            
            query = f"DELETE FROM {table_name} WHERE {where_clause}"
            values = list(where_conditions.values())
            
            cursor.execute(query, values)
            self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Error deleting row from {table_name}: {e}")
            return False

    def delete_many(self, table_name: str, where_conditions: Dict[str, Any]) -> bool:
        """Delete multiple rows from a table."""
        try:
            cursor = self.connection.cursor()
            where_clause = ' AND '.join([f"{k}=?" for k in where_conditions.keys()])
            
            query = f"DELETE FROM {table_name} WHERE {where_clause}"
            values = list(where_conditions.values())
            
            cursor.execute(query, values)
            self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Error deleting rows from {table_name}: {e}")
            return False

    def add_column(self, table_name: str, column_name: str, column_type: str) -> bool:
        """Add a column to an existing table.
        
        Args:
            table_name: Name of the table
            column_name: Name of the column to add
            column_type: SQLite data type for the column
            
        Returns:
            bool: True if column was added successfully
        """
        try:
            conn = self.create_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
            conn.commit()
            conn.close()
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error adding column {column_name} to {table_name}: {e}")
            if conn:
                conn.close()
            return False
    
    def drop_column(self, table_name: str, column_name: str) -> bool:
        """Drop a column from a table.
        
        Note: SQLite doesn't support DROP COLUMN directly, so this is a workaround.
        
        Args:
            table_name: Name of the table
            column_name: Name of the column to drop
            
        Returns:
            bool: True if column was dropped successfully
        """
        try:
            conn = self.create_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            
            # Get current schema
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            # Create new table without the column
            new_columns = [col[1] for col in columns if col[1] != column_name]
            new_schema = ", ".join([f"{col[1]} {col[2]}" for col in columns if col[1] != column_name])
            
            # Create new table
            cursor.execute(f"CREATE TABLE {table_name}_new ({new_schema})")
            
            # Copy data
            columns_str = ", ".join(new_columns)
            cursor.execute(f"INSERT INTO {table_name}_new ({columns_str}) SELECT {columns_str} FROM {table_name}")
            
            # Drop old table and rename new one
            cursor.execute(f"DROP TABLE {table_name}")
            cursor.execute(f"ALTER TABLE {table_name}_new RENAME TO {table_name}")
            
            conn.commit()
            conn.close()
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error dropping column {column_name} from {table_name}: {e}")
            if conn:
                conn.close()
            return False
    
    def create_index(self, table_name: str, index_name: str, columns: List[str]) -> bool:
        """Create an index on a table.
        
        Args:
            table_name: Name of the table
            index_name: Name of the index
            columns: List of column names to index
            
        Returns:
            bool: True if index was created successfully
        """
        try:
            conn = self.create_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            columns_str = ", ".join(columns)
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({columns_str})")
            conn.commit()
            conn.close()
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error creating index {index_name} on {table_name}: {e}")
            if conn:
                conn.close()
            return False

    def get_indexes(self, table_name: str) -> List[str]:
        """Get list of indexes for a table."""
        conn = None
        try:
            conn = self.create_connection()
            if not conn:
                return []
            
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND tbl_name=?
            """, (table_name,))
            
            indexes = [row[0] for row in cursor.fetchall()]
            return indexes
            
        except sqlite3.Error as e:
            logger.error(f"Error getting indexes for table {table_name}: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def drop_index(self, index_name: str) -> bool:
        """Drop an index by name."""
        conn = None
        try:
            conn = self.create_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            # Use string formatting for index name since SQLite doesn't support parameterized index names
            cursor.execute(f"DROP INDEX IF EXISTS {index_name}")
            conn.commit()
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error dropping index {index_name}: {e}")
            return False
        finally:
            if conn:
                conn.close()

class TransactionContext:
    """Context manager for database transactions."""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.conn = None
        
    def __enter__(self):
        self.conn = self.db_manager.create_connection()
        if self.conn:
            self.conn.execute("BEGIN TRANSACTION")
            # Set the transaction connection on the manager
            self.db_manager._transaction_conn = self.conn
        return self.db_manager
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type is not None:
                # Rollback on exception
                self.conn.rollback()
                logger.error(f"Transaction rolled back due to exception: {exc_val}")
            else:
                # Commit on success
                self.conn.commit()
            self.conn.close()
            # Clear the transaction connection from the manager
            if hasattr(self.db_manager, '_transaction_conn'):
                delattr(self.db_manager, '_transaction_conn')
            # Clear the connection from the manager to prevent reuse
            self.db_manager.connection = None

# Helper functions for tests
def _connect(db_path: str) -> sqlite3.Connection:
    """Create a connection to the database."""
    return create_connection(db_path)

def _close(conn: sqlite3.Connection) -> None:
    """Close a database connection."""
    if conn:
        conn.close()

def _execute(conn: sqlite3.Connection, query: str, params: tuple = ()) -> None:
    """Execute a SQL query."""
    cursor = conn.cursor()
    cursor.execute(query, params)

def _commit(conn: sqlite3.Connection) -> None:
    """Commit a transaction."""
    conn.commit()

def _fetchone(conn: sqlite3.Connection, query: str, params: tuple = ()) -> tuple:
    """Fetch one row from a query."""
    cursor = conn.cursor()
    cursor.execute(query, params)
    return cursor.fetchone()

def _fetchall(conn: sqlite3.Connection, query: str, params: tuple = ()) -> list:
    """Fetch all rows from a query."""
    cursor = conn.cursor()
    cursor.execute(query, params)
    return cursor.fetchall()

def _create_table(conn: sqlite3.Connection, table_name: str, schema: str) -> None:
    """Create a table."""
    cursor = conn.cursor()
    cursor.execute(f"CREATE TABLE {table_name} ({schema})")

def _drop_table(conn: sqlite3.Connection, table_name: str) -> None:
    """Drop a table."""
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE {table_name}")