import sqlite3
import logging
import json
import math
from typing import Dict, List, Optional, Any, Tuple, Set
import pandas as pd
import numpy as np
from pathlib import Path
import utils

logger = logging.getLogger(__name__)

# Constants
SQLITE_MAX_VARIABLE_NUMBER = 999  # SQLite's default limit

class SQLiteManager:
    """Class-based implementation of SQLite database management."""
    
    def __init__(self, db_path: str):
        """Initialize with database path."""
        self.db_path = db_path
        self.connection = None
        self._connect()
        
    def _connect(self) -> None:
        """Establish database connection with proper settings."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            # Enable foreign keys
            self.connection.execute("PRAGMA foreign_keys=ON;")
            # Set journal mode to WAL for better concurrency
            self.connection.execute("PRAGMA journal_mode=WAL;")
            # Set synchronous mode for better performance
            self.connection.execute("PRAGMA synchronous=NORMAL;")
            # Set temp store to memory for better performance
            self.connection.execute("PRAGMA temp_store=MEMORY;")
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database {self.db_path}: {e}", exc_info=True)
            raise
            
    def initialize_database(self) -> bool:
        """Initialize or verify the database schema."""
        if not self.connection:
            logger.error("No database connection")
            return False
            
        try:
            cursor = self.connection.cursor()
            # Use DEFERRED transaction for initialization
            self.connection.execute("BEGIN DEFERRED;")
            
            # --- Metadata Tables ---
            cursor.execute("CREATE TABLE IF NOT EXISTS symbols (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT UNIQUE NOT NULL);")
            cursor.execute("CREATE TABLE IF NOT EXISTS timeframes (id INTEGER PRIMARY KEY AUTOINCREMENT, timeframe TEXT UNIQUE NOT NULL);")
            cursor.execute("CREATE TABLE IF NOT EXISTS indicators (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL);")
            
            # --- Indicator Configuration Table ---
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS indicator_configs (
                config_id INTEGER PRIMARY KEY AUTOINCREMENT,
                indicator_id INTEGER NOT NULL,
                config_hash TEXT NOT NULL,
                config_json TEXT NOT NULL,
                FOREIGN KEY (indicator_id) REFERENCES indicators(id) ON DELETE CASCADE,
                UNIQUE (indicator_id, config_hash)
            );""")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicator_configs_indicator_id ON indicator_configs(indicator_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicator_configs_hash ON indicator_configs(config_hash);")
            
            # --- Correlation Results Table ---
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS correlations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol_id INTEGER NOT NULL,
                timeframe_id INTEGER NOT NULL,
                indicator_config_id INTEGER NOT NULL,
                lag INTEGER NOT NULL,
                correlation_value REAL,
                FOREIGN KEY (symbol_id) REFERENCES symbols(id) ON DELETE CASCADE,
                FOREIGN KEY (timeframe_id) REFERENCES timeframes(id) ON DELETE CASCADE,
                FOREIGN KEY (indicator_config_id) REFERENCES indicator_configs(config_id) ON DELETE CASCADE,
                UNIQUE(symbol_id, timeframe_id, indicator_config_id, lag)
            );""")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_correlations_main ON correlations (symbol_id, timeframe_id, indicator_config_id, lag);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_correlations_config_lag ON correlations (indicator_config_id, lag);")
            
            self.connection.commit()
            logger.info(f"Database schema initialized/verified: {self.db_path}")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error operating on DB schema '{self.db_path}': {e}", exc_info=True)
            try:
                self.connection.rollback()
            except Exception as rb_err:
                logger.error(f"Rollback failed during schema init: {rb_err}")
            return False
            
    def _get_or_create_id(self, table: str, name_col: str, value: str) -> int:
        """Get or create ID for a value in a metadata table."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(f"SELECT id FROM {table} WHERE {name_col} = ?", (value,))
            result = cursor.fetchone()
            if result:
                return result[0]
                
            cursor.execute(f"INSERT INTO {table} ({name_col}) VALUES (?)", (value,))
            self.connection.commit()
            return cursor.lastrowid
            
        except sqlite3.Error as e:
            logger.error(f"Error in _get_or_create_id for {table}.{name_col}={value}: {e}", exc_info=True)
            try:
                self.connection.rollback()
            except Exception as rb_err:
                logger.error(f"Rollback failed in _get_or_create_id: {rb_err}")
            raise
            
    def get_or_create_indicator_config_id(self, indicator_name: str, params: Dict[str, Any]) -> int:
        """Get or create ID for an indicator configuration."""
        cursor = self.connection.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE;")
            indicator_id = self._get_or_create_id('indicators', 'name', indicator_name)
            config_hash = utils.get_config_hash(params)
            config_str = json.dumps(params, sort_keys=True, separators=(',', ':'))
            
            cursor.execute("SELECT config_id, config_json FROM indicator_configs WHERE indicator_id = ? AND config_hash = ?",
                         (indicator_id, config_hash))
            result = cursor.fetchone()
            
            if result:
                config_id_found, existing_json = result
                if existing_json == config_str:
                    self.connection.commit()
                    return config_id_found
                else:
                    logger.error(f"HASH COLLISION/DATA MISMATCH: ID {indicator_id}, hash {config_hash}")
                    self.connection.rollback()
                    raise ValueError("Hash collision or data mismatch detected")
                    
            cursor.execute("INSERT INTO indicator_configs (indicator_id, config_hash, config_json) VALUES (?, ?, ?)",
                         (indicator_id, config_hash, config_str))
            new_config_id = cursor.lastrowid
            
            if new_config_id is None:
                cursor.execute("SELECT config_id FROM indicator_configs WHERE indicator_id = ? AND config_hash = ?",
                             (indicator_id, config_hash))
                result = cursor.fetchone()
                if result:
                    new_config_id = result[0]
                else:
                    raise sqlite3.Error("INSERT successful but could not retrieve new config_id")
                    
            self.connection.commit()
            return new_config_id
            
        except sqlite3.IntegrityError:
            self.connection.rollback()
            cursor.execute("SELECT config_id, config_json FROM indicator_configs WHERE indicator_id = ? AND config_hash = ?",
                         (indicator_id, config_hash))
            result = cursor.fetchone()
            if result:
                config_id_found, existing_json = result
                if existing_json == config_str:
                    return config_id_found
                else:
                    logger.critical(f"CRITICAL: IntegrityError AND JSON mismatch after re-query")
                    raise ValueError("Data inconsistency after IntegrityError")
            else:
                logger.critical(f"CRITICAL: Cannot find config ID after IntegrityError")
                raise ValueError("Cannot resolve config ID after IntegrityError")
                
        except Exception as e:
            logger.error(f"Error get/create config ID for '{indicator_name}': {e}", exc_info=True)
            try:
                self.connection.rollback()
            except Exception as rb_err:
                logger.error(f"Rollback failed: {rb_err}")
            raise
            
    def batch_insert_correlations(self, data_to_insert: List[Tuple[int, int, int, int, Optional[float]]]) -> bool:
        """Insert multiple correlation values in a single transaction."""
        if not data_to_insert:
            logger.info("No correlation data provided for batch insert")
            return True
            
        cursor = self.connection.cursor()
        try:
            prepared_data = []
            for s_id, t_id, cfg_id, lag, corr_val in data_to_insert:
                db_value = float(corr_val) if pd.notna(corr_val) and isinstance(corr_val, (int, float, np.number)) else None
                prepared_data.append((s_id, t_id, cfg_id, lag, db_value))
                
            cursor.execute("BEGIN IMMEDIATE;")
            cursor.executemany("""
                INSERT OR REPLACE INTO correlations 
                (symbol_id, timeframe_id, indicator_config_id, lag, correlation_value) 
                VALUES (?, ?, ?, ?, ?)
            """, prepared_data)
            self.connection.commit()
            logger.info(f"Batch inserted/replaced {len(prepared_data)} correlations")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"DB error during batch correlation insert: {e}", exc_info=True)
            try:
                self.connection.rollback()
            except Exception as rb_err:
                logger.error(f"Rollback failed: {rb_err}")
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error during batch insert: {e}", exc_info=True)
            try:
                self.connection.rollback()
            except Exception as rb_err:
                logger.error(f"Rollback failed: {rb_err}")
            return False
            
    def fetch_correlations(self, symbol_id: int, timeframe_id: int, config_ids: List[int]) -> Dict[int, List[Optional[float]]]:
        """Fetch correlations for multiple config IDs in batches."""
        if not config_ids:
            return {}
            
        batch_size = SQLITE_MAX_VARIABLE_NUMBER
        num_batches = math.ceil(len(config_ids) / batch_size)
        logger.info(f"Fetching correlations in {num_batches} batches")
        
        all_rows = []
        cursor = self.connection.cursor()
        
        try:
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                batch_ids = config_ids[start_idx:end_idx]
                if not batch_ids:
                    continue
                    
                placeholders = ','.join('?' * len(batch_ids))
                query = f"""
                    SELECT indicator_config_id, lag, correlation_value
                    FROM correlations
                    WHERE symbol_id = ? AND timeframe_id = ? AND indicator_config_id IN ({placeholders})
                    ORDER BY indicator_config_id, lag;
                """
                params = [symbol_id, timeframe_id] + batch_ids
                cursor.execute(query, params)
                batch_rows = cursor.fetchall()
                all_rows.extend(batch_rows)
                
            if not all_rows:
                return {cfg_id: [] for cfg_id in config_ids}
                
            max_lag = max((row[1] for row in all_rows if isinstance(row[1], int)), default=0)
            if max_lag <= 0:
                return {cfg_id: [] for cfg_id in config_ids}
                
            # Organize results by config_id
            correlations = {cfg_id: [None] * (max_lag + 1) for cfg_id in config_ids}
            for cfg_id, lag, corr_val in all_rows:
                if cfg_id in correlations and 0 <= lag <= max_lag:
                    correlations[cfg_id][lag] = corr_val
                    
            return correlations
            
        except sqlite3.Error as e:
            logger.error(f"SQLite error fetching correlations: {e}", exc_info=True)
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching correlations: {e}", exc_info=True)
            return {}
            
    def get_distinct_config_ids_for_pair(self, symbol_id: int, timeframe_id: int) -> List[int]:
        """Get distinct config IDs with correlation data for a symbol/timeframe pair."""
        cursor = self.connection.cursor()
        try:
            cursor.execute("""
                SELECT DISTINCT indicator_config_id 
                FROM correlations 
                WHERE symbol_id = ? AND timeframe_id = ?
            """, (symbol_id, timeframe_id))
            rows = cursor.fetchall()
            config_ids = [row[0] for row in rows if row[0] is not None and isinstance(row[0], int)]
            logger.info(f"Found {len(config_ids)} distinct config IDs")
            return config_ids
        except sqlite3.Error as e:
            logger.error(f"Error getting distinct config IDs: {e}", exc_info=True)
            return []
            
    def get_indicator_configs_by_ids(self, config_ids: List[int]) -> List[Dict[str, Any]]:
        """Fetch indicator configuration details for multiple config IDs."""
        if not config_ids:
            return []
            
        batch_size = SQLITE_MAX_VARIABLE_NUMBER
        num_batches = math.ceil(len(config_ids) / batch_size)
        logger.info(f"Fetching {len(config_ids)} config details in {num_batches} batches")
        
        all_rows = []
        cursor = self.connection.cursor()
        
        try:
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                batch_ids = config_ids[start_idx:end_idx]
                if not batch_ids:
                    continue
                    
                placeholders = ','.join('?' * len(batch_ids))
                query = f"""
                    SELECT ic.config_id, i.name AS indicator_name, ic.config_json
                    FROM indicator_configs ic
                    JOIN indicators i ON ic.indicator_id = i.id
                    WHERE ic.config_id IN ({placeholders});
                """
                cursor.execute(query, batch_ids)
                batch_rows = cursor.fetchall()
                all_rows.extend(batch_rows)
                
            configs_processed = []
            for cfg_id, name, json_str in all_rows:
                try:
                    params = json.loads(json_str)
                    configs_processed.append({
                        'config_id': cfg_id,
                        'indicator_name': name,
                        'params': params
                    })
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode JSON for config_id {cfg_id}")
                except Exception as e:
                    logger.error(f"Error processing config {cfg_id}: {e}")
                    
            return configs_processed
            
        except sqlite3.Error as e:
            logger.error(f"SQLite error fetching config details: {e}", exc_info=True)
            return []
            
        except Exception as e:
            logger.error(f"Error fetching config details: {e}", exc_info=True)
            return []
            
    def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the database."""
        try:
            if not self.connection:
                raise ValueError("No active database connection")
                
            # Ensure the backup directory exists
            backup_dir = Path(backup_path).parent
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create backup connection
            backup_conn = sqlite3.connect(backup_path)
            try:
                # Copy the database
                self.connection.backup(backup_conn)
                backup_conn.close()
                logger.info(f"Database backup created: {backup_path}")
                return True
            except Exception as e:
                logger.error(f"Error during backup: {e}", exc_info=True)
                try:
                    backup_conn.close()
                except Exception:
                    pass
                return False
                
        except Exception as e:
            logger.error(f"Error preparing backup: {e}", exc_info=True)
            return False
            
    def restore_database(self, backup_path: str) -> bool:
        """Restore database from backup."""
        try:
            if not self.connection:
                raise ValueError("No active database connection")
                
            # Verify backup exists
            if not Path(backup_path).exists():
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
                
            # Create backup connection
            backup_conn = sqlite3.connect(backup_path)
            try:
                # Restore from backup
                backup_conn.backup(self.connection)
                backup_conn.close()
                logger.info(f"Database restored from: {backup_path}")
                return True
            except Exception as e:
                logger.error(f"Error during restore: {e}", exc_info=True)
                try:
                    backup_conn.close()
                except Exception:
                    pass
                return False
                
        except Exception as e:
            logger.error(f"Error preparing restore: {e}", exc_info=True)
            return False
            
    def close(self) -> None:
        """Close the database connection."""
        if self.connection:
            try:
                self.connection.close()
                self.connection = None
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing connection: {e}", exc_info=True)
                
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def create_table(self, table_name: str, columns: Dict[str, str]) -> None:
        """Create a table with the given name and columns."""
        if not table_name or not isinstance(table_name, str) or not columns or not isinstance(columns, dict):
            raise ValueError("Invalid table name or columns for create_table.")
        columns_def = ', '.join([f'{col} {col_type}' for col, col_type in columns.items()])
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_def});"
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql)
            self.connection.commit()
        except sqlite3.Error as e:
            logger.error(f"Error creating table {table_name}: {e}", exc_info=True)
            try:
                self.connection.rollback()
            except Exception as rb_err:
                logger.error(f"Rollback failed during create_table: {rb_err}")
            raise

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        if not table_name or not isinstance(table_name, str):
            return False
        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
        return cursor.fetchone() is not None

    def insert_data(self, table_name: str, data: Dict[str, Any]) -> None:
        """Insert a single row into the specified table."""
        if not table_name or not isinstance(table_name, str) or not data or not isinstance(data, dict):
            raise ValueError("Invalid table name or data for insert_data.")
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        values = tuple(data.values())
        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders});"
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql, values)
            self.connection.commit()
        except sqlite3.Error as e:
            logger.error(f"Error inserting data into {table_name}: {e}", exc_info=True)
            try:
                self.connection.rollback()
            except Exception as rb_err:
                logger.error(f"Rollback failed during insert_data: {rb_err}")
            raise

    def insert_many(self, table_name: str, data_list: List[Dict[str, Any]]) -> None:
        """Insert multiple rows into the specified table."""
        if not table_name or not isinstance(table_name, str) or not data_list or not isinstance(data_list, list):
            raise ValueError("Invalid table name or data_list for insert_many.")
        if not all(isinstance(d, dict) for d in data_list):
            raise ValueError("All items in data_list must be dicts.")
        columns = ', '.join(data_list[0].keys())
        placeholders = ', '.join(['?' for _ in data_list[0]])
        values = [tuple(d.values()) for d in data_list]
        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders});"
        try:
            cursor = self.connection.cursor()
            cursor.executemany(sql, values)
            self.connection.commit()
        except sqlite3.Error as e:
            logger.error(f"Error inserting many rows into {table_name}: {e}", exc_info=True)
            try:
                self.connection.rollback()
            except Exception as rb_err:
                logger.error(f"Rollback failed during insert_many: {rb_err}")
            raise

    def query(self, sql: str, params: Optional[Tuple[Any, ...]] = None) -> List[Tuple[Any, ...]]:
        """Execute a SQL query and return the results as a list of tuples."""
        cursor = self.connection.cursor()
        try:
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Error executing query: {e}", exc_info=True)
            raise

    def query_to_dataframe(self, sql: str, params: Optional[Tuple[Any, ...]] = None) -> pd.DataFrame:
        """Execute a SQL query and return the results as a pandas DataFrame."""
        cursor = self.connection.cursor()
        try:
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return pd.DataFrame(rows, columns=columns)
        except sqlite3.Error as e:
            logger.error(f"Error executing query_to_dataframe: {e}", exc_info=True)
            raise
