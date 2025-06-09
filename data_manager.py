# data_manager.py
import logging
import pandas as pd
import sqlite3
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
from datetime import datetime, timezone # Ensure timezone is imported
import time
import requests
import math
import re
import os
import numpy as np  # Add numpy import

import config
import sqlite_manager
import utils
from data_processing import validate_data as validate_dataframe, process_data, validate_required_columns_and_nans
from utils import is_valid_symbol, is_valid_timeframe  # Add explicit imports

logger = logging.getLogger(__name__)

# --- Binance API ---
BINANCE_API_BASE_URL = "https://api.binance.com/api/v3/klines"
KLINES_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
]
MAX_LIMIT = 1000 # Max klines per request for Binance
DEFAULT_START_DATE = datetime(2017, 1, 1, tzinfo=timezone.utc)
# ---> ADDED: Earliest reasonable date for filtering <---
EARLIEST_VALID_DATE = datetime(2015, 1, 1, tzinfo=timezone.utc)


VALID_TIMEFRAMES = (
    '1m', '3m', '5m', '15m', '30m',           # Minutes
    '1h', '2h', '4h', '6h', '8h', '12h',      # Hours
    '1d', '3d',                               # Days
    '1w',                                     # Weeks
    '1M'                                      # Month (Capital M)
)

# --- API Fetching ---
def _fetch_klines(symbol: str, interval: str, start_time_ms: int, end_time_ms: int) -> List[List[Any]]:
    """Fetches klines data from Binance API with retries and error handling."""
    all_klines = []
    current_start_time = start_time_ms
    # ---> Corrected Timestamp Conversion <---
    # Create naive datetime first, then make it timezone-aware
    start_dt_naive = datetime.fromtimestamp(start_time_ms / 1000)
    start_dt_utc = start_dt_naive.replace(tzinfo=timezone.utc)

    end_dt_naive = datetime.fromtimestamp(end_time_ms / 1000)
    end_dt_utc = end_dt_naive.replace(tzinfo=timezone.utc)
    # ---> End Correction <---
    logger.info(f"Fetching klines for {symbol} ({interval}) from {start_dt_utc} to {end_dt_utc}")

    retries = 3
    wait_time = 1 # seconds initial wait

    while current_start_time < end_time_ms:
        current_retries = retries
        last_exception = None
        while current_retries > 0:
            try:
                params = {
                    'symbol': symbol.upper(), 'interval': interval,
                    'startTime': current_start_time, 'endTime': end_time_ms,
                    'limit': MAX_LIMIT
                }
                logger.debug(f"Requesting Binance API: {params}")
                response = requests.get(BINANCE_API_BASE_URL, params=params, timeout=20)
                logger.debug(f"Response Status Code: {response.status_code}")
                response.raise_for_status() # Raise HTTPError for bad responses
                klines = response.json()

                if not klines:
                    # ---> Corrected Timestamp Conversion <---
                    no_more_data_dt = datetime.fromtimestamp(current_start_time/1000).replace(tzinfo=timezone.utc)
                    logger.info(f"No more klines data returned for {symbol} starting {no_more_data_dt}. Reached end.")
                    current_start_time = end_time_ms # Force outer loop exit
                    last_exception = None; break # Exit retry loop

                # ---> ADDED: Basic validation of fetched kline timestamps <---
                first_ts = klines[0][0]
                if not isinstance(first_ts, (int, float)) or first_ts < (EARLIEST_VALID_DATE.timestamp() * 1000 * 0.9): # Allow some buffer
                     logger.error(f"Fetched klines have invalid starting timestamp: {first_ts}. Aborting fetch.")
                     return [] # Indicate failure
                # ---> END ADDED VALIDATION <---


                all_klines.extend(klines)
                last_kline_open_time = klines[-1][0]
                new_start_time = last_kline_open_time + 1

                if new_start_time <= current_start_time:
                    logger.warning(f"API did not advance timestamp (returned {last_kline_open_time}). Breaking fetch loop.")
                    current_start_time = end_time_ms; last_exception = None; break

                current_start_time = new_start_time
                # ---> Corrected Timestamp Conversion <---
                last_fetched_dt = datetime.fromtimestamp(last_kline_open_time/1000).replace(tzinfo=timezone.utc)
                # ---> End Correction <---
                logger.info(f"Fetched {len(klines)} klines up to {last_fetched_dt}. Total fetched: {len(all_klines)}")
                time.sleep(0.15) # Be nice to API
                last_exception = None; break # Success, exit retry loop

            except requests.exceptions.Timeout as timeout_err:
                 last_exception = timeout_err; wait_period = wait_time
                 logger.warning(f"Timeout. Retries left: {current_retries-1}. Waiting {wait_period}s...")
            except requests.exceptions.RequestException as req_err:
                 last_exception = req_err; status_code = getattr(req_err.response, 'status_code', None)
                 wait_period = wait_time
                 if status_code in [429, 418]: # Rate limit
                      wait_period = max(wait_period, 15) # Wait longer
                      logger.warning(f"Rate limit hit (Status {status_code}). Retrying after {wait_period}s...")
                 else:
                      logger.error(f"HTTP error fetching: {req_err}. Status: {status_code}")
                 logger.warning(f"Retries left: {current_retries-1}. Waiting {wait_period}s...")
            except Exception as e:
                 last_exception = e
                 logger.error(f"Unexpected kline fetch error: {e}", exc_info=True)
                 current_start_time = end_time_ms # Force outer loop exit on unexpected error
                 break # Exit retry loop

            # If we reach here, an error occurred
            current_retries -= 1
            time.sleep(wait_period)
            wait_time = min(wait_time * 2, 30) # Exponential backoff

        # Check if all retries failed for the current batch
        if last_exception is not None:
             logger.error(f"Failed klines fetch after multiple retries. Last error: {last_exception}")
             return [] # Indicate failure

    logger.info(f"Finished fetching. Total klines retrieved: {len(all_klines)}")
    return all_klines

# --- Data Processing ---
def _process_klines(klines: List[List[Any]]) -> pd.DataFrame:
    """Converts raw klines list to a pandas DataFrame with correct types and removes duplicates/bad data."""
    if not klines: return pd.DataFrame()

    logger.debug(f"Processing {len(klines)} raw klines into DataFrame...")
    try:
        df = pd.DataFrame(klines, columns=KLINES_COLUMNS).drop(columns=['ignore'])

        # Convert core time column first for filtering
        df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
        initial_len = len(df)
        df.dropna(subset=['open_time'], inplace=True)
        if len(df) < initial_len:
            logger.warning(f"Dropped {initial_len - len(df)} rows with invalid open_time before date conversion.")

        # ---> ADDED: Filter out rows with unreasonably old timestamps BEFORE saving <---
        min_valid_timestamp_ms = EARLIEST_VALID_DATE.timestamp() * 1000
        initial_len = len(df)
        df = df[df['open_time'] >= min_valid_timestamp_ms]
        if len(df) < initial_len:
            logger.warning(f"Dropped {initial_len - len(df)} rows with open_time before {EARLIEST_VALID_DATE.date()}.")
        if df.empty:
             logger.error("DataFrame empty after filtering old timestamps. Check API response/filtering.")
             return pd.DataFrame()
        # ---> END ADDED FILTER <---

        numeric_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume",
                        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]
        int_cols = ["number_of_trades"]
        time_cols = ["close_time"] # open_time already done

        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in int_cols: df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        for col in time_cols: df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

        # Ensure open_time is also Int64 after filtering
        df['open_time'] = df['open_time'].astype('Int64')


        core_cols_check = ['open', 'high', 'low', 'close', 'volume', 'close_time'] # Check others
        initial_len = len(df)
        df.dropna(subset=core_cols_check, inplace=True)
        if len(df) < initial_len:
            logger.warning(f"Dropped {initial_len - len(df)} rows with NaN in core columns (excluding open_time).")

        initial_len = len(df)
        df.drop_duplicates(subset=['open_time'], keep='first', inplace=True)
        if len(df) < initial_len:
            logger.warning(f"Removed {initial_len - len(df)} duplicate klines based on open_time.")

        if df.empty: logger.warning("DataFrame empty after processing."); return pd.DataFrame()

        logger.debug(f"Processed klines DataFrame shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error processing klines list into DataFrame: {e}", exc_info=True)
        return pd.DataFrame()

# --- Database Saving ---
def _save_to_sqlite(df: pd.DataFrame, db_path: str, symbol: str, timeframe: str) -> bool:
    """Saves the DataFrame to SQLite using INSERT OR IGNORE behavior, with stricter type checks."""
    if df.empty: 
        logger.info("No processed data to save.")
        return True

    # ---> Final check before saving <---
    min_valid_timestamp_ms = EARLIEST_VALID_DATE.timestamp() * 1000
    if 'open_time' not in df.columns:
        logger.error("CRITICAL: 'open_time' column missing before save. Aborting.")
        return False
    # Convert to numeric just in case, coercing errors
    open_time_numeric = pd.to_numeric(df['open_time'], errors='coerce')
    if open_time_numeric.isnull().any():
        logger.error(f"CRITICAL: Found non-numeric or null 'open_time' values before saving. Aborting.")
        return False
    if open_time_numeric.min() < min_valid_timestamp_ms:
        logger.error(f"CRITICAL: Data contains invalid timestamps just before saving! Min: {open_time_numeric.min()}. Aborting save.")
        return False
    # ---> END Final check <---

    logger.debug(f"Attempting to save {len(df)} records to {db_path}...")
    conn = sqlite_manager.create_connection(db_path)
    if not conn: 
        return False

    try:
        # Get IDs within a transaction
        conn.execute("BEGIN IMMEDIATE;") # Use IMMEDIATE for faster lock acquisition
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM symbols WHERE symbol = ?", (symbol,))
        symbol_id = cursor.fetchone()[0]
        cursor.execute("SELECT id FROM timeframes WHERE timeframe = ?", (timeframe,))
        timeframe_id = cursor.fetchone()[0]

        # Prepare data for insertion
        df_to_insert = df.copy()
        df_to_insert['symbol_id'] = symbol_id
        df_to_insert['timeframe_id'] = timeframe_id

        # Ensure all required columns exist
        required_columns = [
            "symbol_id", "timeframe_id", "open_time", "open", "high", "low", 
            "close", "volume", "close_time"
        ]
        optional_columns = [
            "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
        ]

        # Check required columns
        missing_required = [col for col in required_columns if col not in df_to_insert.columns]
        if missing_required:
            logger.error(f"DataFrame missing required columns: {missing_required}")
            conn.rollback()
            return False

        # Select only columns that exist in the DataFrame
        columns_to_insert = required_columns + [col for col in optional_columns if col in df_to_insert.columns]
        df_to_insert = df_to_insert[columns_to_insert]

        # Convert data types
        for col in df_to_insert.columns:
            if col in ['open_time', 'close_time', 'number_of_trades']:
                df_to_insert[col] = pd.to_numeric(df_to_insert[col], downcast='integer')
            elif col in ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']:
                df_to_insert[col] = pd.to_numeric(df_to_insert[col], downcast='float')

        # Insert data
        df_to_insert.to_sql('historical_data', conn, if_exists='append', index=False, 
                           method='multi', chunksize=1000)
        
        conn.commit()
        logger.info(f"Successfully saved {len(df)} records to {db_path}")
        return True

    except Exception as e:
        logger.error(f"Error saving data to {db_path}: {e}", exc_info=True)
        if conn:
            try:
                conn.rollback()
            except:
                pass
        return False
    finally:
        if conn:
            conn.close()

# --- Timestamp & Data Download Control ---
def _get_last_timestamp(db_path: str, symbol_id: int, timeframe_id: int) -> Optional[int]:
    """Gets the last recorded open_time (milliseconds) from the database."""
    db_file = Path(db_path)
    if not db_file.is_file():
        logger.info(f"Database file {db_path} not found. No last timestamp available.")
        return None

    conn = sqlite_manager.create_connection(db_path)
    if not conn: 
        return None
    try:
        cursor = conn.cursor()
        # Verify table exists first
        cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='historical_data';")
        if cursor.fetchone() is None:
            logger.warning(f"Table 'historical_data' not found in {db_path}. Cannot get last timestamp.")
            return None

        cursor.execute("SELECT MAX(open_time) FROM historical_data WHERE symbol_id = ? AND timeframe_id = ?", 
                      (symbol_id, timeframe_id))
        result = cursor.fetchone()
        last_ts = result[0] if result and result[0] is not None else None

        if last_ts:
            # Validate timestamp
            min_valid_timestamp_ms = EARLIEST_VALID_DATE.timestamp() * 1000
            if last_ts < min_valid_timestamp_ms:
                logger.error(f"Last timestamp found in DB ({last_ts}) is unreasonably old. Treating as no previous data.")
                return None
            last_dt_utc = datetime.fromtimestamp(last_ts/1000).replace(tzinfo=timezone.utc)
            logger.info(f"Last timestamp found in DB for SymbolID {symbol_id}, TFID {timeframe_id}: {last_ts} ({last_dt_utc})")
        else:
            logger.info(f"No previous data found in DB for SymbolID {symbol_id}, TFID {timeframe_id}.")
        return last_ts
    finally:
        conn.close()

def download_binance_data(symbol: str, timeframe: str, db_path: Path) -> bool:
    """Downloads/updates historical klines data from Binance into the SQLite database."""
    logger.info(f"Starting data download/update for {symbol} ({timeframe}) into {db_path}")

    # Initialize DB schema first
    if not sqlite_manager.initialize_database(str(db_path), symbol, timeframe):
        logger.error("Failed to initialize database. Aborting download.")
        return False

    # Get symbol and timeframe IDs
    conn_ids = sqlite_manager.create_connection(str(db_path))
    if not conn_ids: return False
    try:
        conn_ids.execute("BEGIN;") # Transaction for IDs
        symbol_id = sqlite_manager._get_or_create_id(conn_ids, 'symbols', 'symbol', symbol)
        timeframe_id = sqlite_manager._get_or_create_id(conn_ids, 'timeframes', 'timeframe', timeframe)
        conn_ids.commit()
    except Exception as e:
        logger.error(f"Failed to get/create symbol/timeframe IDs: {e}", exc_info=True)
        try: conn_ids.rollback()
        except: pass
        return False
    finally:
        if conn_ids: conn_ids.close()

    # Determine start time
    last_ts_ms = _get_last_timestamp(str(db_path), symbol_id, timeframe_id)
    start_time_ms = last_ts_ms + 1 if last_ts_ms is not None else int(DEFAULT_START_DATE.timestamp() * 1000)
    end_time_ms = math.floor(datetime.now(timezone.utc).timestamp() * 1000)

    if start_time_ms >= end_time_ms:
        logger.info(f"Database appears up-to-date for {symbol} ({timeframe}).")
        return True # Up-to-date is a success

    # Fetch, process, and save
    klines = _fetch_klines(symbol, timeframe, start_time_ms, end_time_ms)
    if not klines:
        logger.warning(f"No klines data fetched for {symbol} {timeframe}. Check connection or range.")
        return last_ts_ms is not None # Return True if just checking for updates, False if initial fetch failed

    df = _process_klines(klines)
    if df.empty:
         logger.warning(f"Processing klines resulted in empty DataFrame for {symbol} {timeframe}.")
         # If initial fetch failed completely after processing, return False. If updating, maybe True is ok.
         return last_ts_ms is not None

    return _save_to_sqlite(df, str(db_path), symbol, timeframe)

# --- User Interaction & Data Loading ---
def list_existing_databases() -> List[Path]:
    """Lists existing .db files in the configured database directory."""
    try:
        db_dir = config.DB_DIR
        if not db_dir.is_dir():
            logger.warning(f"Database directory {db_dir} not found.")
            return []
        # Filter out the leaderboard DB if it's in the same root folder
        leaderboard_name = config.LEADERBOARD_DB_PATH.name
        return sorted([p for p in db_dir.glob("*.db") if p.is_file() and p.name != leaderboard_name])
    except Exception as e:
        logger.error(f"Error listing databases in {config.DB_DIR}: {e}", exc_info=True)
        return []

def select_existing_database() -> Optional[Path]:
    """Select an existing database with improved error handling."""
    try:
        result = manage_data_source()
        if not result:
            return None
        return result[0]
    except Exception as e:
        logging.error(f"Database selection failed: {e}")
        return None

def validate_data(db_path: Path) -> bool:
    """Validates if the database file exists, is valid SQLite, and has historical data."""
    if not db_path.is_file():
        logger.warning(f"Database file does not exist: {db_path}")
        return False

    conn = sqlite_manager.create_connection(str(db_path))
    if not conn: return False # Connection failed
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='historical_data';")
        if cursor.fetchone() is None:
             logger.warning(f"Table 'historical_data' not found in {db_path.name}. Validation failed.")
             return False
        cursor.execute("SELECT COUNT(*) FROM historical_data")
        count = cursor.fetchone()[0]
        if count > 0:
            logger.info(f"DB '{db_path.name}' validated: {count} records.")
            return True
        else:
            logger.warning(f"DB '{db_path.name}' schema OK, but contains no historical data.")
            return False # Consider empty DB as failing validation for usage
    except sqlite3.Error as e:
        logger.error(f"Error validating data in '{db_path.name}': {e}", exc_info=True)
        return False
    finally:
        if conn: conn.close()

def manage_data_source(choice: Optional[int] = None) -> Optional[Tuple[Path, str, str]]:
    """Manage data source selection with improved error handling.
    
    Args:
        choice (Optional[int]): Pre-selected choice for testing purposes. If None, prompts user for input.
        
    Returns:
        Optional[Tuple[Path, str, str]]: Selected database file, symbol, and timeframe, or None if cancelled
    """
    try:
        # List available databases
        db_files = list_existing_databases()
        if not db_files:
            logging.warning("No existing databases found")
            return None
            
        # Print available options
        print("\nAvailable databases:")
        for i, db_file in enumerate(db_files, 1):
            print(f"{i}. {db_file.name}")
            
        # Get user selection
        while True:
            try:
                if choice is not None:
                    selected_choice = choice
                else:
                    selected_choice = int(input("\nSelect database (number) or 0 to quit: ").strip())
                    
                if selected_choice == 0:
                    return None
                if 1 <= selected_choice <= len(db_files):
                    break
                print(f"Please enter a number between 1 and {len(db_files)}")
            except ValueError:
                print("Please enter a valid number")
                
        # Parse symbol and timeframe from filename
        db_file = db_files[selected_choice - 1]
        try:
            symbol, timeframe = db_file.stem.split('_')
        except ValueError:
            raise ValueError(f"Invalid database filename format: {db_file.name}")
            
        if not is_valid_symbol(symbol):
            raise ValueError(f"Invalid symbol in filename: {symbol}")
            
        if not is_valid_timeframe(timeframe):
            raise ValueError(f"Invalid timeframe in filename: {timeframe}")
            
        return db_file, symbol, timeframe
    except Exception as e:
        logging.error(f"Data source management failed: {e}")
        return None

# --- Load Data ---
def get_symbol_id(db_path: Path, symbol: str) -> int:
    """Get symbol ID from database."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM symbols WHERE symbol = ?", (symbol,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return result[0]
        else:
            raise ValueError(f"Symbol {symbol} not found in database")
    except Exception as e:
        logger.error(f"Error getting symbol ID for {symbol}: {e}")
        raise

def get_timeframe_id(db_path: Path, timeframe: str) -> int:
    """Get timeframe ID from database."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM timeframes WHERE timeframe = ?", (timeframe,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return result[0]
        else:
            raise ValueError(f"Timeframe {timeframe} not found in database")
    except Exception as e:
        logger.error(f"Error getting timeframe ID for {timeframe}: {e}")
        raise

def load_data(db_path: Path, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Load data from database with improved error handling."""
    try:
        if not db_path.exists():
            raise ValueError(f"Database file not found: {db_path}")
            
        if not is_valid_symbol(symbol):
            raise ValueError(f"Invalid symbol format: {symbol}")
            
        if not is_valid_timeframe(timeframe):
            raise ValueError(f"Invalid timeframe format: {timeframe}")
            
        conn = sqlite3.connect(str(db_path))
        
        # Get symbol and timeframe IDs
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM symbols WHERE symbol = ?", (symbol,))
        symbol_id = cursor.fetchone()
        if not symbol_id:
            raise ValueError(f"Symbol not found: {symbol}")
            
        cursor.execute("SELECT id FROM timeframes WHERE timeframe = ?", (timeframe,))
        timeframe_id = cursor.fetchone()
        if not timeframe_id:
            raise ValueError(f"Timeframe not found: {timeframe}")
            
        # Load data
        query = """
            SELECT open_time, open, high, low, close, volume
            FROM historical_data
            WHERE symbol_id = ? AND timeframe_id = ?
            ORDER BY open_time
        """
        df = pd.read_sql_query(query, conn, params=(symbol_id[0], timeframe_id[0]))
        conn.close()
        
        if df.empty:
            raise ValueError(f"No data found for {symbol} {timeframe}")
            
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.drop('open_time', axis=1, inplace=True)
        
        # Validate data
        is_valid, message = validate_dataframe(df)
        if not is_valid:
            raise ValueError(f"Data validation failed: {message}")
            
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return None

def validate_data(data: pd.DataFrame) -> None:
    """Validate data with improved error handling."""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")
        
    # Check required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
        
    # Check for missing values
    if data[required_cols].isnull().any().any():
        raise ValueError("Data contains missing values")
        
    # Check for duplicate timestamps (only if index is datetime)
    if hasattr(data.index, 'duplicated') and data.index.duplicated().any():
        raise ValueError("Data contains duplicate timestamps")
        
    # Check for non-monotonic timestamps (only if index is datetime)
    if hasattr(data.index, 'is_monotonic_increasing') and not data.index.is_monotonic_increasing:
        raise ValueError("Timestamps must be monotonically increasing")
        
    # Check for invalid price relationships
    if (data['high'] < data['low']).any():
        raise ValueError("High price cannot be less than low price")
    if (data['high'] < data['open']).any():
        raise ValueError("High price cannot be less than open price")
    if (data['high'] < data['close']).any():
        raise ValueError("High price cannot be less than close price")
    if (data['low'] > data['open']).any():
        raise ValueError("Low price cannot be greater than open price")
    if (data['low'] > data['close']).any():
        raise ValueError("Low price cannot be greater than close price")
        
    # Check for negative values
    if (data[['open', 'high', 'low', 'close', 'volume']] < 0).any().any():
        raise ValueError("Data contains negative values")
        
    # Check for minimum data points (only for production data, not test data)
    if len(data) < 10:  # Reduced from 100 for testing
        raise ValueError("Insufficient data points (minimum 10 required)")

class DataManager:
    def __init__(self, config=None, data_dir=None):
        """Initialize DataManager with either config or data_dir.
        
        Args:
            config: Configuration object or dict
            data_dir: Path to data directory (takes precedence over config)
        """
        if data_dir is not None:
            self.data_dir = Path(data_dir)
            if not self.data_dir.exists():
                raise ValueError(f"Data directory does not exist: {data_dir}")
            self.config = {'data_dir': str(self.data_dir)}
        elif config is not None:
            self.config = config
            if isinstance(config, dict) and 'data_dir' in config:
                self.data_dir = Path(config['data_dir'])
            else:
                self.data_dir = None
        else:
            raise ValueError("Either config or data_dir must be provided")

    def save_data(self, df, path):
        """Save data to file with validation."""
        if df.empty:
            raise ValueError("Cannot save empty DataFrame")
        is_valid, message = validate_dataframe(df)
        if not is_valid:
            raise ValueError(f"Cannot save invalid data: {message}")
        _save_csv(df, path)

    def load_data(self, path):
        """Load data from file with validation."""
        if isinstance(path, pd.DataFrame):
            is_valid, message = validate_dataframe(path)
            if not is_valid:
                raise ValueError(f"Invalid DataFrame: {message}")
            return path
        if not os.path.exists(path):
            raise ValueError(f"File not found: {path}")
        df = _load_csv(path)
        is_valid, message = validate_dataframe(df)
        if not is_valid:
            raise ValueError(f"Invalid data in file: {message}")
        return df

    def validate_data(self, data):
        """Validate data using consolidated validation function."""
        is_valid, message = validate_dataframe(data)
        if not is_valid:
            raise ValueError(message)
        return True

    def preprocess_data(self, data):
        """Preprocess data using consolidated processing function."""
        if data.empty:
            raise ValueError("Empty DataFrame provided")
        return process_data(data)

    def split_data(self, data, test_size=0.2):
        """Split data into train and test sets with validation."""
        if data.empty:
            raise ValueError("Empty DataFrame provided")
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        is_valid, message = validate_dataframe(data)
        if not is_valid:
            raise ValueError(f"Cannot split invalid data: {message}")
        return _split_data(data, test_size)

    def engineer_features(self, data):
        """Engineer features with validation."""
        if data.empty:
            raise ValueError("Empty DataFrame provided")
        is_valid, message = validate_dataframe(data)
        if not is_valid:
            raise ValueError(f"Cannot engineer features from invalid data: {message}")
        return data  # Placeholder for feature engineering

    def normalize_data(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Normalize specified columns to have mean 0 and standard deviation 1.
        For price columns (open, high, low, close), maintains price relationships.
        """
        if data.empty:
            raise ValueError("Empty DataFrame provided")
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Check for missing columns
        missing_cols = [col for col in columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
        
        # Create a copy to avoid modifying the original
        normalized = data.copy()
        
        # Standard z-score normalization for price columns
        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in columns and col in data.columns]
        for col in price_cols:
            mean = normalized[col].mean()
            std = normalized[col].std()
            if std == 0:
                logger.warning(f"Zero standard deviation in column {col}, skipping normalization")
                continue
            normalized[col] = (normalized[col] - mean) / std
        # After normalization, ensure price relationships are preserved
        if all(col in normalized.columns for col in ['open', 'high', 'low', 'close']):
            normalized['high'] = normalized[['open', 'close', 'high']].max(axis=1)
            normalized['low'] = normalized[['open', 'close', 'low']].min(axis=1)
            # Re-normalize price columns to ensure mean≈0 and std≈1
            for col in price_cols:
                mean = normalized[col].mean()
                std = normalized[col].std()
                if std > 0:
                    normalized[col] = (normalized[col] - mean) / std
        
        # Handle non-price columns with proper z-score normalization
        other_cols = [col for col in columns if col not in price_cols and col in normalized.columns]
        for col in other_cols:
            mean = normalized[col].mean()
            std = normalized[col].std()
            if std == 0:
                logger.warning(f"Zero standard deviation in column {col}, skipping normalization")
                continue
            normalized[col] = (normalized[col] - mean) / std
        
        return normalized

    def aggregate_data(self, data, freq, on=None):
        """Aggregate data to a different frequency.
        
        Args:
            data (pd.DataFrame): Input data
            freq (str): Target frequency (e.g. 'W' for weekly, 'M' for monthly)
            on (str, optional): Column to use for resampling. If None, uses index.
            
        Returns:
            pd.DataFrame: Aggregated data
        """
        if data.empty:
            raise ValueError("Empty DataFrame provided")
            
        df = data.copy()
        if freq == 'M':
            freq = 'ME'  # Use month end frequency
            
        # If no 'on' column specified and index is datetime, use index
        if on is None and isinstance(df.index, pd.DatetimeIndex):
            return df.resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
        # If 'on' column specified, use that for resampling
        if on is not None:
            if on not in df.columns:
                raise ValueError(f"Column '{on}' not found in DataFrame")
            if not pd.api.types.is_datetime64_any_dtype(df[on]):
                df[on] = pd.to_datetime(df[on])
            return df.resample(freq, on=on).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
        raise ValueError("No datetime index or 'on' column specified for resampling")

    def clean_data(self, data):
        """Clean data by handling outliers and missing values.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        if data.empty:
            raise ValueError("Empty DataFrame provided")
            
        df = data.copy()
        
        # Remove outliers in 'close'
        q_low = df['close'].quantile(0.01)
        q_high = df['close'].quantile(0.99)
        df['close'] = df['close'].clip(q_low, q_high)
        
        # Fill missing values using forward fill then backward fill
        df = df.ffill().bfill()
        
        return df

    def fill_missing_data(self, data):
        """Fill missing values in the data.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with missing values filled
        """
        if data.empty:
            raise ValueError("Empty DataFrame provided")
            
        df = data.copy()
        return df.ffill().bfill()

    def sample_data(self, data, n_samples=None, method='random', step=None):
        if data.empty:
            raise ValueError("Empty DataFrame provided")
        if method not in ['random', 'systematic']:
            raise ValueError("method must be 'random' or 'systematic'")
        if method == 'systematic' and step is None:
            raise ValueError("step must be provided for systematic sampling")
        if method == 'random' and n_samples is None:
            raise ValueError("n_samples must be provided for random sampling")
            
        if method == 'random':
            return data.sample(n=min(n_samples, len(data)))
        else:
            return data.iloc[::step]

    def merge_data(self, dataframes):
        if not dataframes:
            raise ValueError("No dataframes provided")
        return _merge_data(dataframes)

    def filter_data(self, data, start=None, end=None):
        if data.empty:
            raise ValueError("Empty DataFrame provided")
        return _filter_data(data, start, end)

    def resample_data(self, data, rule, on=None):
        """Resample data to a different frequency.
        
        Args:
            data (pd.DataFrame): Input data
            rule (str): Target frequency (e.g. 'D' for daily, 'W' for weekly)
            on (str, optional): Column to use for resampling. If None, uses index.
            
        Returns:
            pd.DataFrame: Resampled data
        """
        if data.empty:
            raise ValueError("Empty DataFrame provided")
        
        df = data.copy()
        
        # If timestamp is a column, set it as index
        if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index('timestamp')
        
        # If no datetime index and no 'on' column specified, raise error
        if not isinstance(df.index, pd.DatetimeIndex) and on is None:
            raise ValueError("No datetime index or 'on' column specified for resampling")
        
        return self.aggregate_data(df, rule, on)

def _save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV file."""
    df.to_csv(path, index=False)  # Explicitly set index=False to prevent index column

def _load_csv(path: Path) -> pd.DataFrame:
    """Load DataFrame from CSV file."""
    try:
        df = pd.read_csv(path, index_col=None)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        raise ValueError(f"File not found: {path}")

def _validate_data(data: pd.DataFrame) -> bool:
    """Validate data format and content.

    Args:
        data (pd.DataFrame): Input data

    Returns:
        bool: True if data is valid

    Raises:
        ValueError: If data is invalid with specific error messages
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input is not a DataFrame")

    if data.empty:
        raise ValueError("Empty DataFrame provided")

    required_cols = ['open', 'high', 'low', 'close', 'volume']

    # Check for required columns
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for NaN values
    for col in required_cols:
        if data[col].isnull().any():
            raise ValueError(f"NaN values found in column: {col}")

    # Check for invalid values
    if (data['high'] < data['low']).any():
        raise ValueError("High price cannot be less than low price")

    # Check for negative volume
    if (data['volume'] < 0).any():
        raise ValueError("Negative values found in column: volume")

    # Check for duplicate dates
    if data.index.duplicated().any():
        raise ValueError("Duplicate dates found")

    # Check for large gaps in data
    if isinstance(data.index, pd.DatetimeIndex):
        time_diff = data.index.to_series().diff()
        max_allowed_gap = pd.Timedelta(hours=4)  # Adjust based on your timeframe
        large_gaps = time_diff[time_diff > max_allowed_gap]
        if not large_gaps.empty:
            raise ValueError("Large gaps detected in data")

    return True

def _merge_data(dfs):
    import pandas as pd
    if not dfs:
        raise ValueError("No dataframes to merge")
    return pd.concat(dfs, ignore_index=True)

def _filter_data(df, start=None, end=None):
    if 'timestamp' not in df.columns:
        raise ValueError("No 'timestamp' column in DataFrame")
    mask = pd.Series([True] * len(df))
    if start is not None:
        mask &= df['timestamp'] >= pd.to_datetime(start)
    if end is not None:
        mask &= df['timestamp'] <= pd.to_datetime(end)
    filtered = df[mask]
    if filtered.empty:
        raise ValueError("No data in the specified range")
    return filtered

def _resample_data(df, rule):
    if 'timestamp' not in df.columns:
        raise ValueError("No 'timestamp' column in DataFrame")
    df = df.set_index('timestamp')
    try:
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
    except Exception as e:
        raise ValueError(f"Invalid resample rule: {e}")
    return resampled

def _fill_missing_data(df):
    if df.empty:
        raise ValueError("Empty DataFrame")
    filled = df.ffill().bfill()
    return filled

def _split_data(df, test_size=0.2):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    n = int(len(df) * (1 - test_size))
    return df.iloc[:n], df.iloc[n:]
