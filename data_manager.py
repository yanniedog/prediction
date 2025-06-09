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
    if df.empty: logger.info("No processed data to save."); return True

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
    if not conn: return False

    try:
        # Get IDs within a transaction
        conn.execute("BEGIN IMMEDIATE;") # Use IMMEDIATE for faster lock acquisition
        symbol_id = sqlite_manager._get_or_create_id(conn, 'symbols', 'symbol', symbol)
        timeframe_id = sqlite_manager._get_or_create_id(conn, 'timeframes', 'timeframe', timeframe)
        # We commit IDs later with the data

        df_to_insert = df.copy()
        df_to_insert['symbol_id'] = symbol_id
        df_to_insert['timeframe_id'] = timeframe_id

        db_columns = [
            "symbol_id", "timeframe_id", "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
        ]
        # Ensure only existing columns are selected
        cols_to_keep = [col for col in db_columns if col in df_to_insert.columns]
        df_to_insert = df_to_insert[cols_to_keep]

        required_db_cols = ["symbol_id", "timeframe_id", "open_time", "open", "high", "low", "close", "volume", "close_time"]
        if not all(col in df_to_insert.columns for col in required_db_cols):
            logger.error(f"DataFrame missing required DB columns for saving: {required_db_cols}. Have: {list(df_to_insert.columns)}. Aborting save.")
            conn.rollback()
            return False

        # --- Explicit Type Conversion Before Saving ---
        logger.debug("Performing explicit type conversion before saving...")
        type_map = {
            "symbol_id": 'int64', "timeframe_id": 'int64',
            "open_time": 'int64', "close_time": 'int64', # Crucial: Ensure these are standard integers
            "open": 'float64', "high": 'float64', "low": 'float64', "close": 'float64', "volume": 'float64',
            "quote_asset_volume": 'float64',
            "number_of_trades": 'float64', # Store as float to allow NULLs easily in SQLite via pandas
            "taker_buy_base_asset_volume": 'float64', "taker_buy_quote_asset_volume": 'float64'
        }
        conversion_errors = False
        for col, target_type in type_map.items():
            if col in df_to_insert.columns:
                try:
                    current_dtype = df_to_insert[col].dtype
                    logger.debug(f"Converting column '{col}' (current: {current_dtype}) to {target_type}...")
                    if pd.api.types.is_integer_dtype(current_dtype) and df_to_insert[col].isnull().any():
                        # If integer column has nulls, convert to float for SQLite compatibility via pandas
                         if target_type.startswith('int'):
                             logger.warning(f"Column '{col}' is integer with NaNs. Converting to float64 for SQLite storage.")
                             df_to_insert[col] = df_to_insert[col].astype('float64')
                         else:
                             df_to_insert[col] = df_to_insert[col].astype(target_type) # Keep target type if already float
                    elif current_dtype != target_type:
                         # Explicitly convert non-nullable integers to standard Python int first if possible
                         if target_type == 'int64' and pd.api.types.is_integer_dtype(current_dtype):
                             # Ensure it's not already int64 before converting, avoid unnecessary work
                             if str(current_dtype) != 'int64':
                                  df_to_insert[col] = df_to_insert[col].astype('int64')
                         else:
                             df_to_insert[col] = df_to_insert[col].astype(target_type)
                except Exception as conv_err:
                    logger.error(f"Failed to convert column '{col}' to {target_type}: {conv_err}", exc_info=True)
                    conversion_errors = True
            else:
                logger.warning(f"Column '{col}' defined in type_map not found in DataFrame to insert.")

        if conversion_errors:
            logger.error("Errors occurred during type conversion before saving. Aborting.")
            conn.rollback()
            return False
        # --- End Explicit Type Conversion ---

        # Add a final sanity check log for open_time type and min value
        logger.debug(f"Final check before to_sql: 'open_time' dtype={df_to_insert['open_time'].dtype}, min={df_to_insert['open_time'].min()}")
        if df_to_insert['open_time'].min() < min_valid_timestamp_ms:
             logger.error(f"CRITICAL: 'open_time' invalid ({df_to_insert['open_time'].min()}) immediately before to_sql. Aborting.")
             conn.rollback()
             return False

        try:
            # Use pandas to_sql with 'append'. DB's UNIQUE constraint handles duplicates.
            df_to_insert.to_sql('historical_data', conn, if_exists='append', index=False, method='multi', chunksize=1000)
            conn.commit() # Commit data and IDs together
            logger.info(f"Successfully appended {len(df_to_insert)} records to {db_path}. DB's UNIQUE constraint handles duplicates.")
            return True
        except sqlite3.IntegrityError as ie:
             conn.rollback() # Rollback on integrity error
             logger.warning(f"Integrity error during append (likely duplicate open_time): {ie}. Transaction rolled back.")
             # Duplicates mean the data is likely already there, consider success?
             # Let's return False to be safe, as other integrity issues could cause this.
             return False
        except sqlite3.Error as db_err: # Catch other DB errors during to_sql
            conn.rollback()
            logger.error(f"SQLite error during save: {db_err}. Rolling back.")
            return False

    except Exception as e:
        try: conn.rollback()
        except: pass
        logger.error(f"Unexpected error saving data: {e}", exc_info=True)
        return False
    finally:
        if conn:
            conn.close()
            logger.debug(f"Closed connection to {db_path} after save attempt.")

# --- Timestamp & Data Download Control ---
def _get_last_timestamp(db_path: str, symbol_id: int, timeframe_id: int) -> Optional[int]:
    """Gets the last recorded open_time (milliseconds) from the database."""
    db_file = Path(db_path)
    if not db_file.is_file():
         logger.info(f"Database file {db_path} not found. No last timestamp available.")
         return None

    conn = sqlite_manager.create_connection(db_path)
    if not conn: return None
    try:
        cursor = conn.cursor()
        # Verify table exists first
        cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='historical_data';")
        if cursor.fetchone() is None:
             logger.warning(f"Table 'historical_data' not found in {db_path}. Cannot get last timestamp.")
             return None

        cursor.execute("SELECT MAX(open_time) FROM historical_data WHERE symbol_id = ? AND timeframe_id = ?", (symbol_id, timeframe_id))
        result = cursor.fetchone()
        last_ts = result[0] if result and result[0] is not None else None

        if last_ts:
             # ---> ADDED: Validate last_ts here too <---
             min_valid_timestamp_ms = EARLIEST_VALID_DATE.timestamp() * 1000
             if last_ts < min_valid_timestamp_ms:
                 logger.error(f"Last timestamp found in DB ({last_ts}) is unreasonably old. Treating as no previous data.")
                 return None
             # ---> END ADDED VALIDATION <---
             # ---> Corrected Timestamp Conversion <---
             last_dt_utc = datetime.fromtimestamp(last_ts/1000).replace(tzinfo=timezone.utc)
             # ---> End Correction <---
             logger.info(f"Last timestamp found in DB for SymbolID {symbol_id}, TFID {timeframe_id}: {last_ts} ({last_dt_utc})")
        else:
             logger.info(f"No previous data found in DB for SymbolID {symbol_id}, TFID {timeframe_id}.")
        return last_ts
    except sqlite3.Error as e:
        logger.error(f"Error getting last timestamp from '{db_path}': {e}", exc_info=True)
        return None
    finally:
        if conn: conn.close()

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
    """Prompts user to select an existing database."""
    db_files = list_existing_databases()
    if not db_files:
        print("No existing symbol databases found in:", config.DB_DIR)
        return None

    print("\nExisting Symbol Databases:")
    for idx, db_path in enumerate(db_files): print(f"{idx + 1}. {db_path.name}")

    while True:
        try:
            choice = input(f"Select database by number (1-{len(db_files)}) or 'n' for new/update: ").strip().lower()
            if choice == 'n': return None
            selected_idx = int(choice) - 1
            if 0 <= selected_idx < len(db_files):
                selected_db = db_files[selected_idx]
                logger.info(f"User selected existing database: {selected_db.name}")
                return selected_db
            else: print("Invalid selection.")
        except ValueError: print("Invalid input. Please enter a number or 'n'.")
        except Exception as e:
            logger.error(f"Error during database selection: {e}", exc_info=True)
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

def manage_data_source() -> Optional[Tuple[Path, str, str]]:
    """Handles user interaction for selecting, updating, or downloading data."""
    while True:
        print("\n--- Data Source ---")
        action = input("Select action: [S]elect existing, [U]pdate existing, [D]ownload new/update (default), [Q]uit: ").strip().lower() or 'd'
        logger.info(f"User selected data source action: '{action}'")

        selected_db = None
        symbol, timeframe = None, None

        if action == 's':
            selected_db = select_existing_database()
            if selected_db:
                logger.info(f"Validating selected DB: {selected_db.name}")
                if validate_data(selected_db):
                     try:
                        base_name = selected_db.stem
                        # Be more robust splitting symbol/timeframe
                        parts = base_name.split('_', 1)
                        if len(parts) == 2:
                             symbol, timeframe = parts
                             return selected_db, symbol.upper(), timeframe
                        else:
                            logger.error(f"Invalid filename format: {selected_db.name}")
                            print("Error: Invalid DB filename format (expected 'SYMBOL_timeframe.db').")
                     except Exception as parse_err:
                         logger.error(f"Error parsing filename {selected_db.name}: {parse_err}")
                         print("Error: Could not parse filename.")
                else:
                    print(f"Selected database '{selected_db.name}' is empty or invalid. Try Update or Download.")
            # If selected_db is None (user chose 'n'), loop continues

        elif action == 'u':
            selected_db = select_existing_database()
            if selected_db:
                try:
                    base_name = selected_db.stem
                    parts = base_name.split('_', 1)
                    if len(parts) == 2:
                        symbol, timeframe = parts
                        print(f"Updating data for {symbol.upper()} ({timeframe})...")
                        success = download_binance_data(symbol.upper(), timeframe, selected_db)
                        if success and validate_data(selected_db):
                            print("Update successful or data already up-to-date.")
                            return selected_db, symbol.upper(), timeframe
                        else:
                            print("Update failed or DB validation failed after update. Check logs.")
                    else:
                         logger.error(f"Invalid filename format: {selected_db.name}")
                         print("Error: Invalid DB filename format (expected 'SYMBOL_timeframe.db').")
                except Exception as update_err:
                     logger.error(f"Error during update selection for {selected_db.name}: {update_err}")
                     print("Error processing update selection.")
            # If selected_db is None, loop continues

        elif action == 'd':
            default_sym = config.DEFAULTS.get("symbol", "BTCUSDT")
            symbol_input = input(f"Enter symbol [default: {default_sym}]: ").strip().upper() or default_sym
            symbol = symbol_input

            print("\nCommon Binance Timeframes:")
            tf_groups = {"Minutes": [], "Hours": [], "Days": [], "Weeks": [], "Month": []}
            for tf in VALID_TIMEFRAMES:
                if 'm' in tf: tf_groups["Minutes"].append(tf)
                elif 'h' in tf: tf_groups["Hours"].append(tf)
                elif 'd' in tf: tf_groups["Days"].append(tf)
                elif 'w' in tf: tf_groups["Weeks"].append(tf)
                elif 'M' in tf: tf_groups["Month"].append(tf)
            for group, tfs in tf_groups.items():
                if tfs: print(f"  {group}: {', '.join(tfs)}")

            default_tf = config.DEFAULTS.get("timeframe", "1d")
            timeframe_input = input(f"Enter timeframe [default: {default_tf}]: ").strip() or default_tf
            timeframe = timeframe_input # Keep original case for API call

            if timeframe not in VALID_TIMEFRAMES:
                logger.warning(f"User timeframe '{timeframe}' not in common list. Ensure valid for API.")
                print(f"Warning: '{timeframe}' not common. Ensure valid Binance string.")

            if not symbol: print("Symbol cannot be empty."); continue

            safe_timeframe_fn = re.sub(r'[\\/*?:"<>|\s]+', '_', timeframe)
            db_filename = config.DB_NAME_TEMPLATE.format(symbol=symbol, timeframe=safe_timeframe_fn)
            db_path = config.DB_DIR / db_filename
            print(f"\nDownloading/Updating data for {symbol} ({timeframe}) into {db_path.name}...")

            success = download_binance_data(symbol, timeframe, db_path)
            if success and validate_data(db_path):
                 print("Download/Update successful.")
                 return db_path, symbol, timeframe
            else:
                 print("Download/Update failed or DB validation failed. Check logs.")

        elif action == 'q':
            logger.info("User quit data source selection.")
            return None
        else:
            print("Invalid action. Please choose S, U, D, or Q.")
            logger.warning(f"Invalid user action input: '{action}'")

# --- Load Data ---
def load_data(db_path: Path) -> Optional[pd.DataFrame]:
    """Loads historical data from the specified SQLite database file."""
    if not db_path.exists():
        logger.error(f"Database file does not exist: {db_path}")
        return None

    logger.info(f"Loading data from {db_path}...")
    conn = sqlite_manager.create_connection(str(db_path))
    if not conn: return None

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='historical_data';")
        if cursor.fetchone() is None:
             logger.error(f"Table 'historical_data' not found in {db_path}. Cannot load data.")
             return None

        query = "SELECT * FROM historical_data ORDER BY open_time ASC"
        df = pd.read_sql_query(query, conn)
        logger.info(f"Loaded {len(df)} records from DB.")

        if df.empty: 
            logger.warning("Loaded DataFrame is empty.")
            raise ValueError("Insufficient data points")

        # Basic data cleaning and type conversion
        df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
        df.dropna(subset=['open_time'], inplace=True)
        if df.empty: 
            logger.warning("DF empty after dropping invalid open_time.")
            raise ValueError("Insufficient data points")

        # Filter out unreasonably old timestamps
        min_valid_timestamp_ms = EARLIEST_VALID_DATE.timestamp() * 1000
        initial_len = len(df)
        df = df[df['open_time'] >= min_valid_timestamp_ms]
        if len(df) < initial_len:
             logger.warning(f"Filtered out {initial_len - len(df)} rows with open_time before {EARLIEST_VALID_DATE.date()} during load.")
        if df.empty: 
            logger.error("DataFrame empty after filtering old timestamps during load.")
            raise ValueError("Insufficient data points")

        df['date'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df.dropna(subset=['date'], inplace=True)
        if df.empty: 
            logger.warning("DF empty after dropping invalid date conversion.")
            raise ValueError("Insufficient data points")

        # Set date as index for validation
        df.set_index('date', inplace=True)
        
        # Validate data
        try:
            _validate_data(df)
        except ValueError as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise

        logger.info(f"Data loaded and validated. Final shape: {df.shape}")
        return df

    except (pd.errors.DatabaseError, sqlite3.Error) as e:
        logger.error(f"Database error loading data from {db_path}: {e}", exc_info=True)
        raise ValueError(f"Database error: {str(e)}")
    except ValueError as e:
        raise  # Re-raise validation errors
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}", exc_info=True)
        raise ValueError(f"Error loading data: {str(e)}")
    finally:
        if conn: conn.close()

class DataManager:
    def __init__(self, config):
        self.config = config

    def save_data(self, df, path):
        if df.empty:
            raise ValueError("Cannot save empty DataFrame")
        _save_csv(df, path)

    def load_data(self, path):
        if isinstance(path, pd.DataFrame):
            return path
        if not os.path.exists(path):
            raise ValueError(f"File not found: {path}")
        return _load_csv(path)

    def validate_data(self, data):
        if data.empty:
            raise ValueError("Empty DataFrame provided")
        return _validate_data(data)

    def preprocess_data(self, data):
        if data.empty:
            raise ValueError("Empty DataFrame provided")
        return self.clean_data(data)

    def split_data(self, data, test_size=0.2):
        if data.empty:
            raise ValueError("Empty DataFrame provided")
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        return _split_data(data, test_size)

    def engineer_features(self, data):
        if data.empty:
            raise ValueError("Empty DataFrame provided")
        return data  # Placeholder for feature engineering

    def normalize_data(self, data, columns=None):
        """Normalize data using z-score normalization.
        
        Args:
            data (pd.DataFrame): Input data
            columns (list, optional): List of columns to normalize. If None, normalizes all numeric columns.
            
        Returns:
            pd.DataFrame: Normalized data with mean=0 and std=1
        """
        if data.empty:
            raise ValueError("Empty DataFrame provided")
            
        df = data.copy()
        
        # Store index if it's datetime
        is_datetime_index = isinstance(df.index, pd.DatetimeIndex)
        if is_datetime_index:
            index = df.index
            df = df.reset_index(drop=True)
        
        # Select columns to normalize
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
        else:
            # Validate columns exist
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
            # Validate columns are numeric
            non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]
            if non_numeric:
                raise ValueError(f"Non-numeric columns: {non_numeric}")
            numeric_cols = columns
        
        # Z-score normalization
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std != 0:
                df[col] = (df[col] - mean) / std
        
        # Restore index if it was datetime
        if is_datetime_index:
            df.index = index
        
        return df

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

def _validate_data(data):
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
    if (data['volume'] < 0).any():
        raise ValueError("Volume cannot be negative")
        
    # Check for duplicate dates if index is datetime
    if isinstance(data.index, pd.DatetimeIndex):
        if not data.index.is_monotonic_increasing:
            raise ValueError("Dates must be monotonically increasing")
        if data.index.duplicated().any():
            raise ValueError("Duplicate dates found")
        # Check for large gaps in the datetime index
        diffs = data.index.to_series().diff().dropna()
        if len(diffs) > 0:
            median_diff = diffs.median()
            if (diffs > median_diff * 3).any():
                raise ValueError("Large gaps detected in data")
        
    # Check for sufficient data points
    if len(data) < 30:  # Minimum required data points
        raise ValueError("Insufficient data points")
        
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
