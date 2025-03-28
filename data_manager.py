# data_manager.py
import logging
import pandas as pd
import sqlite3
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta, timezone # Ensure timezone is imported
import time
import requests # For Binance API calls

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

def _fetch_klines(symbol: str, interval: str, start_time_ms: int, end_time_ms: int) -> List[List[Any]]:
    """Fetches klines data from Binance API."""
    all_klines = []
    current_start_time = start_time_ms
    # Use UTC for logging start/end times consistently
    logger.info(f"Fetching klines for {symbol} ({interval}) from {datetime.fromtimestamp(start_time_ms/1000, tz=timezone.utc)} to {datetime.fromtimestamp(end_time_ms/1000, tz=timezone.utc)}")

    retries = 3
    wait_time = 1 # seconds

    while current_start_time < end_time_ms:
        current_retries = retries
        last_exception = None
        while current_retries > 0:
            try:
                params = {
                    'symbol': symbol.upper(),
                    'interval': interval,
                    'startTime': current_start_time,
                    'endTime': end_time_ms,
                    'limit': MAX_LIMIT
                }
                logger.debug(f"Requesting Binance API: {params}")
                response = requests.get(BINANCE_API_BASE_URL, params=params, timeout=20) # Increased timeout
                logger.debug(f"Response Status Code: {response.status_code}")
                response.raise_for_status() # Raise HTTPError for bad responses (4XX, 5XX)
                klines = response.json()

                if not klines:
                    # If no data and it's the first fetch, log warning, otherwise it's just end of data
                    if not all_klines and current_start_time == start_time_ms:
                        logger.warning(f"No initial klines data returned for {symbol} starting {datetime.fromtimestamp(current_start_time/1000, tz=timezone.utc)}")
                    current_start_time = end_time_ms # Force outer loop exit
                    last_exception = None # Reset exception as it's not an error state
                    break # Exit retry loop, go to outer loop check

                all_klines.extend(klines)
                # Binance returns [open_time, open, ..., close_time, ...]
                last_kline_open_time = klines[-1][0]
                new_start_time = last_kline_open_time + 1 # Start next fetch from the ms after the last kline received

                if new_start_time <= current_start_time: # Avoid infinite loop if API returns unexpected data
                    logger.warning("API returned overlapping data or did not advance timestamp. Breaking fetch.")
                    current_start_time = end_time_ms # Force outer loop exit
                    last_exception = None
                    break # Exit retry loop

                current_start_time = new_start_time
                logger.info(f"Fetched {len(klines)} klines up to {datetime.fromtimestamp(last_kline_open_time/1000, tz=timezone.utc)}. Total: {len(all_klines)}")
                time.sleep(0.15) # Respect Binance API rate limits
                last_exception = None # Reset exception on success
                break # Success, exit retry loop

            except requests.exceptions.Timeout as timeout_err:
                 last_exception = timeout_err
                 logger.warning(f"Timeout fetching klines. Retries left: {current_retries-1}. Waiting {wait_time}s...")
                 current_retries -= 1
                 time.sleep(wait_time)
                 wait_time *= 2 # Exponential backoff
            except requests.exceptions.RequestException as req_err:
                 last_exception = req_err
                 # Check for specific rate limit errors (e.g., 429, 418)
                 if hasattr(req_err, 'response') and req_err.response is not None:
                     if req_err.response.status_code == 429 or req_err.response.status_code == 418:
                          logger.warning(f"Rate limit hit (Status {req_err.response.status_code}). Retrying after longer wait...")
                          wait_time = max(wait_time, 10) # Wait at least 10s for rate limits
                     else:
                          logger.error(f"HTTP error fetching klines: {req_err}. Status: {req_err.response.status_code}")
                          # Maybe don't retry on non-rate-limit client/server errors? For now, retry.
                 else:
                      logger.error(f"Request error fetching klines: {req_err}.")

                 current_retries -= 1
                 if current_retries > 0: logger.warning(f"Retries left: {current_retries}. Waiting {wait_time}s...")
                 time.sleep(wait_time)
                 wait_time *= 2 # Exponential backoff
            except Exception as e:
                 last_exception = e
                 logger.error(f"Unexpected error during kline fetch: {e}", exc_info=True)
                 current_start_time = end_time_ms # Force outer loop exit on unexpected error
                 break # Exit retry loop

        if last_exception is not None: # If all retries failed
             logger.error(f"Failed to fetch klines after multiple retries. Last error: {last_exception}")
             return [] # Return empty list on persistent failure

    logger.info(f"Finished fetching. Total klines retrieved: {len(all_klines)}")
    return all_klines

def _process_klines(klines: List[List[Any]]) -> pd.DataFrame:
    """Converts raw klines list to a pandas DataFrame with correct types."""
    if not klines:
        return pd.DataFrame()
    logger.debug(f"Processing {len(klines)} raw klines into DataFrame...")
    try:
        df = pd.DataFrame(klines, columns=KLINES_COLUMNS)
        df = df.drop(columns=['ignore'])

        # Convert columns to appropriate types
        numeric_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume",
                        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]
        int_cols = ["number_of_trades"]
        time_cols = ["open_time", "close_time"] # Keep as ms integers for DB

        # Use apply with pd.to_numeric for robust conversion
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in int_cols:
            # Convert to float first to handle potential non-int strings before nullable Int64
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float).astype('Int64')
        for col in time_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

        # Drop rows where essential time/price/volume data is missing *after* conversion
        core_cols_check = time_cols + ['open', 'high', 'low', 'close', 'volume']
        initial_len = len(df)
        df.dropna(subset=core_cols_check, inplace=True)
        if len(df) < initial_len:
            logger.warning(f"Dropped {initial_len - len(df)} rows with NaN in core columns during processing.")

        # Remove potential duplicates based on open_time
        initial_len = len(df)
        df.drop_duplicates(subset=['open_time'], keep='first', inplace=True)
        if len(df) < initial_len:
            logger.warning(f"Removed {initial_len - len(df)} duplicate klines based on open_time.")

        if df.empty:
             logger.warning("DataFrame became empty after processing and cleaning.")
             return pd.DataFrame()

        # Log first 5 rows after processing
        logger.info(f"Processed klines sample (first 5 rows):\n{df.head(5).to_string()}")
        logger.debug(f"Processed klines DataFrame shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error processing klines list into DataFrame: {e}", exc_info=True)
        return pd.DataFrame() # Return empty on failure

def _save_to_sqlite(df: pd.DataFrame, db_path: str, symbol: str, timeframe: str) -> bool:
    """Saves the DataFrame to the historical_data table in SQLite."""
    if df.empty:
        logger.info("No processed data to save.")
        return True # Not an error if there was nothing to save

    logger.debug(f"Attempting to save {len(df)} records to {db_path}...")
    conn = sqlite_manager.create_connection(db_path)
    if not conn:
        return False

    try:
        # Get IDs within the same connection context if possible
        symbol_id = sqlite_manager._get_or_create_id(conn, 'symbols', 'symbol', symbol)
        timeframe_id = sqlite_manager._get_or_create_id(conn, 'timeframes', 'timeframe', timeframe)

        df_to_insert = df.copy()
        df_to_insert['symbol_id'] = symbol_id
        df_to_insert['timeframe_id'] = timeframe_id

        # Explicitly list columns expected by the DB schema
        db_columns = [
            "symbol_id", "timeframe_id", "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
        ]
        # Select only columns present in both df and db_columns list, maintaining order
        cols_to_keep = [col for col in db_columns if col in df_to_insert.columns]
        df_to_insert = df_to_insert[cols_to_keep]

        # Use INSERT OR IGNORE behavior via handling IntegrityError
        logger.debug(f"Executing to_sql with if_exists='append', chunksize=1000")
        df_to_insert.to_sql('historical_data', conn, if_exists='append', index=False, method='multi', chunksize=1000)

        conn.commit()
        logger.info(f"Successfully saved/updated {len(df_to_insert)} records to {db_path}")
        return True

    except sqlite3.IntegrityError as ie:
         conn.rollback()
         logger.warning(f"Integrity error saving data (likely duplicates): {ie}.")
         return True # Treat as non-fatal if likely duplicate
    except sqlite3.Error as e:
        conn.rollback()
        logger.error(f"Error saving data to SQLite '{db_path}': {e}", exc_info=True)
        return False
    except Exception as e:
        conn.rollback()
        logger.error(f"Unexpected error saving data: {e}", exc_info=True)
        return False
    finally:
        if conn:
            conn.close()
            logger.debug(f"Closed connection to {db_path} after save attempt.")

def _get_last_timestamp(db_path: str, symbol_id: int, timeframe_id: int) -> Optional[int]:
    """Gets the last recorded open_time (ms) from the database."""
    if not Path(db_path).exists():
         logger.info(f"Database file {db_path} does not exist. No last timestamp.")
         return None

    conn = sqlite_manager.create_connection(db_path)
    if not conn: return None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='historical_data';")
        if cursor.fetchone() is None:
             logger.warning(f"Table 'historical_data' not found in {db_path}. Cannot get last timestamp.")
             return None

        cursor.execute("""
            SELECT MAX(open_time) FROM historical_data
            WHERE symbol_id = ? AND timeframe_id = ?
        """, (symbol_id, timeframe_id))
        result = cursor.fetchone()
        last_ts = result[0] if result and result[0] is not None else None
        if last_ts:
             logger.info(f"Last timestamp found in DB for SymbolID {symbol_id}, TFID {timeframe_id}: {last_ts} ({datetime.fromtimestamp(last_ts/1000, tz=timezone.utc)})")
        else:
             logger.info(f"No last timestamp found in DB for SymbolID {symbol_id}, TFID {timeframe_id}.")
        return last_ts
    except sqlite3.Error as e:
        logger.error(f"Error getting last timestamp from '{db_path}': {e}", exc_info=True)
        return None
    finally:
        if conn: conn.close()

def download_binance_data(symbol: str, timeframe: str, db_path: Path) -> bool:
    """Downloads data from Binance and saves to SQLite database."""
    logger.info(f"Starting data download process for {symbol} ({timeframe}) into {db_path}")

    if not sqlite_manager.initialize_database(str(db_path)):
        logger.error("Failed to initialize database. Aborting download.")
        return False

    conn = sqlite_manager.create_connection(str(db_path))
    if not conn: return False
    try:
        symbol_id = sqlite_manager._get_or_create_id(conn, 'symbols', 'symbol', symbol)
        timeframe_id = sqlite_manager._get_or_create_id(conn, 'timeframes', 'timeframe', timeframe)
    except Exception as e:
        logger.error(f"Failed to get symbol/timeframe IDs: {e}", exc_info=True)
        if conn: conn.close()
        return False
    finally:
        if conn: conn.close()

    last_ts = _get_last_timestamp(str(db_path), symbol_id, timeframe_id)

    start_dt_default = datetime(2017, 1, 1, tzinfo=timezone.utc)
    start_time_ms = last_ts + 1 if last_ts is not None else int(start_dt_default.timestamp() * 1000)
    end_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    if start_time_ms >= end_time_ms:
        logger.info(f"Database appears up-to-date for {symbol} ({timeframe}) (Start: {datetime.fromtimestamp(start_time_ms/1000, tz=timezone.utc)}, End: {datetime.fromtimestamp(end_time_ms/1000, tz=timezone.utc)}). No download needed.")
        return True

    klines = _fetch_klines(symbol, timeframe, start_time_ms, end_time_ms)
    if not klines:
        logger.warning(f"No klines data fetched for {symbol} {timeframe} in the required range.")
        is_up_to_date = (last_ts is not None and start_time_ms == last_ts + 1)
        return is_up_to_date

    df = _process_klines(klines)
    if df.empty:
         logger.warning(f"Processing fetched klines resulted in empty DataFrame for {symbol} {timeframe}.")
         return last_ts is not None

    save_success = _save_to_sqlite(df, str(db_path), symbol, timeframe)
    return save_success

def list_existing_databases() -> List[Path]:
    """Lists existing .db files in the database directory."""
    return sorted([p for p in config.DB_DIR.glob("*.db") if p.is_file()])

def select_existing_database() -> Optional[Path]:
    """Prompts the user to select an existing database."""
    db_files = list_existing_databases()
    if not db_files:
        print("No existing databases found in:", config.DB_DIR)
        logger.warning("No existing databases found.")
        return None

    print("\nExisting Databases:")
    for idx, db_path in enumerate(db_files):
        print(f"{idx + 1}. {db_path.name}")

    while True:
        try:
            choice = input(f"Select database by number (1-{len(db_files)}) or 'n' for new/update: ").strip().lower()
            if choice == 'n':
                logger.info("User chose 'n' (new/update) instead of selecting existing DB.")
                return None # Indicates user wants to update/download, not select
            selected_idx = int(choice) - 1
            if 0 <= selected_idx < len(db_files):
                selected_db = db_files[selected_idx]
                logger.info(f"User selected existing database: {selected_db.name}")
                return selected_db
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input. Please enter a number or 'n'.")
        except Exception as e:
            logger.error(f"Error during database selection: {e}", exc_info=True)
            print("An error occurred during selection. Please check logs.")
            return None # Exit selection on unexpected error

def validate_data(db_path: Path) -> bool:
    """Quickly validates if the database contains any historical data."""
    conn = sqlite_manager.create_connection(str(db_path))
    if not conn: return False
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='historical_data';")
        if cursor.fetchone() is None:
             logger.warning(f"Table 'historical_data' does not exist in {db_path.name}. Validation failed.")
             return False
        cursor.execute("SELECT COUNT(*) FROM historical_data")
        count = cursor.fetchone()[0]
        if count > 0:
            logger.info(f"Database '{db_path.name}' contains {count} historical records.")
            return True
        else:
            logger.warning(f"Database '{db_path.name}' contains no historical data.")
            return False
    except sqlite3.Error as e:
        logger.error(f"Error validating data in '{db_path.name}': {e}", exc_info=True)
        return False
    finally:
        if conn: conn.close()

def manage_data_source() -> Optional[Tuple[Path, str, str]]:
    """Handles user interaction for selecting, updating, or downloading data."""
    while True:
        print("\n--- Data Source ---")
        action = input("Select action: [S]elect, [U]pdate, [D]ownload (default), [Q]uit: ").strip().lower() or 'd'
        logger.info(f"User selected data source action: '{action}'") # Log the action

        if action == 's':
            selected_db = select_existing_database()
            if selected_db: # User selected a file
                logger.info(f"User attempting to select DB: {selected_db.name}")
                if validate_data(selected_db):
                     try:
                        base_name = selected_db.stem
                        symbol, timeframe = base_name.split('_', 1)
                        logger.info(f"Data source selected and validated: {selected_db.name}")
                        return selected_db, symbol.upper(), timeframe
                     except ValueError:
                         logger.error(f"Could not parse symbol/timeframe from filename: {selected_db.name}")
                         print("Error: Invalid database filename format (expected 'SYMBOL_timeframe.db').")
                else:
                    print(f"Selected database '{selected_db.name}' is empty or invalid. Please Update or Download.")
            # If selected_db is None (user chose 'n'), loop continues

        elif action == 'u':
            selected_db = select_existing_database()
            if selected_db: # User selected a file to update
                logger.info(f"User attempting to update DB: {selected_db.name}")
                try:
                    base_name = selected_db.stem
                    symbol, timeframe = base_name.split('_', 1)
                    print(f"Updating data for {symbol.upper()} ({timeframe})...")
                    logger.info(f"Calling download_binance_data for update: {symbol.upper()}, {timeframe}")
                    success = download_binance_data(symbol.upper(), timeframe, selected_db)
                    if success and validate_data(selected_db):
                        print("Update successful.")
                        logger.info(f"Data source updated and validated: {selected_db.name}")
                        return selected_db, symbol.upper(), timeframe
                    elif success:
                         print("Data is already up-to-date.")
                         logger.info(f"Data source already up-to-date: {selected_db.name}")
                         return selected_db, symbol.upper(), timeframe
                    else:
                        print("Update failed. Check logs.")
                except ValueError:
                    logger.error(f"Could not parse symbol/timeframe from filename: {selected_db.name}")
                    print("Error: Invalid database filename format (expected 'SYMBOL_timeframe.db').")
            # If selected_db is None (user chose 'n'), loop continues

        elif action == 'd':
            symbol = input("Enter symbol (default BTCUSDT): ").strip().upper() or "BTCUSDT"
            # --- DEFAULT TIMEFRAME CHANGED HERE ---
            timeframe = input("Enter timeframe (default 1w): ").strip().lower() or "1w"
            # --- END CHANGE ---
            logger.info(f"User requested download for Symbol='{symbol}', Timeframe='{timeframe}'")

            if not symbol or not timeframe:
                print("Symbol and timeframe cannot be empty.")
                continue

            db_filename = config.DB_NAME_TEMPLATE.format(symbol=symbol, timeframe=timeframe)
            db_path = config.DB_DIR / db_filename
            print(f"Downloading/Updating data for {symbol} ({timeframe}) into {db_path.name}...")
            logger.info(f"Calling download_binance_data for download/update: {symbol}, {timeframe}")
            success = download_binance_data(symbol, timeframe, db_path)

            if success and validate_data(db_path):
                 print("Download/Update successful.")
                 logger.info(f"Data source downloaded/updated and validated: {db_path.name}")
                 return db_path, symbol, timeframe
            elif success:
                 print("Data source is up-to-date, but validation failed or no data.")
                 logger.warning(f"Data source {db_path.name} reported success/up-to-date but failed validation.")
            else:
                 print("Download/Update failed. Check logs.")

        elif action == 'q':
            logger.info("User quit data source selection.")
            return None
        else:
            print("Invalid action.")
            logger.warning(f"Invalid user action input: '{action}'")

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
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='historical_data';")
        if cursor.fetchone() is None:
             logger.error(f"Table 'historical_data' not found in {db_path}. Cannot load data.")
             return None

        query = "SELECT * FROM historical_data ORDER BY open_time ASC"
        df = pd.read_sql_query(query, conn)
        logger.info(f"Loaded {len(df)} records from DB.")

        if not df.empty:
            logger.info(f"Loaded data sample from DB (first 5 rows):\n{df.head(5).to_string()}")
        else:
            logger.warning("Loaded DataFrame from DB is empty (no sample to show).")
            return None

        df['date'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)

        required_cols = ['open_time', 'date', 'open', 'high', 'low', 'close', 'volume']
        core_numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
             logger.error(f"Loaded data missing required columns: {missing}")
             return None

        for col in core_numeric_cols:
             df[col] = pd.to_numeric(df[col], errors='coerce')

        initial_len = len(df)
        df.dropna(subset=core_numeric_cols + ['date'], inplace=True)
        dropped_rows = initial_len - len(df)
        if dropped_rows > 0:
             logger.info(f"Dropped {dropped_rows} rows with NaNs in core columns during loading.")

        if df.empty:
            logger.warning("DataFrame became empty after dropping NaNs in core columns.")
            return None

        if not df['date'].is_monotonic_increasing:
             logger.warning("Loaded data 'date' column is not monotonic. Sorting again.")
             df.sort_values('date', inplace=True)

        logger.info(f"Data loaded and basic validation passed. Final shape: {df.shape}")
        return df

    except (pd.errors.DatabaseError, sqlite3.Error) as e:
        logger.error(f"Database error loading data from {db_path}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}", exc_info=True)
        return None
    finally:
        if conn:
            conn.close()