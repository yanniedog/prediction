import os, sys, sqlite3, logging, pandas as pd
from config import DB_PATH
from sqlite_data_manager import create_connection, create_tables

def load_data(symbol, timeframe):
    logging.basicConfig(level=logging.INFO)
    db_path = DB_PATH
    if not os.path.exists(db_path):
        logging.warning(f"DB '{db_path}' missing. Creating new DB.")
        conn = create_connection(db_path)
        if conn:
            create_tables(conn)
            conn.close()
        else:
            logging.error(f"Failed to connect to '{db_path}'. Exiting.")
            sys.exit(1)
    try:
        conn = sqlite3.connect(db_path)
        logging.info(f"Connected to DB at: {db_path}")
    except sqlite3.Error as e:
        logging.error(f"DB connection failed: {e}")
        sys.exit(1)
    query = """
    SELECT klines.* FROM klines
    JOIN symbols ON klines.symbol_id = symbols.id
    JOIN timeframes ON klines.timeframe_id = timeframes.id
    WHERE symbols.symbol = ? AND timeframes.timeframe = ?
    ORDER BY open_time ASC
    """
    try:
        data = pd.read_sql_query(query, conn, params=(symbol, timeframe), parse_dates=['open_time', 'close_time'])
        logging.info("Data loaded from DB.")
    except pd.io.sql.DatabaseError as e:
        logging.error(f"Query failed: {e}")
        conn.close()
        sys.exit(1)
    conn.close()
    if data.empty:
        logging.warning("No data in DB.")
        return data, False, os.path.basename(db_path)
    time_col = 'open_time'
    is_rev = data[time_col].is_monotonic_decreasing
    if is_rev:
        data = data.sort_values(time_col).reset_index(drop=True)
        logging.info("Data sorted chronologically.")
    data.dropna(inplace=True)
    logging.info(f"Columns: {list(data.columns)}")
    logging.info(f"Data head:\n{data.head()}")
    for col in ['open_time', 'close_time']:
        data[col] = data[col].dt.tz_localize(None)
    logging.info("Removed timezone from 'open_time' and 'close_time'.")
    if not pd.api.types.is_datetime64_any_dtype(data['open_time']):
        data['open_time'] = pd.to_datetime(data['open_time'], errors='coerce')
    try:
        data['TimeDiff'] = data['open_time'].diff().dt.total_seconds()
        if data['TimeDiff'].isna().all():
            logging.error("All TimeDiffs NaN. Exiting.")
            sys.exit(1)
        logging.info(f"TimeDiffs head:\n{data['TimeDiff'].head()}")
        data['TimeDiff'].fillna(0, inplace=True)
    except Exception as e:
        logging.error(f"TimeDiff computation failed: {e}")
        sys.exit(1)
    return data, is_rev, os.path.basename(db_path)

if __name__=="__main__":
    symbol = input("Enter symbol (e.g., 'BTCUSDT'): ").strip().upper()
    timeframe = input("Enter timeframe (e.g., '1d'): ").strip()
    df, is_rev, filename = load_data(symbol, timeframe)
