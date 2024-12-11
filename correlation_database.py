# Filename: correlation_database.py

import sqlite3
import logging
from typing import Optional, List, Tuple

class CorrelationDatabase:
    def __init__(self, db_path: str):
        """
        Initializes the CorrelationDatabase instance.

        Args:
            db_path (str): Path to the SQLite database file.
        """
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path)
        self.create_table()

    def create_table(self) -> None:
        """
        Creates the 'correlations' table if it does not exist.
        """
        create_table_query = """
        CREATE TABLE IF NOT EXISTS correlations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol_id INTEGER NOT NULL,
            timeframe_id INTEGER NOT NULL,
            indicator_id INTEGER NOT NULL,
            lag INTEGER NOT NULL,
            correlation_value REAL NOT NULL,
            FOREIGN KEY (symbol_id) REFERENCES symbols(id),
            FOREIGN KEY (timeframe_id) REFERENCES timeframes(id),
            FOREIGN KEY (indicator_id) REFERENCES indicators(id),
            UNIQUE(symbol_id, timeframe_id, indicator_id, lag)
        );
        """
        create_index_queries = [
            "CREATE INDEX IF NOT EXISTS idx_correlations_symbol_timeframe_indicator_lag ON correlations (symbol_id, timeframe_id, indicator_id, lag);"
        ]
        cursor = self.connection.cursor()
        cursor.execute(create_table_query)
        for query in create_index_queries:
            cursor.execute(query)
        self.connection.commit()

    def insert_correlation(self, symbol: str, timeframe: str, indicator_name: str, lag: int, correlation_value: float) -> None:
        """
        Inserts a correlation record into the database.

        Args:
            symbol (str): Trading symbol.
            timeframe (str): Timeframe of the data.
            indicator_name (str): Name of the indicator.
            lag (int): Lag period.
            correlation_value (float): Correlation value.
        """
        cursor = self.connection.cursor()

        # Get or insert symbol_id
        cursor.execute("SELECT id FROM symbols WHERE symbol = ?", (symbol,))
        result = cursor.fetchone()
        if result:
            symbol_id = result[0]
        else:
            cursor.execute("INSERT INTO symbols (symbol) VALUES (?)", (symbol,))
            symbol_id = cursor.lastrowid

        # Get or insert timeframe_id
        cursor.execute("SELECT id FROM timeframes WHERE timeframe = ?", (timeframe,))
        result = cursor.fetchone()
        if result:
            timeframe_id = result[0]
        else:
            cursor.execute("INSERT INTO timeframes (timeframe) VALUES (?)", (timeframe,))
            timeframe_id = cursor.lastrowid

        # Get or insert indicator_id
        cursor.execute("SELECT id FROM indicators WHERE name = ?", (indicator_name,))
        result = cursor.fetchone()
        if result:
            indicator_id = result[0]
        else:
            cursor.execute("INSERT INTO indicators (name) VALUES (?)", (indicator_name,))
            indicator_id = cursor.lastrowid

        # Insert or replace correlation record
        insert_query = """
        INSERT OR REPLACE INTO correlations (
            symbol_id, timeframe_id, indicator_id, lag, correlation_value
        ) VALUES (?, ?, ?, ?, ?);
        """
        try:
            cursor.execute(insert_query, (symbol_id, timeframe_id, indicator_id, lag, correlation_value))
            self.connection.commit()
            logging.debug(f"Inserted correlation: Symbol: {symbol}, Timeframe: {timeframe}, Indicator: {indicator_name}, Lag: {lag}, Correlation: {correlation_value}")
        except sqlite3.Error as e:
            logging.error(f"Failed to insert correlation: {e}")

    def get_correlation(self, symbol: str, timeframe: str, indicator_name: str, lag: int) -> Optional[float]:
        """
        Retrieves a correlation value for a specific symbol, timeframe, indicator, and lag.

        Args:
            symbol (str): Trading symbol.
            timeframe (str): Timeframe of the data.
            indicator_name (str): Name of the indicator.
            lag (int): Lag period.

        Returns:
            Optional[float]: Correlation value if found, else None.
        """
        cursor = self.connection.cursor()

        # Get IDs
        cursor.execute("SELECT id FROM symbols WHERE symbol = ?", (symbol,))
        result = cursor.fetchone()
        if result:
            symbol_id = result[0]
        else:
            return None  # Symbol not found

        cursor.execute("SELECT id FROM timeframes WHERE timeframe = ?", (timeframe,))
        result = cursor.fetchone()
        if result:
            timeframe_id = result[0]
        else:
            return None  # Timeframe not found

        cursor.execute("SELECT id FROM indicators WHERE name = ?", (indicator_name,))
        result = cursor.fetchone()
        if result:
            indicator_id = result[0]
        else:
            return None  # Indicator not found

        # Fetch correlation
        select_query = """
        SELECT correlation_value FROM correlations
        WHERE symbol_id = ? AND timeframe_id = ? AND indicator_id = ? AND lag = ?;
        """
        cursor.execute(select_query, (symbol_id, timeframe_id, indicator_id, lag))
        result = cursor.fetchone()
        return result[0] if result else None

    def get_all_correlations(self) -> List[Tuple[str, str, str, int, float]]:
        """
        Retrieves all correlation records from the database.

        Returns:
            List[Tuple[str, str, str, int, float]]: List of correlation records.
        """
        select_query = """
        SELECT symbols.symbol, timeframes.timeframe, indicators.name, correlations.lag, correlations.correlation_value
        FROM correlations
        JOIN symbols ON correlations.symbol_id = symbols.id
        JOIN timeframes ON correlations.timeframe_id = timeframes.id
        JOIN indicators ON correlations.indicator_id = indicators.id;
        """
        cursor = self.connection.cursor()
        cursor.execute(select_query)
        return cursor.fetchall()

    def close(self) -> None:
        """
        Closes the database connection.
        """
        self.connection.close()