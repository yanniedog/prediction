# correlation_database.py

import sqlite3
from typing import List

class CorrelationDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path)
        self.create_tables()

    def create_tables(self) -> None:
        """
        Create necessary tables if they do not exist.
        """
        create_table_queries = [
            """
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS timeframes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timeframe TEXT UNIQUE
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE
            );
            """,
            """
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
        ]
        cursor = self.connection.cursor()
        for query in create_table_queries:
            cursor.execute(query)
        self.connection.commit()

    def insert_correlation(self, symbol: str, timeframe: str, indicator_name: str, lag: int, correlation_value: float) -> None:
        """
        Inserts or updates a correlation value into the database.

        Parameters:
        - symbol: Trading symbol (e.g., 'SOLUSDT').
        - timeframe: Timeframe interval (e.g., '1w').
        - indicator_name: Name of the indicator.
        - lag: Lag value.
        - correlation_value: Correlation coefficient.
        """
        cursor = self.connection.cursor()
        cursor.execute("INSERT OR IGNORE INTO symbols (symbol) VALUES (?)", (symbol,))
        cursor.execute("SELECT id FROM symbols WHERE symbol = ?", (symbol,))
        symbol_id = cursor.fetchone()[0]
        
        cursor.execute("INSERT OR IGNORE INTO timeframes (timeframe) VALUES (?)", (timeframe,))
        cursor.execute("SELECT id FROM timeframes WHERE timeframe = ?", (timeframe,))
        timeframe_id = cursor.fetchone()[0]
        
        cursor.execute("INSERT OR IGNORE INTO indicators (name) VALUES (?)", (indicator_name,))
        cursor.execute("SELECT id FROM indicators WHERE name = ?", (indicator_name,))
        indicator_id = cursor.fetchone()[0]
        
        cursor.execute("""
            INSERT OR REPLACE INTO correlations 
            (symbol_id, timeframe_id, indicator_id, lag, correlation_value) 
            VALUES (?, ?, ?, ?, ?)
        """, (symbol_id, timeframe_id, indicator_id, lag, correlation_value))
        
        self.connection.commit()

    def get_correlations(self, symbol: str, timeframe: str, indicator_name: str, max_lag: int) -> List[float]:
        """
        Retrieves all correlation values for a given symbol, timeframe, and indicator up to max_lag.

        Parameters:
        - symbol: Trading symbol (e.g., 'SOLUSDT').
        - timeframe: Timeframe interval (e.g., '1w').
        - indicator_name: Name of the indicator.
        - max_lag: Maximum lag to retrieve.

        Returns:
        - List of correlation values ordered by lag ascending.
        """
        cursor = self.connection.cursor()
        
        cursor.execute("SELECT id FROM symbols WHERE symbol = ?", (symbol,))
        symbol_row = cursor.fetchone()
        if not symbol_row:
            return []
        symbol_id = symbol_row[0]
        
        cursor.execute("SELECT id FROM timeframes WHERE timeframe = ?", (timeframe,))
        timeframe_row = cursor.fetchone()
        if not timeframe_row:
            return []
        timeframe_id = timeframe_row[0]
        
        cursor.execute("SELECT id FROM indicators WHERE name = ?", (indicator_name,))
        indicator_row = cursor.fetchone()
        if not indicator_row:
            return []
        indicator_id = indicator_row[0]
        
        cursor.execute("""
            SELECT correlation_value 
            FROM correlations 
            WHERE symbol_id = ? AND timeframe_id = ? AND indicator_id = ? AND lag BETWEEN 1 AND ?
            ORDER BY lag ASC
        """, (symbol_id, timeframe_id, indicator_id, max_lag))
        
        rows = cursor.fetchall()
        return [row[0] for row in rows]

    def close(self) -> None:
        """
        Closes the database connection.
        """
        self.connection.close()