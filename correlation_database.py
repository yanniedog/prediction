# correlation_database.py
import logging
import sqlite3

logger = logging.getLogger()
class CorrelationDatabase:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        self.conn.execute("""
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
        );""")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_correlations ON correlations (symbol_id, timeframe_id, indicator_id, lag);")
        self.conn.commit()

    def insert_correlation(self, symbol, timeframe, indicator, lag, value):
        cursor = self.conn.cursor()
        symbol_id = self._get_or_create_id('symbols', 'symbol', symbol, cursor)
        timeframe_id = self._get_or_create_id('timeframes', 'timeframe', timeframe, cursor)
        indicator_id = self._get_or_create_id('indicators', 'name', indicator, cursor)
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO correlations (symbol_id, timeframe_id, indicator_id, lag, correlation_value)
                VALUES (?, ?, ?, ?, ?);""", (symbol_id, timeframe_id, indicator_id, lag, value))
            self.conn.commit()
        except sqlite3.Error:
            pass

    def get_correlation(self, symbol, timeframe, indicator, lag):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT c.correlation_value FROM correlations c
            JOIN symbols s ON c.symbol_id = s.id
            JOIN timeframes t ON c.timeframe_id = t.id
            JOIN indicators i ON c.indicator_id = i.id
            WHERE s.symbol = ? AND t.timeframe = ? AND i.name = ? AND c.lag = ?;""",
            (symbol, timeframe, indicator, lag))
        res = cursor.fetchone()
        return res[0] if res else None

    def get_all_correlations(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT s.symbol, t.timeframe, i.name, c.lag, c.correlation_value
            FROM correlations c
            JOIN symbols s ON c.symbol_id = s.id
            JOIN timeframes t ON c.timeframe_id = t.id
            JOIN indicators i ON c.indicator_id = i.id;""")
        return cursor.fetchall()

    def _get_or_create_id(self, table, column, value, cursor):
        cursor.execute(f"SELECT id FROM {table} WHERE {column} = ?", (value,))
        res = cursor.fetchone()
        if res: return res[0]
        cursor.execute(f"INSERT INTO {table} ({column}) VALUES (?)", (value,))
        return cursor.lastrowid

    def close(self):
        self.conn.close()
