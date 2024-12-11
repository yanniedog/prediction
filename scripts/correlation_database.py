# correlation_database.py
import sqlite3

class CorrelationDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path)
        self.create_table()

    def create_table(self) -> None:
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
        index_query = "CREATE INDEX IF NOT EXISTS idx_correlations_symbol_timeframe_indicator_lag ON correlations (symbol_id, timeframe_id, indicator_id, lag);"
        c = self.connection.cursor()
        c.execute(create_table_query)
        c.execute(index_query)
        self.connection.commit()

    def insert_correlation(self, symbol: str, timeframe: str, indicator_name: str, lag: int, correlation_value: float) -> None:
        c = self.connection.cursor()
        symbol_id = self._get_or_insert(c, "symbols", "symbol", symbol)
        timeframe_id = self._get_or_insert(c, "timeframes", "timeframe", timeframe)
        indicator_id = self._get_or_insert(c, "indicators", "name", indicator_name)
        insert_query = """INSERT OR REPLACE INTO correlations (symbol_id, timeframe_id, indicator_id, lag, correlation_value) VALUES (?, ?, ?, ?, ?);"""
        c.execute(insert_query, (symbol_id, timeframe_id, indicator_id, lag, correlation_value))
        self.connection.commit()

    def get_correlation(self, symbol: str, timeframe: str, indicator_name: str, lag: int):
        c = self.connection.cursor()
        symbol_id = self._get_id(c, "symbols", "symbol", symbol)
        timeframe_id = self._get_id(c, "timeframes", "timeframe", timeframe)
        indicator_id = self._get_id(c, "indicators", "name", indicator_name)
        if symbol_id is None or timeframe_id is None or indicator_id is None:
            return None
        c.execute("SELECT correlation_value FROM correlations WHERE symbol_id=? AND timeframe_id=? AND indicator_id=? AND lag=?",(symbol_id, timeframe_id, indicator_id, lag))
        r = c.fetchone()
        return r[0] if r else None

    def get_all_correlations(self):
        q = """SELECT symbols.symbol, timeframes.timeframe, indicators.name, correlations.lag, correlations.correlation_value
        FROM correlations
        JOIN symbols ON correlations.symbol_id = symbols.id
        JOIN timeframes ON correlations.timeframe_id = timeframes.id
        JOIN indicators ON correlations.indicator_id = indicators.id;"""
        c = self.connection.cursor()
        c.execute(q)
        return c.fetchall()

    def close(self) -> None:
        self.connection.close()

    def _get_or_insert(self, c, table, column, value):
        c.execute(f"SELECT id FROM {table} WHERE {column} = ?", (value,))
        r = c.fetchone()
        if r:
            return r[0]
        c.execute(f"INSERT INTO {table} ({column}) VALUES (?)", (value,))
        return c.lastrowid

    def _get_id(self, c, table, column, value):
        c.execute(f"SELECT id FROM {table} WHERE {column} = ?", (value,))
        r = c.fetchone()
        return r[0] if r else None
