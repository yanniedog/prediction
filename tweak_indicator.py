# tweak_indicator.py
import sys, sqlite3, itertools
from pathlib import Path
import pandas as pd
from indicators import compute_all_indicators
from config import DB_PATH

def create_connection(db_file):
    try:
        return sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(f"SQLite connection error: {e}")
        return None

def fetch_available_indicators():
    dummy_data = pd.DataFrame({
        'open': [1, 2, 3, 4, 5],
        'high': [2, 3, 4, 5, 6],
        'low': [1, 1.5, 2, 2.5, 3],
        'close': [2, 2.5, 3, 3.5, 4],
        'volume': [100, 150, 200, 250, 300]
    })
    try:
        processed_data = compute_all_indicators(dummy_data)
        if isinstance(processed_data, pd.DataFrame):
            return sorted(processed_data.columns)
        else:
            raise ValueError("`compute_all_indicators` did not return a DataFrame.")
    except Exception as e:
        print(f"Error fetching indicators: {e}")
        return []

def generate_configurations(params, defaults):
    return [
        dict(zip(params, values))
        for values in itertools.product(*[range(max(1, defaults[p] - 5), defaults[p] + 5) for p in params])
    ]

def insert_tweaked_configs(indicator_name, configurations):
    conn = create_connection(DB_PATH)
    if not conn:
        print("Database connection failed.")
        sys.exit(1)
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO indicators (name) VALUES (?)", (indicator_name,))
    for config in configurations:
        config_str = f"{indicator_name}_" + "_".join(f"{k}{v}" for k, v in config.items())
        cursor.execute("INSERT OR IGNORE INTO indicators (name) VALUES (?)", (config_str,))
    conn.commit()
    conn.close()

def main():
    if len(sys.argv) < 3:
        print("Usage: python tweak-indicator.py <SYMBOL> <TIMEFRAME>")
        sys.exit(1)

    symbol, timeframe = sys.argv[1:3]
    indicators = fetch_available_indicators()
    if not indicators:
        print("No indicators available. Check `indicators.py` or `compute_all_indicators`.")
        sys.exit(1)

    print("Available indicators:")
    for idx, indicator in enumerate(indicators, 1):
        print(f"{idx}. {indicator}")

    choice = int(input("Select an indicator by number: ")) - 1
    if choice < 0 or choice >= len(indicators):
        print("Invalid choice.")
        sys.exit(1)

    selected_indicator = indicators[choice]
    print(f"Selected indicator: {selected_indicator}")

    default_params = {"timeperiod": 14}
    configurations = generate_configurations(default_params.keys(), default_params)
    insert_tweaked_configs(selected_indicator, configurations)
    print(f"Configurations for {selected_indicator} added to the database.")

if __name__ == "__main__":
    main()
