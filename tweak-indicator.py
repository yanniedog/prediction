import sys
import sqlite3
import itertools
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
    from indicators import compute_all_indicators
    indicator_funcs = [func for func in dir(compute_all_indicators) if not func.startswith("_")]
    return sorted(indicator_funcs)

def generate_configurations(params, default_values):
    ranges = {
        param: range(max(1, default_values.get(param, 1) - 5), default_values.get(param, 1) + 5)
        for param in params
    }
    return list(itertools.product(*ranges.values()))

def insert_tweaked_configs(indicator_name, configurations):
    conn = create_connection(DB_PATH)
    if conn is None:
        print("Database connection failed.")
        sys.exit(1)

    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO indicators (name) VALUES (?)", (indicator_name,))
    for config in configurations:
        config_str = f"{indicator_name}_" + "_".join([f"{key}{val}" for key, val in config.items()])
        cursor.execute("INSERT OR IGNORE INTO indicators (name) VALUES (?)", (config_str,))

    conn.commit()
    conn.close()

def main():
    if len(sys.argv) < 3:
        print("Usage: python tweak-indicator.py <SYMBOL> <TIMEFRAME>")
        sys.exit(1)

    symbol, timeframe = sys.argv[1:3]

    indicators = fetch_available_indicators()
    print("Available indicators:")
    for idx, indicator in enumerate(indicators, start=1):
        print(f"{idx}. {indicator}")

    choice = int(input("Select an indicator by number: ")) - 1
    if choice < 0 or choice >= len(indicators):
        print("Invalid choice.")
        sys.exit(1)

    selected_indicator = indicators[choice]
    print(f"Selected indicator: {selected_indicator}")

    default_params = {
        "timeperiod": 14,  # Example default parameter, customize this as needed
    }

    configurations = generate_configurations(default_params.keys(), default_params)
    insert_tweaked_configs(selected_indicator, configurations)
    print(f"Configurations for {selected_indicator} added to the database.")

if __name__ == "__main__":
    main()
