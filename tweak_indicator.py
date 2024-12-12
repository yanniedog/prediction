# tweak_indicator.py
import sys, sqlite3, itertools
from pathlib import Path
import pandas as pd
import numpy as np
from indicators import compute_all_indicators
from config import DB_PATH
from sqlite_data_manager import initialize_database, insert_indicator_configs, create_connection, create_tables

def fetch_available_indicators():
    dummy_data = pd.DataFrame({
        'open': np.random.random(200) * 100,
        'high': np.random.random(200) * 100,
        'low': np.random.random(200) * 100,
        'close': np.random.random(200) * 100,
        'volume': np.random.randint(1, 1000, 200)
    })
    try:
        # Compute all default indicators
        processed_data = compute_all_indicators(dummy_data)
        # Now, extract indicator names
        return sorted(processed_data.columns)
    except Exception as e:
        print(f"Error fetching indicators: {e}")
        return []

def generate_configurations(params, defaults):
    return [
        dict(zip(params, values))
        for values in itertools.product(*[range(max(1, defaults[p] - 5), defaults[p] + 5) for p in params])
    ]

def insert_tweaked_configs(indicator_name, configurations):
    initialize_database(DB_PATH)  # Ensure tables are created
    conn = create_connection(DB_PATH)
    if not conn:
        print("Database connection failed.")
        sys.exit(1)
    create_tables(conn)  # Ensure tables are present
    try:
        insert_indicator_configs(conn, indicator_name, configurations)
    except Exception as e:
        print(f"Error inserting configurations: {e}")
    finally:
        conn.close()

def main():
    if len(sys.argv) < 3:
        print("Usage: python tweak_indicator.py <SYMBOL> <TIMEFRAME>")
        sys.exit(1)
    symbol, timeframe = sys.argv[1:3]
    initialize_database(DB_PATH)  # Ensure database is initialized
    indicators = fetch_available_indicators()
    if not indicators:
        print("No indicators available. Check `indicators.py` or `compute_all_indicators`.")
        sys.exit(1)
    print("Available indicators:")
    for idx, indicator in enumerate(indicators, 1):
        print(f"{idx}. {indicator}")
    choice = input("Select an indicator by number: ").strip()
    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(indicators):
        print("Invalid choice.")
        sys.exit(1)
    selected_indicator = indicators[int(choice) - 1]
    print(f"Selected indicator: {selected_indicator}")
    default_params = {"timeperiod": 14}
    configurations = generate_configurations(default_params.keys(), default_params)
    insert_tweaked_configs(selected_indicator, configurations)
    print(f"Configurations for {selected_indicator} added to the database.")

if __name__ == "__main__":
    main()
